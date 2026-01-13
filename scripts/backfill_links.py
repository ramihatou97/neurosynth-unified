#!/usr/bin/env python3
"""
NeuroSynth - Backfill Chunk-Image Links
========================================

Re-runs TriPassLinker on existing documents where embeddings exist
but chunk-image links are missing or incomplete.

This fixes the pipeline ordering bug where linking happened BEFORE
embeddings were generated, causing Pass 3 (semantic similarity) to fail.

Usage:
    cd <project-root>
    source venv/bin/activate
    DATABASE_URL=$DATABASE_URL python scripts/backfill_links.py

    # For a specific document:
    python scripts/backfill_links.py --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571

    # Dry run (preview only):
    python scripts/backfill_links.py --dry-run
"""

import asyncio
import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# LIGHTWEIGHT MODELS FOR LINKING
# =============================================================================

@dataclass
class ChunkForLinking:
    """Minimal chunk representation for TriPassLinker."""
    id: str
    content: str
    page_start: int
    page_end: int
    cuis: List[str] = field(default_factory=list)
    text_embedding: Optional[np.ndarray] = None
    figure_refs: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_db_row(cls, row: dict) -> "ChunkForLinking":
        """Create from database row."""
        embedding = None
        raw_emb = row.get('text_embedding')
        if raw_emb is None:
            raw_emb = row.get('embedding')
        if raw_emb is not None and (isinstance(raw_emb, (list, np.ndarray)) and len(raw_emb) > 0):
            embedding = np.array(raw_emb)

        # Extract figure references from content
        import re
        figure_refs = re.findall(
            r"(?i)(?:figure|fig\.?)\s*(\d+(?:\.\d+)?[a-z]?)",
            row.get('content', '')
        )

        return cls(
            id=str(row['id']),
            content=row.get('content', ''),
            page_start=row.get('page_number') or row.get('page_start') or 0,
            page_end=row.get('page_number') or row.get('page_end') or 0,
            cuis=row.get('cuis') or [],
            text_embedding=embedding,
            figure_refs=figure_refs
        )


@dataclass
class ImageForLinking:
    """Minimal image representation for TriPassLinker."""
    id: str
    page_number: int
    vlm_caption: Optional[str] = None
    caption_embedding: Optional[np.ndarray] = None
    cuis: List[str] = field(default_factory=list)
    figure_id: Optional[str] = None
    is_decorative: bool = False
    chunk_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_db_row(cls, row: dict) -> "ImageForLinking":
        """Create from database row."""
        embedding = None
        raw_emb = row.get('caption_embedding')
        if raw_emb is not None and (isinstance(raw_emb, (list, np.ndarray)) and len(raw_emb) > 0):
            embedding = np.array(raw_emb)

        # Try to extract figure ID from caption
        figure_id = None
        if row.get('vlm_caption'):
            import re
            match = re.search(
                r"(?i)(?:figure|fig\.?)\s*(\d+(?:\.\d+)?[a-z]?)",
                row['vlm_caption']
            )
            if match:
                figure_id = f"figure {match.group(1)}"

        # Also check metadata
        metadata = row.get('metadata') or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        if not figure_id and metadata.get('figure_id'):
            figure_id = metadata['figure_id']

        return cls(
            id=str(row['id']),
            page_number=row.get('page_number') or 0,
            vlm_caption=row.get('vlm_caption'),
            caption_embedding=embedding,
            cuis=row.get('cuis') or [],
            figure_id=figure_id,
            is_decorative=row.get('is_decorative') or False
        )


@dataclass
class LinkResult:
    """Result of linking a chunk to an image."""
    chunk_id: str
    image_id: str
    strength: float
    match_type: str
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SIMPLIFIED TRIPASS LINKER
# =============================================================================

class BackfillTriPassLinker:
    """
    Simplified TriPassLinker for backfill operations.
    
    Pass 1: Deterministic regex match ("Fig 6.3" → image.figure_id)
    Pass 2: UMLS CUI Jaccard overlap (threshold 0.25)
    Pass 3: Embedding cosine similarity (threshold 0.55)
    
    Fusion: (semantic × 0.55) + (cui × 0.45) ≥ 0.55
    """

    FIGURE_PATTERN = None  # Compiled on first use

    def __init__(
        self,
        semantic_threshold: float = 0.55,
        cui_threshold: float = 0.25,
        semantic_weight: float = 0.55,
        cui_weight: float = 0.45,
        page_buffer: int = 2,
    ):
        self.semantic_threshold = semantic_threshold
        self.cui_threshold = cui_threshold
        self.semantic_weight = semantic_weight
        self.cui_weight = cui_weight
        self.page_buffer = page_buffer

        if BackfillTriPassLinker.FIGURE_PATTERN is None:
            import re
            BackfillTriPassLinker.FIGURE_PATTERN = re.compile(
                r"(?i)(?:figure|fig\.?)\s*(\d+(?:\.\d+)?[a-z]?(?:\s*[-–,]\s*\d+(?:\.\d+)?[a-z]?)*)",
            )

    def link(
        self,
        chunks: List[ChunkForLinking],
        images: List[ImageForLinking]
    ) -> List[LinkResult]:
        """Run tri-pass linking algorithm."""
        links: List[LinkResult] = []

        # Build figure reference map
        figure_map = self._build_figure_map(images)

        for image in images:
            if image.is_decorative:
                continue

            # Get candidate chunks within page buffer
            candidates = self._get_candidate_chunks(chunks, image)

            if not candidates:
                continue

            for chunk in candidates:
                result = self._calculate_link(chunk, image, figure_map)

                if result:
                    links.append(result)

                    # Track bidirectional refs
                    if image.id not in chunk.image_ids:
                        chunk.image_ids.append(image.id)
                    if chunk.id not in image.chunk_ids:
                        image.chunk_ids.append(chunk.id)

        # Log summary
        deterministic = sum(1 for l in links if l.match_type == 'deterministic')
        fusion_sem = sum(1 for l in links if l.match_type == 'fusion_semantic')
        fusion_cui = sum(1 for l in links if l.match_type == 'fusion_cui')
        cui_only = sum(1 for l in links if l.match_type == 'cui_only')

        logger.info(
            f"TriPassLinker: Created {len(links)} links "
            f"({deterministic} deterministic, {fusion_sem} fusion-semantic, "
            f"{fusion_cui} fusion-cui, {cui_only} cui-only)"
        )

        return links

    def _build_figure_map(self, images: List[ImageForLinking]) -> Dict[str, ImageForLinking]:
        """Build map of normalized figure IDs to images."""
        figure_map = {}
        for image in images:
            if image.figure_id:
                normalized = self._normalize_figure_ref(image.figure_id)
                figure_map[normalized] = image
                figure_map[image.figure_id.lower().strip()] = image
        return figure_map

    def _normalize_figure_ref(self, ref: str) -> str:
        """Normalize figure reference for matching."""
        import re
        ref = ref.lower().strip()
        ref = re.sub(r"^fig\.?\s*", "figure ", ref)
        ref = ref.replace("figure _", "figure ")
        ref = re.sub(r"\s+", " ", ref)
        return ref

    def _get_candidate_chunks(
        self,
        chunks: List[ChunkForLinking],
        image: ImageForLinking
    ) -> List[ChunkForLinking]:
        """Get chunks within page buffer of image."""
        return [
            c for c in chunks
            if abs(c.page_start - image.page_number) <= self.page_buffer
            or abs(c.page_end - image.page_number) <= self.page_buffer
        ]

    def _calculate_link(
        self,
        chunk: ChunkForLinking,
        image: ImageForLinking,
        figure_map: Dict[str, ImageForLinking]
    ) -> Optional[LinkResult]:
        """Calculate link using tri-pass algorithm."""

        # PASS 1: Deterministic regex match
        if self._check_deterministic_match(chunk, image, figure_map):
            return LinkResult(
                chunk_id=chunk.id,
                image_id=image.id,
                strength=1.0,
                match_type='deterministic',
                details={"pass": 1, "method": "figure_reference"}
            )

        # PASS 2: CUI overlap
        cui_score = 0.0
        if chunk.cuis and image.cuis:
            cui_score = self._calculate_cui_overlap(set(chunk.cuis), set(image.cuis))

        # PASS 3: Semantic similarity
        sem_score = 0.0
        if chunk.text_embedding is not None and image.caption_embedding is not None:
            sem_score = self._calculate_cosine_similarity(
                chunk.text_embedding,
                image.caption_embedding
            )

        # Scoring logic
        # If both scores available, use weighted fusion
        # If only semantic, use semantic score directly
        # If only CUI, use CUI score
        if sem_score > 0 or cui_score > 0:
            # Calculate effective score based on what's available
            if cui_score > 0 and sem_score > 0:
                # Both available - use fusion
                final_score = (sem_score * self.semantic_weight) + (cui_score * self.cui_weight)
                match_type = 'fusion_semantic' if sem_score > cui_score else 'fusion_cui'
            elif sem_score > 0:
                # Only semantic available - use raw score (no penalty for missing CUIs)
                final_score = sem_score
                match_type = 'semantic'
            else:
                # Only CUI available
                final_score = cui_score * 0.8  # Slight penalty for CUI-only
                match_type = 'cui_only'

            # Check threshold
            if final_score >= self.semantic_threshold:
                return LinkResult(
                    chunk_id=chunk.id,
                    image_id=image.id,
                    strength=final_score,
                    match_type=match_type,
                    details={
                        "pass": 3,
                        "semantic_score": round(sem_score, 4),
                        "cui_score": round(cui_score, 4),
                        "final_score": round(final_score, 4),
                    }
                )

        return None

    def _check_deterministic_match(
        self,
        chunk: ChunkForLinking,
        image: ImageForLinking,
        figure_map: Dict[str, ImageForLinking]
    ) -> bool:
        """Check if chunk explicitly references this image."""
        if not chunk.figure_refs:
            return False

        for ref in chunk.figure_refs:
            normalized = self._normalize_figure_ref(f"figure {ref}")
            matched_image = figure_map.get(normalized)
            if matched_image and matched_image.id == image.id:
                return True

        return False

    def _calculate_cui_overlap(self, cuis_a: set, cuis_b: set) -> float:
        """Jaccard similarity between CUI sets."""
        if not cuis_a or not cuis_b:
            return 0.0
        intersection = len(cuis_a & cuis_b)
        union = len(cuis_a | cuis_b)
        return intersection / union if union > 0 else 0.0

    def _calculate_cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray
    ) -> float:
        """Cosine similarity between embeddings."""
        if vec_a is None or vec_b is None:
            return 0.0
        
        # Handle dimension mismatch (1024d text vs 512d image)
        if len(vec_a) != len(vec_b):
            logger.debug(f"Dimension mismatch: {len(vec_a)} vs {len(vec_b)}")
            return 0.0

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def get_documents_for_backfill(
    db,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get documents that need link backfill."""
    
    if document_id:
        # Specific document
        query = """
            SELECT 
                d.id,
                d.title,
                (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) as chunk_count,
                (SELECT COUNT(*) FROM images i WHERE i.document_id = d.id AND NOT i.is_decorative) as image_count,
                (SELECT COUNT(*) FROM links l 
                 JOIN chunks c ON l.chunk_id = c.id 
                 WHERE c.document_id = d.id) as link_count
            FROM documents d
            WHERE d.id = $1::uuid
        """
        rows = await db.fetch(query, document_id)
    else:
        # All documents with chunks and images but few/no links
        query = """
            SELECT 
                d.id,
                d.title,
                (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) as chunk_count,
                (SELECT COUNT(*) FROM images i WHERE i.document_id = d.id AND NOT i.is_decorative) as image_count,
                (SELECT COUNT(*) FROM links l 
                 JOIN chunks c ON l.chunk_id = c.id 
                 WHERE c.document_id = d.id) as link_count
            FROM documents d
            WHERE EXISTS (
                SELECT 1 FROM chunks c 
                WHERE c.document_id = d.id 
                AND c.text_embedding IS NOT NULL
            )
            AND EXISTS (
                SELECT 1 FROM images i 
                WHERE i.document_id = d.id 
                AND i.caption_embedding IS NOT NULL
                AND NOT i.is_decorative
            )
            ORDER BY link_count ASC
        """
        rows = await db.fetch(query)

    return [dict(row) for row in rows]


async def get_chunks_for_document(db, doc_id: str) -> List[ChunkForLinking]:
    """Fetch chunks with embeddings for linking."""
    query = """
        SELECT
            id, content, page_number, text_embedding,
            COALESCE(metadata->>'figure_refs', '[]') as figure_refs_json
        FROM chunks
        WHERE document_id = $1::uuid
          AND text_embedding IS NOT NULL
        ORDER BY page_number, chunk_index
    """
    rows = await db.fetch(query, doc_id)
    return [ChunkForLinking.from_db_row(dict(row)) for row in rows]


async def get_images_for_document(db, doc_id: str) -> List[ImageForLinking]:
    """Fetch images with caption embeddings for linking."""
    query = """
        SELECT
            id, page_number, vlm_caption, caption_embedding,
            is_decorative, metadata
        FROM images
        WHERE document_id = $1::uuid
          AND caption_embedding IS NOT NULL
          AND (is_decorative IS NULL OR NOT is_decorative)
        ORDER BY page_number
    """
    rows = await db.fetch(query, doc_id)
    return [ImageForLinking.from_db_row(dict(row)) for row in rows]


async def delete_existing_links(db, doc_id: str) -> int:
    """Delete existing links for a document."""
    result = await db.execute("""
        DELETE FROM chunk_image_links
        WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = $1::uuid)
    """, doc_id)
    
    # Parse result like "DELETE 42"
    if result:
        parts = result.split()
        if len(parts) >= 2:
            return int(parts[1])
    return 0


async def insert_links(db, links: List[LinkResult], min_score: float = 0.5) -> int:
    """Insert new links into database."""
    if not links:
        return 0

    inserted = 0
    for link in links:
        if link.strength < min_score:
            continue

        try:
            await db.execute("""
                INSERT INTO chunk_image_links (id, chunk_id, image_id, link_type, relevance_score, link_metadata)
                VALUES ($1, $2::uuid, $3::uuid, $4, $5, $6::jsonb)
                ON CONFLICT (chunk_id, image_id) DO UPDATE SET
                    relevance_score = EXCLUDED.relevance_score,
                    link_type = EXCLUDED.link_type,
                    link_metadata = EXCLUDED.link_metadata
            """,
                uuid4(),
                link.chunk_id,
                link.image_id,
                link.match_type,
                link.strength,
                json.dumps(link.details)
            )
            inserted += 1
        except Exception as e:
            logger.warning(f"Failed to insert link {link.chunk_id} -> {link.image_id}: {e}")

    return inserted


async def refresh_materialized_view(db) -> bool:
    """Refresh top_chunk_links materialized view."""
    try:
        await db.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY top_chunk_links")
        return True
    except Exception as e:
        logger.warning(f"Failed to refresh materialized view: {e}")
        # Try without CONCURRENTLY
        try:
            await db.execute("REFRESH MATERIALIZED VIEW top_chunk_links")
            return True
        except Exception as e2:
            logger.error(f"Materialized view refresh failed: {e2}")
            return False


# =============================================================================
# MAIN BACKFILL LOGIC
# =============================================================================

async def backfill_links(
    connection_string: str,
    document_id: Optional[str] = None,
    dry_run: bool = False,
    min_score: float = 0.5
) -> Dict[str, Any]:
    """
    Backfill chunk-image links for documents.

    Args:
        connection_string: PostgreSQL connection string
        document_id: Specific document ID (optional)
        dry_run: If True, don't modify database
        min_score: Minimum link score to insert

    Returns:
        Statistics dictionary
    """
    from src.database import init_database, close_database

    stats = {
        'documents_processed': 0,
        'total_chunks': 0,
        'total_images': 0,
        'links_deleted': 0,
        'links_created': 0,
        'errors': 0
    }

    logger.info("Connecting to database...")
    db = await init_database(connection_string)

    try:
        # Get documents to process
        documents = await get_documents_for_backfill(db, document_id)
        logger.info(f"Found {len(documents)} documents to process")

        if not documents:
            logger.info("No documents need link backfill")
            return stats

        # Initialize linker
        linker = BackfillTriPassLinker()

        for doc in documents:
            doc_id = str(doc['id'])
            title = doc['title'] or doc_id[:8]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {title}")
            logger.info(f"  Chunks: {doc['chunk_count']}, Images: {doc['image_count']}, Existing Links: {doc['link_count']}")

            try:
                # Fetch chunks and images
                chunks = await get_chunks_for_document(db, doc_id)
                images = await get_images_for_document(db, doc_id)

                stats['total_chunks'] += len(chunks)
                stats['total_images'] += len(images)

                if not chunks or not images:
                    logger.warning(f"  Skipping: No chunks ({len(chunks)}) or images ({len(images)}) with embeddings")
                    continue

                # Log embedding stats
                chunks_with_emb = sum(1 for c in chunks if c.text_embedding is not None)
                images_with_emb = sum(1 for i in images if i.caption_embedding is not None)
                logger.info(f"  Chunks with embedding: {chunks_with_emb}/{len(chunks)}")
                logger.info(f"  Images with caption embedding: {images_with_emb}/{len(images)}")

                # Run linker
                links = linker.link(chunks, images)
                qualified_links = [l for l in links if l.strength >= min_score]

                logger.info(f"  Generated {len(links)} links, {len(qualified_links)} above threshold ({min_score})")

                if dry_run:
                    logger.info("  [DRY RUN] Would delete existing links and insert new ones")
                    stats['links_created'] += len(qualified_links)
                else:
                    # Delete existing links
                    deleted = await delete_existing_links(db, doc_id)
                    stats['links_deleted'] += deleted
                    logger.info(f"  Deleted {deleted} existing links")

                    # Insert new links
                    inserted = await insert_links(db, links, min_score)
                    stats['links_created'] += inserted
                    logger.info(f"  Inserted {inserted} new links")

                stats['documents_processed'] += 1

            except Exception as e:
                logger.error(f"  Error processing document: {e}")
                stats['errors'] += 1
                continue

        # Refresh materialized view
        if not dry_run and stats['links_created'] > 0:
            logger.info("\nRefreshing materialized view...")
            if await refresh_materialized_view(db):
                logger.info("  ✓ Materialized view refreshed")
            else:
                logger.warning("  ✗ Materialized view refresh failed")

        return stats

    finally:
        await close_database()


def print_summary(stats: dict, dry_run: bool = False):
    """Print backfill summary."""
    print()
    print("=" * 60)
    print(f"LINK BACKFILL {'PREVIEW (DRY RUN)' if dry_run else 'COMPLETE'}")
    print("=" * 60)
    print()
    print("Statistics:")
    print(f"  Documents processed:  {stats['documents_processed']}")
    print(f"  Total chunks:         {stats['total_chunks']}")
    print(f"  Total images:         {stats['total_images']}")
    print(f"  Links deleted:        {stats['links_deleted']}")
    print(f"  Links created:        {stats['links_created']}")
    print(f"  Errors:               {stats['errors']}")
    print()

    if stats['links_created'] > 0 and not dry_run:
        print("Next steps:")
        print("  1. Verify links: SELECT COUNT(*) FROM links;")
        print("  2. Rebuild FAISS: POST /api/v1/indexes/rebuild")
        print("  3. Test search with include_images=true")
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill chunk-image links using TriPassLinker"
    )

    parser.add_argument(
        "--database", "-d",
        default=os.getenv(
            "DATABASE_URL",
            "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth"
        ),
        help="PostgreSQL connection string"
    )

    parser.add_argument(
        "--document-id", "-doc",
        default=None,
        help="Process specific document ID only"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying database"
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum link score to insert (default: 0.5)"
    )

    args = parser.parse_args()

    print("NeuroSynth Link Backfill")
    print("=" * 60)
    print(f"Database: {args.database.split('@')[-1] if '@' in args.database else 'configured'}")
    if args.document_id:
        print(f"Document: {args.document_id}")
    print(f"Min Score: {args.min_score}")
    print(f"Dry Run: {args.dry_run}")
    print()

    stats = await backfill_links(
        connection_string=args.database,
        document_id=args.document_id,
        dry_run=args.dry_run,
        min_score=args.min_score
    )

    print_summary(stats, args.dry_run)

    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
