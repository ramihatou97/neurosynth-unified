#!/usr/bin/env python3
"""
NeuroSynth - Backfill Quality Scores
=====================================

Computes and persists quality scores for existing chunks:
- readability_score: How clear and readable the chunk is
- coherence_score: How well sentences connect logically
- completeness_score: Whether chunk is self-contained
- type_specific_score: Score based on chunk type requirements

Usage:
    python scripts/backfill_quality_scores.py --all
    python scripts/backfill_quality_scores.py --document-id <uuid>
    python scripts/backfill_quality_scores.py --dry-run
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def get_chunks_needing_scoring(
    conn,
    document_id: Optional[UUID] = None,
    limit: int = 500
) -> List[Dict]:
    """Find chunks with zero quality scores."""

    query = """
        SELECT id, document_id, content, chunk_type, summary
        FROM chunks
        WHERE (readability_score IS NULL OR readability_score = 0)
          AND content IS NOT NULL
          AND length(content) > 50
    """

    params = []
    if document_id:
        query += " AND document_id = $1"
        params.append(document_id)

    query += f" ORDER BY document_id, chunk_index LIMIT ${len(params) + 1}"
    params.append(limit)

    rows = await conn.fetch(query, *params)
    return [dict(row) for row in rows]


async def update_quality_scores(
    conn,
    chunk_id: UUID,
    readability: float,
    coherence: float,
    completeness: float,
    type_specific: float
) -> bool:
    """Update chunk with quality scores."""
    try:
        await conn.execute(
            """
            UPDATE chunks
            SET readability_score = $1,
                coherence_score = $2,
                completeness_score = $3,
                type_specific_score = $4
            WHERE id = $5
            """,
            readability, coherence, completeness, type_specific, chunk_id
        )
        return True
    except Exception as e:
        logger.error(f"Failed to update chunk {chunk_id}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Backfill quality scores")
    parser.add_argument("--document-id", type=str, help="Specific document UUID")
    parser.add_argument("--all", action="store_true", help="Process all documents")
    parser.add_argument("--limit", type=int, default=1000, help="Max chunks to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if not args.document_id and not args.all:
        parser.error("Must specify --document-id or --all")

    # Import quality scorer
    try:
        from src.core.quality_scorer import get_quality_scorer, ChunkQualityScorer
        from src.shared.models import SemanticChunk
    except ImportError as e:
        logger.error(f"Failed to import quality scorer: {e}")
        return

    import asyncpg
    from dotenv import load_dotenv

    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth")

    conn = await asyncpg.connect(database_url.replace("+asyncpg", ""))
    scorer = get_quality_scorer()

    try:
        doc_id = UUID(args.document_id) if args.document_id else None
        chunks = await get_chunks_needing_scoring(conn, doc_id, args.limit)

        print(f"\nFound {len(chunks)} chunks needing quality scoring")

        if args.dry_run:
            print("\n[DRY RUN] Would score:")
            for chunk in chunks[:10]:
                print(f"  {chunk['id']}: {chunk['chunk_type']} - {len(chunk['content'])} chars")
            if len(chunks) > 10:
                print(f"  ... and {len(chunks) - 10} more")
            return

        # Score chunks
        success = 0
        failed = 0
        score_totals = {'readability': 0, 'coherence': 0, 'completeness': 0, 'type_specific': 0}

        for i, chunk_data in enumerate(chunks):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(chunks)}...")

            try:
                # Create SemanticChunk object for scoring
                chunk = SemanticChunk(
                    id=str(chunk_data['id']),
                    content=chunk_data['content'],
                    chunk_type=chunk_data['chunk_type'] or 'general',
                    summary=chunk_data.get('summary'),
                )

                # Score the chunk
                scorer.score_chunk(chunk)

                # Update database
                if await update_quality_scores(
                    conn,
                    chunk_data['id'],
                    chunk.readability_score,
                    chunk.coherence_score,
                    chunk.completeness_score,
                    chunk.type_specific_score
                ):
                    success += 1
                    score_totals['readability'] += chunk.readability_score
                    score_totals['coherence'] += chunk.coherence_score
                    score_totals['completeness'] += chunk.completeness_score
                    score_totals['type_specific'] += chunk.type_specific_score
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Error scoring chunk {chunk_data['id']}: {e}")
                failed += 1

        print(f"\n{'='*50}")
        print(f"Results: {success} succeeded, {failed} failed")

        if success > 0:
            print(f"\nAverage scores:")
            print(f"  Readability:   {score_totals['readability']/success:.3f}")
            print(f"  Coherence:     {score_totals['coherence']/success:.3f}")
            print(f"  Completeness:  {score_totals['completeness']/success:.3f}")
            print(f"  Type-specific: {score_totals['type_specific']/success:.3f}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
