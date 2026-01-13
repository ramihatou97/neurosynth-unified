#!/usr/bin/env python3
"""
NeuroSynth - Extract Figure IDs from Images
============================================

Extracts figure identifiers from image captions and surrounding text.
Enables deterministic linking (Pass 1 of TriPassLinker) which provides
the highest confidence chunk-image connections.

Usage:
    python scripts/extract_figure_ids.py --all
    python scripts/extract_figure_ids.py --document-id <uuid>
    python scripts/extract_figure_ids.py --dry-run
"""

import asyncio
import argparse
import logging
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Figure ID extraction patterns (ordered by specificity)
FIGURE_PATTERNS = [
    # "Figure 6.3A" or "Fig. 6.3a"
    re.compile(r'(?:Figure|Fig\.?)\s*(\d+\.\d+[a-zA-Z]?)', re.I),
    # "Figure 6A" or "Fig 6a"
    re.compile(r'(?:Figure|Fig\.?)\s*(\d+[a-zA-Z])', re.I),
    # "Figure 6" or "Fig. 6"
    re.compile(r'(?:Figure|Fig\.?)\s*(\d+)', re.I),
    # "Plate 3.2"
    re.compile(r'(?:Plate|Panel)\s*(\d+(?:\.\d+)?[a-zA-Z]?)', re.I),
]


def extract_figure_id(caption: str, storage_path: str = None) -> Optional[str]:
    """
    Extract figure ID from caption or filename.

    Returns normalized figure ID like 'fig_6.3' or 'fig_6a'.
    """
    # Try caption first
    if caption:
        for pattern in FIGURE_PATTERNS:
            match = pattern.search(caption)
            if match:
                fig_num = match.group(1).lower()
                return f"fig_{fig_num}"

    # Try filename as fallback
    if storage_path:
        filename = Path(storage_path).stem.lower()
        # Common naming patterns: "figure_6_3", "fig6-3", "f6.3"
        file_patterns = [
            re.compile(r'(?:figure|fig)[_-]?(\d+[_.-]?\d*[a-z]?)', re.I),
            re.compile(r'f(\d+[_.-]?\d*[a-z]?)', re.I),
        ]
        for pattern in file_patterns:
            match = pattern.search(filename)
            if match:
                fig_num = match.group(1).replace('_', '.').replace('-', '.')
                return f"fig_{fig_num}"

    return None


async def get_images_without_figure_id(
    conn,
    document_id: Optional[UUID] = None,
    limit: int = 500
) -> List[Dict]:
    """Find images that need figure ID extraction."""

    query = """
        SELECT
            i.id,
            i.document_id,
            i.vlm_caption,
            i.storage_path,
            i.page_number,
            d.title as document_title
        FROM images i
        JOIN documents d ON i.document_id = d.id
        WHERE (i.figure_id IS NULL OR i.figure_id = '')
          AND i.is_decorative = FALSE
    """

    params = []
    if document_id:
        query += " AND i.document_id = $1"
        params.append(document_id)

    query += f" ORDER BY i.document_id, i.page_number LIMIT ${len(params) + 1}"
    params.append(limit)

    rows = await conn.fetch(query, *params)
    return [dict(row) for row in rows]


async def update_figure_id(conn, image_id: UUID, figure_id: str) -> bool:
    """Update image with extracted figure ID."""
    try:
        await conn.execute(
            """
            UPDATE images
            SET figure_id = $1
            WHERE id = $2
            """,
            figure_id, image_id
        )
        return True
    except Exception as e:
        logger.error(f"Failed to update image {image_id}: {e}")
        return False


def analyze_extraction_results(results: List[Tuple[str, str, str]]) -> Dict:
    """Analyze extraction patterns for reporting."""
    stats = {
        'total': len(results),
        'from_caption': 0,
        'from_filename': 0,
        'no_match': 0,
        'patterns': {}
    }

    for img_id, figure_id, source in results:
        if figure_id:
            if source == 'caption':
                stats['from_caption'] += 1
            else:
                stats['from_filename'] += 1

            # Track pattern distribution
            if '.' in figure_id:
                pattern = 'decimal (6.3)'
            elif any(c.isalpha() for c in figure_id.replace('fig_', '')):
                pattern = 'lettered (6a)'
            else:
                pattern = 'simple (6)'
            stats['patterns'][pattern] = stats['patterns'].get(pattern, 0) + 1
        else:
            stats['no_match'] += 1

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Extract figure IDs from images")
    parser.add_argument("--document-id", type=str, help="Specific document UUID")
    parser.add_argument("--all", action="store_true", help="Process all documents")
    parser.add_argument("--limit", type=int, default=1000, help="Max images to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if not args.document_id and not args.all:
        parser.error("Must specify --document-id or --all")

    import asyncpg
    from dotenv import load_dotenv

    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth")

    conn = await asyncpg.connect(database_url.replace("+asyncpg", ""))

    try:
        doc_id = UUID(args.document_id) if args.document_id else None
        images = await get_images_without_figure_id(conn, doc_id, args.limit)

        print(f"\nFound {len(images)} images without figure IDs")

        if not images:
            print("Nothing to process.")
            return

        # Extract figure IDs
        results = []
        for img in images:
            figure_id = extract_figure_id(img['vlm_caption'], img['storage_path'])
            source = 'caption' if img['vlm_caption'] and figure_id else 'filename'
            results.append((img['id'], figure_id, source))

            if figure_id:
                logger.debug(f"Extracted {figure_id} from {source} for image {img['id']}")

        # Analyze results
        stats = analyze_extraction_results(results)
        print(f"\nExtraction Analysis:")
        print(f"  From caption: {stats['from_caption']}")
        print(f"  From filename: {stats['from_filename']}")
        print(f"  No match: {stats['no_match']}")
        print(f"  Patterns: {stats['patterns']}")

        if args.dry_run:
            print("\n[DRY RUN] Would update:")
            for img, (img_id, figure_id, source) in zip(images[:15], results[:15]):
                if figure_id:
                    caption_preview = (img['vlm_caption'] or '')[:50]
                    print(f"  {figure_id} <- {source}: {caption_preview}...")
            return

        # Update database
        success = 0
        for img_id, figure_id, _ in results:
            if figure_id:
                if await update_figure_id(conn, img_id, figure_id):
                    success += 1

        print(f"\n{'='*50}")
        print(f"Results: {success} figure IDs extracted and saved")
        print(f"Coverage: {success}/{len(images)} ({100*success/len(images):.1f}%)")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
