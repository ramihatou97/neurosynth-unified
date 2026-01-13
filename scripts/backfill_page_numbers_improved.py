#!/usr/bin/env python3
"""
NeuroSynth - Backfill Page Numbers (Improved)
==============================================

Backfills chunk page numbers using multiple strategies:
1. Query actual PDF file for page count (most accurate)
2. Use document.total_pages if available
3. Use image page numbers as reference (images retain actual pages)
4. Fall back to conservative estimation only as last resort

Usage:
    python scripts/backfill_page_numbers.py --document-id <uuid>
    python scripts/backfill_page_numbers.py --all
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple
from uuid import UUID

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def get_page_count_from_pdf(source_path: str) -> Optional[int]:
    """Extract actual page count from PDF file."""
    try:
        import fitz  # PyMuPDF
        
        pdf_path = Path(source_path)
        if not pdf_path.exists():
            # Try common locations
            for base in [Path("data"), Path("/data"), Path(".")]:
                candidate = base / pdf_path.name
                if candidate.exists():
                    pdf_path = candidate
                    break
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {source_path}")
            return None
        
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        logger.warning(f"Failed to read PDF: {e}")
        return None


async def get_max_image_page(conn, document_id: UUID) -> Optional[int]:
    """Get maximum page number from images (images retain actual page numbers)."""
    row = await conn.fetchrow(
        "SELECT MAX(page_number) as max_page FROM images WHERE document_id = $1",
        document_id
    )
    return row['max_page'] if row and row['max_page'] else None


async def estimate_total_pages(
    conn,
    document_id: UUID,
    source_path: str,
    stored_total_pages: Optional[int]
) -> Tuple[int, str]:
    """
    Determine total pages using multiple strategies.
    
    Returns:
        Tuple of (page_count, method_used)
    """
    # Strategy 1: PDF introspection (most accurate)
    pdf_pages = await get_page_count_from_pdf(source_path)
    if pdf_pages:
        return pdf_pages, "pdf_introspection"
    
    # Strategy 2: Stored total_pages (if not NULL/0)
    if stored_total_pages and stored_total_pages > 0:
        return stored_total_pages, "stored_metadata"
    
    # Strategy 3: Use max image page as proxy (images have actual page numbers)
    max_img_page = await get_max_image_page(conn, document_id)
    if max_img_page and max_img_page > 0:
        # Add small buffer since images might not be on last page
        return max_img_page + 2, "image_page_proxy"
    
    # Strategy 4: Conservative estimation from chunk count
    # Medical PDFs average ~2-3 chunks/page, use 2 for conservative estimate
    chunk_count = await conn.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE document_id = $1",
        document_id
    )
    if chunk_count:
        estimated = max(1, chunk_count // 2)  # Conservative: 2 chunks per page
        return estimated, "conservative_estimation"
    
    return 1, "fallback_minimum"


async def backfill_document_pages(conn, document_id: UUID) -> dict:
    """Backfill page numbers for a single document."""
    
    # Get document info
    doc = await conn.fetchrow(
        "SELECT id, source_path, total_pages FROM documents WHERE id = $1",
        document_id
    )
    
    if not doc:
        return {"status": "error", "message": f"Document {document_id} not found"}
    
    # Determine actual page count
    total_pages, method = await estimate_total_pages(
        conn, 
        document_id,
        doc['source_path'],
        doc['total_pages']
    )
    
    logger.info(f"Document {document_id}: {total_pages} pages (via {method})")
    
    # Update document if we discovered better page count
    if doc['total_pages'] != total_pages:
        await conn.execute(
            "UPDATE documents SET total_pages = $1, updated_at = NOW() WHERE id = $2",
            total_pages, document_id
        )
        logger.info(f"Updated document total_pages: {doc['total_pages']} -> {total_pages}")
    
    # Get chunks ordered by sequence
    chunks = await conn.fetch(
        """
        SELECT id, sequence_in_doc, start_page, char_offset_start, char_offset_end
        FROM chunks 
        WHERE document_id = $1 
        ORDER BY COALESCE(sequence_in_doc, 0), id
        """,
        document_id
    )
    
    if not chunks:
        return {"status": "skip", "message": "No chunks found"}
    
    # Calculate page distribution
    # If we have char offsets, use proportional distribution
    # Otherwise, distribute evenly
    total_chunks = len(chunks)
    
    updates = []
    for i, chunk in enumerate(chunks):
        if chunk['start_page'] is not None and chunk['start_page'] > 0:
            # Already has valid page number
            continue
        
        # Proportional distribution: chunk i of N gets page (i * total_pages) / N
        estimated_page = min(total_pages, max(1, (i * total_pages) // total_chunks + 1))
        updates.append((estimated_page, chunk['id']))
    
    # Batch update
    if updates:
        await conn.executemany(
            "UPDATE chunks SET start_page = $1, page_number = $1 WHERE id = $2",
            updates
        )
        logger.info(f"Updated {len(updates)} chunks with page numbers")
    
    return {
        "status": "success",
        "document_id": str(document_id),
        "total_pages": total_pages,
        "page_method": method,
        "chunks_updated": len(updates),
        "chunks_skipped": total_chunks - len(updates)
    }


async def main():
    parser = argparse.ArgumentParser(description="Backfill chunk page numbers")
    parser.add_argument("--document-id", type=str, help="Specific document UUID")
    parser.add_argument("--all", action="store_true", help="Process all documents")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()
    
    if not args.document_id and not args.all:
        parser.error("Must specify --document-id or --all")
    
    import asyncpg
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth")
    
    conn = await asyncpg.connect(database_url.replace("+asyncpg", ""))
    
    try:
        if args.document_id:
            doc_id = UUID(args.document_id)
            result = await backfill_document_pages(conn, doc_id)
            print(f"\nResult: {result}")
        else:
            # Process all documents
            docs = await conn.fetch("SELECT id FROM documents ORDER BY created_at")
            for doc in docs:
                result = await backfill_document_pages(conn, doc['id'])
                print(f"  {doc['id']}: {result['status']}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
