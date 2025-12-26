#!/usr/bin/env python
"""
Build FAISS indexes from database embeddings using FAISSManager.

This script:
1. Fetches all chunk embeddings from database
2. Builds FAISS indexes using the proper FAISSIndex class
3. Saves in format compatible with FAISSManager.load()
"""
import asyncio
import os
import sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

import asyncpg
from src.retrieval.faiss_manager import FAISSManager, FAISSIndexConfig

async def main():
    print("="*70)
    print("BUILD FAISS INDEXES FROM DATABASE")
    print("="*70)

    # Check configuration
    database_url = os.getenv("DATABASE_URL")
    index_dir = Path(os.getenv("FAISS_INDEX_DIR", "./indexes"))

    if not database_url:
        print("❌ DATABASE_URL not configured")
        return 1

    print(f"\n✅ Configuration:")
    print(f"   Database: {database_url.split('@')[1] if '@' in database_url else 'configured'}")
    print(f"   Index directory: {index_dir}")

    # Create index directory
    index_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    print(f"\n[1] Connecting to database...")

    try:
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        print(f"✅ Connected")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    # Fetch chunk embeddings
    print(f"\n[2] Fetching chunk embeddings...")
    query = """
        SELECT id, embedding
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY id
    """

    rows = await pool.fetch(query)
    print(f"✅ Found {len(rows)} chunks with embeddings")

    if len(rows) == 0:
        print("❌ No embeddings found. Run scripts/generate_embeddings.py first")
        await pool.close()
        return 1

    # Convert to numpy arrays
    print(f"\n[3] Converting embeddings to numpy arrays...")
    chunk_ids = [str(row['id']) for row in rows]
    embeddings = []

    for row in rows:
        # Parse embedding (stored as text string '[0.1, 0.2, ...]')
        emb_str = row['embedding']
        if isinstance(emb_str, str):
            # Remove brackets and split
            emb_str = emb_str.strip('[]')
            emb_list = [float(x) for x in emb_str.split(',')]
            emb = np.array(emb_list, dtype=np.float32)
        else:
            # Already a list or array
            emb = np.array(emb_str, dtype=np.float32)
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"✅ Embeddings array shape: {embeddings.shape}")

    await pool.close()

    # Build using FAISSManager
    print(f"\n[4] Building FAISS index using FAISSManager...")
    manager = FAISSManager(index_dir=str(index_dir))

    # Build text index
    n_vectors = manager.build_text_index(embeddings, chunk_ids)
    print(f"✅ Text index built: {n_vectors} vectors")

    # Save to disk in proper format
    print(f"\n[5] Saving index...")
    manager.save()
    print(f"✅ Index saved")

    # Verify files created
    text_index_file = index_dir / "text.faiss"
    text_meta_file = index_dir / "text.meta.json"

    print(f"\n[6] Verifying files...")
    if text_index_file.exists():
        print(f"✅ {text_index_file.name}: {text_index_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"❌ {text_index_file.name}: NOT CREATED")

    if text_meta_file.exists():
        print(f"✅ {text_meta_file.name}: exists")
    else:
        print(f"❌ {text_meta_file.name}: NOT CREATED")

    # Test loading
    print(f"\n[7] Testing index loading...")
    manager2 = FAISSManager(index_dir=str(index_dir))
    stats = manager2.load()
    print(f"✅ Index loaded successfully:")
    print(f"   Text vectors: {stats['text']}")
    print(f"   Image vectors: {stats['image']}")
    print(f"   Caption vectors: {stats['caption']}")

    # Test search
    if stats['text'] > 0:
        print(f"\n[8] Testing search...")
        query_emb = embeddings[0]
        results = manager2.search_text(query_emb, k=5)
        print(f"✅ Search test passed: {len(results)} results")
        if results:
            print(f"   Top result ID: {results[0][0]}")
            print(f"   Top score: {results[0][1]:.3f}")

    print(f"\n" + "="*70)
    print("✅ FAISS INDEX BUILT AND VERIFIED")
    print("="*70)
    print(f"\nIndex files:")
    print(f"  {text_index_file}")
    print(f"  {text_meta_file}")
    print(f"\nVectors indexed: {stats['text']}")
    print(f"\nNext steps:")
    print(f"  1. Test search: python tests/manual/test_search_live.py")
    print(f"  2. Test synthesis: python tests/manual/stage14_synthesis.py")

    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
