#!/usr/bin/env python3
"""
Final benchmark script for NeuroSynth search performance.

Compares pgvector HNSW search latency and quality.
"""

import asyncio
import os
import time
import statistics
from dotenv import load_dotenv

load_dotenv()


async def run_benchmark():
    import asyncpg
    from src.retrieval import VoyageEmbedder, PostgresVectorSearcher

    print("=" * 60)
    print("NeuroSynth Search Benchmark - pgvector HNSW")
    print("=" * 60)

    # Connect
    pool = await asyncpg.create_pool(
        host='localhost',
        port=5432,
        database='neurosynth',
        user='ramihatoum',
        min_size=2,
        max_size=10
    )

    class DBInterface:
        def __init__(self, pool):
            self.pool = pool
        async def fetch(self, query, *args):
            async with self.pool.acquire() as conn:
                return await conn.fetch(query, *args)

    db = DBInterface(pool)

    # Get database stats
    async with pool.acquire() as conn:
        chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        image_count = await conn.fetchval("SELECT COUNT(*) FROM images WHERE clip_embedding IS NOT NULL")
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")

    print(f"\nDatabase Statistics:")
    print(f"  Documents: {doc_count}")
    print(f"  Chunks with embeddings: {chunk_count}")
    print(f"  Images with embeddings: {image_count}")

    # Check indexes
    async with pool.acquire() as conn:
        indexes = await conn.fetch("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename IN ('chunks', 'images')
            AND indexdef LIKE '%hnsw%'
        """)

    print(f"\nHNSW Indexes:")
    for idx in indexes:
        print(f"  - {idx['indexname']}")

    # Initialize components
    api_key = os.getenv('VOYAGE_API_KEY')
    embedder = VoyageEmbedder(api_key=api_key)
    searcher = PostgresVectorSearcher(database=db, embedder=embedder)

    # Test queries
    test_queries = [
        "pterional craniotomy surgical approach",
        "middle cerebral artery aneurysm clipping",
        "vestibular schwannoma retrosigmoid approach",
        "cavernous sinus anatomy neural structures",
        "endoscopic transsphenoidal pituitary surgery",
    ]

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    all_latencies = []

    for query in test_queries:
        print(f"\nQuery: '{query[:50]}...'")

        # Embed query
        embed_start = time.time()
        query_embedding = await embedder.embed(query)
        embed_time = (time.time() - embed_start) * 1000

        # Run search multiple times
        latencies = []
        for _ in range(5):
            search_start = time.time()
            results = await searcher.search_chunks(
                query_embedding=list(query_embedding),
                top_k=10
            )
            search_time = (time.time() - search_start) * 1000
            latencies.append(search_time)

        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        all_latencies.extend(latencies)

        print(f"  Embedding time: {embed_time:.1f}ms")
        print(f"  Search latency: avg={avg_latency:.1f}ms, min={min_latency:.1f}ms, max={max_latency:.1f}ms")
        print(f"  Results: {len(results)} chunks")
        if results:
            print(f"  Top score: {results[0].semantic_score:.4f}")
            print(f"  Top content: {results[0].content[:80]}...")

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total searches: {len(all_latencies)}")
    print(f"Average latency: {statistics.mean(all_latencies):.1f}ms")
    print(f"Median latency: {statistics.median(all_latencies):.1f}ms")
    print(f"Min latency: {min(all_latencies):.1f}ms")
    print(f"Max latency: {max(all_latencies):.1f}ms")
    print(f"Std deviation: {statistics.stdev(all_latencies):.1f}ms")

    # Performance assessment
    avg = statistics.mean(all_latencies)
    if avg < 10:
        grade = "EXCELLENT"
    elif avg < 50:
        grade = "GOOD"
    elif avg < 100:
        grade = "ACCEPTABLE"
    else:
        grade = "NEEDS OPTIMIZATION"

    print(f"\nPerformance grade: {grade}")

    await pool.close()
    print("\nBenchmark complete.")


if __name__ == '__main__':
    asyncio.run(run_benchmark())
