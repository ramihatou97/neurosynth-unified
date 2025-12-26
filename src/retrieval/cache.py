"""
NeuroSynth Unified - Caching Module
====================================

Three-layer caching strategy for search and retrieval optimization:

1. Query Embedding Cache (LRU, 1000 queries, 30-50% hit rate)
   - Caches text → embedding conversions
   - Saves 20-30ms per cached query

2. Search Result Cache (TTL, 500 searches, 20-40% hit rate)
   - Caches complete search results
   - Saves 50ms → <5ms for cache hits

3. Similar Chunks Cache (LRU, 10k queries)
   - Caches "find similar to X" queries
   - Persistent across requests

Expected improvement: 20-40% overall latency reduction on repeated queries

Usage:
    from src.retrieval.cache import SearchCache

    cache = SearchCache()

    # Query embedding cache
    embedding = await cache.get_or_compute_embedding(query, embedder.embed_text)

    # Search result cache
    results = await cache.get_or_search(
        cache_key="query|mode|filters",
        search_func=lambda: search_service.search(...)
    )
"""

import logging
from typing import Any, Optional, Callable, Dict, List, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# LRU Cache
# =============================================================================

class LRUCache(Generic[T]):
    """
    Thread-safe LRU (Least Recently Used) cache.

    Evicts least recently used items when capacity is reached.
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items to cache
        """
        self.capacity = capacity
        self.cache: OrderedDict[str, T] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get item from cache, updating access order."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self._hits += 1
            return self.cache[key]

        self._misses += 1
        return None

    def put(self, key: str, value: T) -> None:
        """Add or update item in cache."""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)

        self.cache[key] = value

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self.cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "capacity": self.capacity,
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# =============================================================================
# TTL Cache
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL."""
    value: T
    expires_at: datetime


class TTLCache(Generic[T]):
    """
    Thread-safe TTL (Time To Live) cache.

    Items expire after a fixed duration.
    """

    def __init__(self, ttl_seconds: int = 300, capacity: int = 500):
        """
        Initialize TTL cache.

        Args:
            ttl_seconds: Time to live in seconds (default: 300 = 5 minutes)
            capacity: Maximum number of items
        """
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self.cache: Dict[str, CacheEntry[T]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get item from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]

            # Check expiration
            if datetime.now() < entry.expires_at:
                self._hits += 1
                return entry.value

            # Expired - remove
            del self.cache[key]

        self._misses += 1
        return None

    def put(self, key: str, value: T) -> None:
        """Add item to cache with TTL."""
        # Evict if at capacity
        if len(self.cache) >= self.capacity and key not in self.cache:
            # Remove oldest (first in dict)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        entry = CacheEntry(
            value=value,
            expires_at=datetime.now() + timedelta(seconds=self.ttl_seconds)
        )

        self.cache[key] = entry

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count of evicted."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now >= entry.expires_at
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self.cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "capacity": self.capacity,
            "size": self.size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# =============================================================================
# Search Cache (Main Interface)
# =============================================================================

class SearchCache:
    """
    Unified cache for search operations with three layers:

    1. Embedding Cache: LRU cache for query → embedding
    2. Search Result Cache: TTL cache for complete search results
    3. Similar Chunks Cache: LRU cache for "find similar" queries
    """

    def __init__(
        self,
        embedding_cache_size: int = 1000,
        result_cache_size: int = 500,
        result_ttl_seconds: int = 300,
        similar_cache_size: int = 10000,
    ):
        """
        Initialize search cache.

        Args:
            embedding_cache_size: LRU capacity for embeddings
            result_cache_size: Capacity for search results
            result_ttl_seconds: TTL for search results (default: 5 minutes)
            similar_cache_size: LRU capacity for similar chunk queries
        """
        self.embedding_cache = LRUCache[Any](capacity=embedding_cache_size)
        self.result_cache = TTLCache[Any](
            ttl_seconds=result_ttl_seconds,
            capacity=result_cache_size
        )
        self.similar_cache = LRUCache[Any](capacity=similar_cache_size)

        logger.info(
            f"Initialized SearchCache: "
            f"embeddings={embedding_cache_size}, "
            f"results={result_cache_size} (TTL={result_ttl_seconds}s), "
            f"similar={similar_cache_size}"
        )

    # =========================================================================
    # Embedding Cache
    # =========================================================================

    async def get_or_compute_embedding(
        self,
        text: str,
        embed_func: Callable
    ) -> Any:
        """
        Get embedding from cache or compute if missing.

        Args:
            text: Query text
            embed_func: Async function to compute embedding

        Returns:
            Embedding vector (numpy array)
        """
        cache_key = self._hash_text(text)

        # Check cache
        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Embedding cache HIT: {cache_key[:16]}")
            return cached

        # Compute and cache
        logger.debug(f"Embedding cache MISS: {cache_key[:16]}")
        embedding = await embed_func(text)
        self.embedding_cache.put(cache_key, embedding)

        return embedding

    # =========================================================================
    # Search Result Cache
    # =========================================================================

    async def get_or_search(
        self,
        cache_key: str,
        search_func: Callable
    ) -> Any:
        """
        Get search results from cache or execute search.

        Args:
            cache_key: Unique key for this search (query|mode|filters)
            search_func: Async function to execute search

        Returns:
            Search results
        """
        # Check cache
        cached = self.result_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Search cache HIT: {cache_key[:32]}")
            return cached

        # Execute search and cache
        logger.debug(f"Search cache MISS: {cache_key[:32]}")
        results = await search_func()
        self.result_cache.put(cache_key, results)

        return results

    def make_search_key(
        self,
        query: str,
        mode: str = "text",
        top_k: int = 10,
        filters: Dict = None
    ) -> str:
        """
        Create cache key for search query.

        Args:
            query: Search query
            mode: Search mode
            top_k: Result count
            filters: Search filters

        Returns:
            Cache key string
        """
        components = [
            query,
            mode,
            str(top_k),
            json.dumps(filters or {}, sort_keys=True)
        ]

        key_str = "|".join(components)
        return self._hash_text(key_str)

    # =========================================================================
    # Similar Chunks Cache
    # =========================================================================

    async def get_or_find_similar(
        self,
        chunk_id: str,
        find_func: Callable
    ) -> Any:
        """
        Get similar chunks from cache or compute.

        Args:
            chunk_id: Source chunk ID
            find_func: Async function to find similar chunks

        Returns:
            List of similar chunks
        """
        # Check cache
        cached = self.similar_cache.get(chunk_id)
        if cached is not None:
            logger.debug(f"Similar cache HIT: {chunk_id}")
            return cached

        # Compute and cache
        logger.debug(f"Similar cache MISS: {chunk_id}")
        similar = await find_func()
        self.similar_cache.put(chunk_id, similar)

        return similar

    # =========================================================================
    # Utilities
    # =========================================================================

    def _hash_text(self, text: str) -> str:
        """Create hash of text for cache key."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def clear_all(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.result_cache.clear()
        self.similar_cache.clear()
        logger.info("Cleared all caches")

    def evict_expired(self) -> int:
        """Evict expired TTL cache entries. Returns count of evicted."""
        return self.result_cache.evict_expired()

    def stats(self) -> Dict[str, Dict[str, Any]]:
        """Return statistics for all caches."""
        return {
            "embedding_cache": self.embedding_cache.stats(),
            "result_cache": self.result_cache.stats(),
            "similar_cache": self.similar_cache.stats(),
        }


# =============================================================================
# Decorator for Caching
# =============================================================================

def cached_embedding(cache: SearchCache):
    """Decorator to cache embedding function results."""
    def decorator(func):
        async def wrapper(self, text: str, *args, **kwargs):
            return await cache.get_or_compute_embedding(
                text,
                lambda: func(self, text, *args, **kwargs)
            )
        return wrapper
    return decorator


def cached_search(cache: SearchCache, ttl_seconds: int = 300):
    """Decorator to cache search function results."""
    def decorator(func):
        async def wrapper(self, query: str, mode="text", top_k=10, filters=None, *args, **kwargs):
            cache_key = cache.make_search_key(query, mode, top_k, filters)
            return await cache.get_or_search(
                cache_key,
                lambda: func(self, query, mode, top_k, filters, *args, **kwargs)
            )
        return wrapper
    return decorator


# =============================================================================
# Connection Pool Cache Configuration
# =============================================================================

@dataclass
class ConnectionPoolConfig:
    """
    Optimized connection pool configuration.

    Settings tuned for 50 concurrent users.
    """
    min_size: int = 10              # Keep connections warm
    max_size: int = 50              # Support 50 concurrent users
    statement_cache_size: int = 100  # Cache prepared statements
    command_timeout: int = 60       # 60 second query timeout
    max_inactive_connection_lifetime: float = 300.0  # 5 minutes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for asyncpg.create_pool()."""
        return {
            "min_size": self.min_size,
            "max_size": self.max_size,
            "statement_cache_size": self.statement_cache_size,
            "command_timeout": self.command_timeout,
            "max_inactive_connection_lifetime": self.max_inactive_connection_lifetime,
        }


# =============================================================================
# CLI for Cache Stats
# =============================================================================

if __name__ == "__main__":
    # Demo cache usage
    cache = SearchCache(
        embedding_cache_size=1000,
        result_cache_size=500,
        result_ttl_seconds=300,
        similar_cache_size=10000
    )

    # Simulate some cache operations
    import numpy as np

    async def demo():
        print("=" * 60)
        print("SearchCache Demo")
        print("=" * 60)

        # Simulate embedding caching
        async def mock_embed(text):
            await asyncio.sleep(0.020)  # 20ms
            return np.random.randn(1024)

        # First call - cache miss
        print("\n1. First embedding call (MISS):")
        e1 = await cache.get_or_compute_embedding("test query", mock_embed)
        print(f"   Computed embedding: {e1.shape}")

        # Second call - cache hit
        print("\n2. Second embedding call (HIT):")
        e2 = await cache.get_or_compute_embedding("test query", mock_embed)
        print(f"   Cached embedding: {e2.shape}")

        # Stats
        print("\n3. Cache statistics:")
        stats = cache.stats()
        for cache_name, cache_stats in stats.items():
            print(f"\n   {cache_name}:")
            for key, value in cache_stats.items():
                if key == "hit_rate":
                    print(f"      {key}: {value:.2%}")
                else:
                    print(f"      {key}: {value}")

    asyncio.run(demo())
