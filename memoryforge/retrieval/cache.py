"""Memory retrieval cache for faster query performance."""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

import structlog

from memoryforge.core.types import MemoryQuery, MemoryResult

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached entry with metadata."""

    value: T
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hits: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        """Mark entry as accessed."""
        self.hits += 1
        self.last_accessed = datetime.now(timezone.utc)

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry has expired."""
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > ttl_seconds


class LRUCache(Generic[T]):
    """Least Recently Used cache implementation."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> T | None:
        """Get value from cache."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        entry.touch()

        # Move to end (most recently used)
        self._cache.move_to_end(key)

        return entry.value

    def set(self, key: str, value: T) -> None:
        """Set value in cache."""
        if key in self._cache:
            self._cache[key].value = value
            self._cache[key].touch()
            self._cache.move_to_end(key)
        else:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(value=value)

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


class QueryCache:
    """Cache for memory query results.

    Provides:
    - LRU eviction policy
    - TTL-based expiration
    - Query fingerprinting
    - Cache statistics
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: float = 300.0,  # 5 minutes default
        enabled: bool = True,
    ):
        """Initialize query cache.

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries
            enabled: Whether caching is enabled
        """
        self._cache: LRUCache[MemoryResult] = LRUCache(max_size)
        self._ttl_seconds = ttl_seconds
        self._enabled = enabled

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            self.clear()

    def _generate_key(self, query: MemoryQuery) -> str:
        """Generate a cache key from a query."""
        # Create a deterministic representation
        parts = [
            query.query_text,
            str(query.top_k),
            str(query.min_importance),
            ",".join(sorted(l.value for l in query.target_layers)),
            ",".join(sorted(query.tags_filter or [])),
        ]
        key_str = "|".join(parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, query: MemoryQuery) -> MemoryResult | None:
        """Get cached result for a query.

        Args:
            query: The memory query

        Returns:
            Cached result or None if not found/expired
        """
        if not self._enabled:
            return None

        key = self._generate_key(query)
        entry = self._cache._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if entry.is_expired(self._ttl_seconds):
            self._cache.delete(key)
            self._misses += 1
            self._evictions += 1
            return None

        self._hits += 1
        return self._cache.get(key)

    def set(self, query: MemoryQuery, result: MemoryResult) -> None:
        """Cache a query result.

        Args:
            query: The memory query
            result: The query result to cache
        """
        if not self._enabled:
            return

        key = self._generate_key(query)
        self._cache.set(key, result)

        logger.debug(
            "Query cached",
            key=key[:8],
            entries=len(result.entries),
        )

    def invalidate(self, query: MemoryQuery) -> bool:
        """Invalidate a specific cached query.

        Args:
            query: The query to invalidate

        Returns:
            True if entry was found and removed
        """
        key = self._generate_key(query)
        return self._cache.delete(key)

    def invalidate_all(self) -> None:
        """Invalidate all cached queries."""
        self.clear()

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        logger.info("Query cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "enabled": self._enabled,
            "size": len(self._cache),
            "max_size": self._cache._max_size,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(hit_rate, 3),
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0


class EmbeddingCache:
    """Cache for computed embeddings.

    Caches text embeddings to avoid recomputing them for the same content.
    """

    def __init__(
        self,
        max_size: int = 1000,
        enabled: bool = True,
    ):
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings
            enabled: Whether caching is enabled
        """
        self._cache: LRUCache[list[float]] = LRUCache(max_size)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

    def _generate_key(self, text: str, model: str = "default") -> str:
        """Generate a cache key from text and model."""
        key_str = f"{model}:{text}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, text: str, model: str = "default") -> list[float] | None:
        """Get cached embedding.

        Args:
            text: The text that was embedded
            model: The embedding model used

        Returns:
            Cached embedding or None
        """
        if not self._enabled:
            return None

        key = self._generate_key(text, model)
        result = self._cache.get(key)

        if result is not None:
            self._hits += 1
        else:
            self._misses += 1

        return result

    def set(self, text: str, embedding: list[float], model: str = "default") -> None:
        """Cache an embedding.

        Args:
            text: The source text
            embedding: The computed embedding
            model: The embedding model used
        """
        if not self._enabled:
            return

        key = self._generate_key(text, model)
        self._cache.set(key, embedding)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "enabled": self._enabled,
            "size": len(self._cache),
            "max_size": self._cache._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }


class CacheManager:
    """Unified cache manager for all caching needs.

    Manages multiple caches and provides a unified interface
    for cache operations and statistics.
    """

    def __init__(
        self,
        query_cache_size: int = 500,
        query_cache_ttl: float = 300.0,
        embedding_cache_size: int = 1000,
        enabled: bool = True,
    ):
        """Initialize cache manager.

        Args:
            query_cache_size: Max size for query cache
            query_cache_ttl: TTL for query cache entries
            embedding_cache_size: Max size for embedding cache
            enabled: Whether caching is enabled globally
        """
        self.query_cache = QueryCache(
            max_size=query_cache_size,
            ttl_seconds=query_cache_ttl,
            enabled=enabled,
        )
        self.embedding_cache = EmbeddingCache(
            max_size=embedding_cache_size,
            enabled=enabled,
        )
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        self.query_cache.enabled = value
        self.embedding_cache.enabled = value

    def clear_all(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()
        logger.info("All caches cleared")

    def get_stats(self) -> dict:
        """Get statistics for all caches."""
        return {
            "enabled": self._enabled,
            "query_cache": self.query_cache.get_stats(),
            "embedding_cache": self.embedding_cache.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset statistics for all caches."""
        self.query_cache.reset_stats()
        self.embedding_cache._hits = 0
        self.embedding_cache._misses = 0
