"""Semantic search using embeddings for high-quality retrieval."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Awaitable
from uuid import UUID

import numpy as np
import structlog

from memoryforge.core.types import MemoryEntry, MemoryQuery, MemoryResult
from memoryforge.retrieval.cache import EmbeddingCache

logger = structlog.get_logger()


# Type for embedding function
EmbedFunction = Callable[[str], Awaitable[list[float]]]


@dataclass
class SearchResult:
    """A single search result with score."""

    entry: MemoryEntry
    score: float
    match_type: str = "semantic"  # semantic, keyword, hybrid


@dataclass
class SemanticSearchConfig:
    """Configuration for semantic search."""

    # Minimum similarity threshold
    min_similarity: float = 0.5

    # Weight for semantic vs keyword matching (0-1)
    semantic_weight: float = 0.7

    # Number of candidates to consider
    candidate_multiplier: int = 3

    # Enable hybrid search (semantic + keyword)
    hybrid_search: bool = True

    # Boost for exact keyword matches
    keyword_boost: float = 0.2

    # Recency boost factor
    recency_boost: float = 0.1

    # Importance boost factor
    importance_boost: float = 0.15


class SemanticSearch:
    """Semantic search engine using embeddings.

    Provides:
    - Vector similarity search
    - Hybrid semantic + keyword search
    - Importance and recency boosting
    - Result re-ranking
    """

    def __init__(
        self,
        embed_fn: EmbedFunction | None = None,
        config: SemanticSearchConfig | None = None,
        cache: EmbeddingCache | None = None,
    ):
        """Initialize semantic search.

        Args:
            embed_fn: Function to generate embeddings
            config: Search configuration
            cache: Optional embedding cache
        """
        self._embed_fn = embed_fn
        self._config = config or SemanticSearchConfig()
        self._cache = cache or EmbeddingCache()

        # In-memory index
        self._entries: list[MemoryEntry] = []
        self._embeddings: list[list[float]] = []

    def set_embed_function(self, embed_fn: EmbedFunction) -> None:
        """Set the embedding function."""
        self._embed_fn = embed_fn

    async def index(self, entries: list[MemoryEntry]) -> int:
        """Index memory entries for search.

        Args:
            entries: Entries to index

        Returns:
            Number of entries indexed
        """
        if not self._embed_fn:
            raise ValueError("Embedding function not set")

        self._entries = []
        self._embeddings = []

        for entry in entries:
            # Check cache first
            cached = self._cache.get(entry.content)
            if cached:
                embedding = cached
            else:
                embedding = await self._embed_fn(entry.content)
                self._cache.set(entry.content, embedding)

            self._entries.append(entry)
            self._embeddings.append(embedding)

        logger.info(f"Indexed {len(self._entries)} entries for semantic search")
        return len(self._entries)

    async def add_entry(self, entry: MemoryEntry) -> None:
        """Add a single entry to the index.

        Args:
            entry: Entry to add
        """
        if not self._embed_fn:
            raise ValueError("Embedding function not set")

        # Check cache
        cached = self._cache.get(entry.content)
        if cached:
            embedding = cached
        else:
            embedding = await self._embed_fn(entry.content)
            self._cache.set(entry.content, embedding)

        self._entries.append(entry)
        self._embeddings.append(embedding)

    def remove_entry(self, entry_id: UUID) -> bool:
        """Remove an entry from the index.

        Args:
            entry_id: ID of entry to remove

        Returns:
            True if removed
        """
        for i, entry in enumerate(self._entries):
            if entry.id == entry_id:
                self._entries.pop(i)
                self._embeddings.pop(i)
                return True
        return False

    async def search(
        self,
        query: str | MemoryQuery,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for relevant memories.

        Args:
            query: Search query (string or MemoryQuery)
            top_k: Number of results to return

        Returns:
            List of search results with scores
        """
        if not self._entries:
            return []

        if isinstance(query, MemoryQuery):
            query_text = query.query_text
            top_k = query.top_k
        else:
            query_text = query

        # Get query embedding
        if self._embed_fn:
            cached = self._cache.get(query_text, model="query")
            if cached:
                query_embedding = cached
            else:
                query_embedding = await self._embed_fn(query_text)
                self._cache.set(query_text, query_embedding, model="query")
        else:
            # Fallback to keyword search only
            return self._keyword_search(query_text, top_k)

        # Compute similarities
        results = []
        for i, (entry, embedding) in enumerate(zip(self._entries, self._embeddings)):
            # Semantic similarity
            semantic_score = self._cosine_similarity(query_embedding, embedding)

            # Apply minimum threshold
            if semantic_score < self._config.min_similarity:
                continue

            # Calculate final score
            score = semantic_score

            # Hybrid search: add keyword boost
            if self._config.hybrid_search:
                keyword_score = self._keyword_match_score(query_text, entry.content)
                score = (
                    self._config.semantic_weight * semantic_score +
                    (1 - self._config.semantic_weight) * keyword_score
                )

            # Apply boosts
            score = self._apply_boosts(score, entry)

            results.append(SearchResult(
                entry=entry,
                score=score,
                match_type="hybrid" if self._config.hybrid_search else "semantic",
            ))

        # Sort by score and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)

        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def _keyword_match_score(self, query: str, content: str) -> float:
        """Calculate keyword match score."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        matches = len(query_words & content_words)
        return matches / len(query_words)

    def _apply_boosts(self, score: float, entry: MemoryEntry) -> float:
        """Apply importance and recency boosts."""
        # Importance boost
        importance = entry.importance.effective_score
        score += self._config.importance_boost * importance

        # Recency boost (decays over time)
        age_hours = (
            datetime.now(timezone.utc) - entry.created_at
        ).total_seconds() / 3600

        recency_factor = max(0, 1 - (age_hours / 168))  # Decay over 1 week
        score += self._config.recency_boost * recency_factor

        return min(score, 1.0)  # Cap at 1.0

    def _keyword_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Fallback keyword search without embeddings."""
        results = []

        for entry in self._entries:
            score = self._keyword_match_score(query, entry.content)
            if score > 0:
                score = self._apply_boosts(score, entry)
                results.append(SearchResult(
                    entry=entry,
                    score=score,
                    match_type="keyword",
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def search_similar(
        self,
        entry: MemoryEntry,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> list[SearchResult]:
        """Find memories similar to a given entry.

        Args:
            entry: Entry to find similar memories for
            top_k: Number of results
            exclude_self: Exclude the input entry from results

        Returns:
            Similar entries
        """
        if not self._embeddings:
            return []

        # Get embedding for the entry
        if entry.embedding:
            query_embedding = entry.embedding
        elif self._embed_fn:
            query_embedding = await self._embed_fn(entry.content)
        else:
            return []

        results = []
        for i, (e, embedding) in enumerate(zip(self._entries, self._embeddings)):
            if exclude_self and e.id == entry.id:
                continue

            score = self._cosine_similarity(query_embedding, embedding)
            if score >= self._config.min_similarity:
                results.append(SearchResult(entry=e, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict:
        """Get search index statistics."""
        return {
            "indexed_entries": len(self._entries),
            "embedding_dimensions": len(self._embeddings[0]) if self._embeddings else 0,
            "cache_stats": self._cache.get_stats(),
            "config": {
                "min_similarity": self._config.min_similarity,
                "semantic_weight": self._config.semantic_weight,
                "hybrid_search": self._config.hybrid_search,
            },
        }

    def clear(self) -> None:
        """Clear the search index."""
        self._entries.clear()
        self._embeddings.clear()


class SimpleEmbedder:
    """Simple TF-IDF based embedder for testing without ML models."""

    def __init__(self, dim: int = 256):
        """Initialize simple embedder.

        Args:
            dim: Embedding dimension
        """
        self._dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}

    async def embed(self, text: str) -> list[float]:
        """Generate a simple embedding for text.

        This uses a hash-based approach for consistent embeddings.
        """
        import hashlib

        words = text.lower().split()

        # Create embedding using word hashes
        embedding = [0.0] * self._dim

        for word in words:
            # Hash word to get position
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            pos = h % self._dim
            # Use hash bits for sign
            sign = 1 if (h >> 128) % 2 == 0 else -1
            embedding[pos] += sign * (1.0 / len(words))

        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding
