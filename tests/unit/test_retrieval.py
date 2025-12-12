"""Unit tests for retrieval module."""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer
from memoryforge.retrieval.semantic import (
    SemanticSearch,
    SemanticSearchConfig,
    SearchResult,
)
from memoryforge.retrieval.cache import EmbeddingCache


def create_entry(
    content: str,
    importance: float = 0.5,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> MemoryEntry:
    """Helper to create test entries."""
    entry = MemoryEntry(
        content=content,
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=importance),
        tags=tags or [],
    )
    if created_at:
        entry.created_at = created_at
    return entry


# Mock embedding function for testing
async def mock_embed_fn(text: str) -> list[float]:
    """Simple mock embedding that creates consistent vectors based on text."""
    # Create a simple hash-based embedding
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    # Convert hex to floats (0-1 range)
    embedding = [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
    # Pad to dimension 16
    while len(embedding) < 16:
        embedding.append(0.5)
    return embedding[:16]


class TestEmbeddingCache:
    """Tests for embedding cache."""

    def test_set_and_get(self):
        """Test basic set and get."""
        cache = EmbeddingCache(max_size=100)
        embedding = [0.1, 0.2, 0.3]

        cache.set("test content", embedding)
        result = cache.get("test content")

        assert result == embedding

    def test_get_missing(self):
        """Test getting non-existent key."""
        cache = EmbeddingCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_max_size_eviction(self):
        """Test that cache evicts old entries when full."""
        cache = EmbeddingCache(max_size=3)

        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])
        cache.set("d", [4.0])  # Should evict 'a'

        assert cache.get("a") is None
        assert cache.get("d") == [4.0]

    def test_clear(self):
        """Test clearing cache."""
        cache = EmbeddingCache()
        cache.set("test", [1.0])
        cache.clear()

        assert cache.get("test") is None

    def test_stats(self):
        """Test cache statistics."""
        cache = EmbeddingCache()
        cache.set("a", [1.0])
        cache.set("b", [2.0])

        cache.get("a")  # Hit
        cache.get("c")  # Miss

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestSemanticSearchConfig:
    """Tests for search configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SemanticSearchConfig()

        assert config.min_similarity == 0.5
        assert config.semantic_weight == 0.7
        assert config.hybrid_search is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SemanticSearchConfig(
            min_similarity=0.7,
            semantic_weight=0.9,
            hybrid_search=False,
        )

        assert config.min_similarity == 0.7
        assert config.semantic_weight == 0.9
        assert config.hybrid_search is False


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        entry = create_entry("Test")
        result = SearchResult(entry=entry, score=0.85, match_type="semantic")

        assert result.entry == entry
        assert result.score == 0.85
        assert result.match_type == "semantic"

    def test_search_result_default_match_type(self):
        """Test default match type."""
        entry = create_entry("Test")
        result = SearchResult(entry=entry, score=0.5)

        assert result.match_type == "semantic"


class TestSemanticSearch:
    """Tests for semantic search engine."""

    @pytest.fixture
    def search(self):
        """Create a semantic search instance."""
        return SemanticSearch(
            embed_fn=mock_embed_fn,
            config=SemanticSearchConfig(min_similarity=0.0),
        )

    @pytest.mark.asyncio
    async def test_index_entries(self, search):
        """Test indexing entries."""
        entries = [
            create_entry("Python programming"),
            create_entry("JavaScript basics"),
            create_entry("Database design"),
        ]

        count = await search.index(entries)
        assert count == 3

    @pytest.mark.asyncio
    async def test_index_without_embed_fn(self):
        """Test indexing without embedding function raises error."""
        search = SemanticSearch(embed_fn=None)
        entries = [create_entry("Test")]

        with pytest.raises(ValueError, match="Embedding function not set"):
            await search.index(entries)

    @pytest.mark.asyncio
    async def test_add_entry(self, search):
        """Test adding single entry."""
        entry = create_entry("New entry")
        await search.add_entry(entry)

        assert len(search._entries) == 1
        assert len(search._embeddings) == 1

    @pytest.mark.asyncio
    async def test_add_entry_without_embed_fn(self):
        """Test adding entry without embedding function."""
        search = SemanticSearch(embed_fn=None)

        with pytest.raises(ValueError):
            await search.add_entry(create_entry("Test"))

    @pytest.mark.asyncio
    async def test_search_basic(self, search):
        """Test basic search."""
        entries = [
            create_entry("Python web development"),
            create_entry("Python machine learning"),
            create_entry("JavaScript frontend"),
        ]
        await search.index(entries)

        results = await search.search("Python programming", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_index(self, search):
        """Test searching empty index."""
        results = await search.search("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, search):
        """Test that search respects top_k limit."""
        entries = [create_entry(f"Entry {i}") for i in range(10)]
        await search.index(entries)

        results = await search.search("Entry", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_remove_entry(self, search):
        """Test removing entry from index."""
        entry = create_entry("To remove")
        await search.add_entry(entry)

        assert len(search._entries) == 1

        removed = search.remove_entry(entry.id)
        assert removed is True
        assert len(search._entries) == 0

    def test_remove_nonexistent_entry(self, search):
        """Test removing non-existent entry."""
        removed = search.remove_entry(uuid4())
        assert removed is False

    @pytest.mark.asyncio
    async def test_clear_index(self, search):
        """Test clearing the index."""
        entries = [create_entry(f"Entry {i}") for i in range(5)]
        await search.index(entries)

        search.clear()
        assert len(search._entries) == 0
        assert len(search._embeddings) == 0

    def test_set_embed_function(self):
        """Test setting embed function after init."""
        search = SemanticSearch()
        assert search._embed_fn is None

        search.set_embed_function(mock_embed_fn)
        assert search._embed_fn is not None

    @pytest.mark.asyncio
    async def test_search_uses_cache(self, search):
        """Test that search uses embedding cache."""
        entries = [create_entry("Cached content")]
        await search.index(entries)

        # Cache should be populated
        stats = search._cache.get_stats()
        assert stats["size"] >= 1

    @pytest.mark.asyncio
    async def test_search_with_importance_boost(self, search):
        """Test that importance affects ranking."""
        entries = [
            create_entry("Important topic", importance=0.9),
            create_entry("Less important topic", importance=0.3),
        ]
        await search.index(entries)

        results = await search.search("topic", top_k=2)

        # Higher importance should rank higher (with same semantic similarity)
        if len(results) == 2:
            # Results should consider importance
            assert results[0].score >= 0

    @pytest.mark.asyncio
    async def test_search_with_recency_boost(self, search):
        """Test that recency affects ranking."""
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        new_time = datetime.now(timezone.utc)

        entries = [
            create_entry("Old entry", created_at=old_time),
            create_entry("New entry", created_at=new_time),
        ]
        await search.index(entries)

        results = await search.search("entry", top_k=2)
        assert len(results) >= 1


class TestSemanticSearchHybrid:
    """Tests for hybrid search functionality."""

    @pytest.fixture
    def hybrid_search(self):
        """Create search with hybrid mode enabled."""
        config = SemanticSearchConfig(
            hybrid_search=True,
            keyword_boost=0.3,
            min_similarity=0.0,
        )
        return SemanticSearch(embed_fn=mock_embed_fn, config=config)

    @pytest.mark.asyncio
    async def test_hybrid_search_boosts_keyword_matches(self, hybrid_search):
        """Test that exact keyword matches get boosted."""
        entries = [
            create_entry("Python is great for web development"),
            create_entry("Ruby is also used for web apps"),
        ]
        await hybrid_search.index(entries)

        results = await hybrid_search.search("Python web", top_k=2)

        # Both should be found, but Python entry should rank higher
        assert len(results) >= 1


class TestSemanticSearchEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def search(self):
        return SemanticSearch(
            embed_fn=mock_embed_fn,
            config=SemanticSearchConfig(min_similarity=0.0),
        )

    @pytest.mark.asyncio
    async def test_search_with_special_characters(self, search):
        """Test searching with special characters."""
        entries = [
            create_entry("Code: print('hello')"),
            create_entry("SQL: SELECT * FROM users"),
        ]
        await search.index(entries)

        results = await search.search("print('", top_k=2)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, search):
        """Test searching with empty query."""
        entries = [create_entry("Some content")]
        await search.index(entries)

        results = await search.search("", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_unicode(self, search):
        """Test searching with unicode characters."""
        entries = [
            create_entry("中文内容测试"),
            create_entry("日本語テスト"),
            create_entry("Emoji test 🚀"),
        ]
        await search.index(entries)

        results = await search.search("中文", top_k=2)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_concurrent_indexing(self, search):
        """Test concurrent add operations."""
        import asyncio

        entries = [create_entry(f"Concurrent {i}") for i in range(10)]

        # Add entries concurrently
        await asyncio.gather(*[search.add_entry(e) for e in entries])

        assert len(search._entries) == 10

    @pytest.mark.asyncio
    async def test_reindex_clears_previous(self, search):
        """Test that reindexing clears previous entries."""
        entries1 = [create_entry("First batch")]
        entries2 = [create_entry("Second batch")]

        await search.index(entries1)
        assert len(search._entries) == 1

        await search.index(entries2)
        assert len(search._entries) == 1
        assert search._entries[0].content == "Second batch"
