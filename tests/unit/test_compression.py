"""Tests for memory compression pipeline."""

from datetime import datetime, timedelta, timezone

import pytest

from memoryforge.compression import (
    CompressionConfig,
    CompressionLevel,
    CompressionPipeline,
    SummaryCompressor,
    TimeWindowCompressor,
)
from memoryforge.compression.strategies import DeduplicationCompressor
from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer


def create_test_entry(
    content: str,
    importance: float = 0.5,
    tags: list[str] | None = None,
    age_hours: float = 48,
) -> MemoryEntry:
    """Create a test memory entry."""
    created_at = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=importance),
        tags=tags or [],
        created_at=created_at,
        updated_at=created_at,
    )


@pytest.mark.asyncio
async def test_compression_pipeline_basic():
    """Test basic pipeline functionality."""
    pipeline = CompressionPipeline()
    pipeline.add_strategy(SummaryCompressor())

    entries = [
        create_test_entry("Memory 1 about testing"),
        create_test_entry("Memory 2 about testing"),
        create_test_entry("Memory 3 about coding"),
    ]

    result = await pipeline.compress(entries)

    assert result.original_count == 3
    assert result.compressed_count <= 3
    assert len(result.compressed_entries) > 0


@pytest.mark.asyncio
async def test_compression_preserves_important():
    """Test that high-importance memories are preserved."""
    config = CompressionConfig(
        importance_threshold=0.8,
        min_age_hours=1,
    )
    pipeline = CompressionPipeline(config=config)
    pipeline.add_strategy(SummaryCompressor())

    entries = [
        create_test_entry("Important memory", importance=0.95),
        create_test_entry("Regular memory 1", importance=0.5),
        create_test_entry("Regular memory 2", importance=0.4),
    ]

    result = await pipeline.compress(entries)

    # Important memory should be preserved
    important_preserved = any(
        "Important memory" in e.content
        for e in result.compressed_entries
    )
    assert important_preserved


@pytest.mark.asyncio
async def test_compression_preserves_tagged():
    """Test that tagged memories are preserved."""
    config = CompressionConfig(
        preserve_tags=["pinned", "important"],
        min_age_hours=1,
    )
    pipeline = CompressionPipeline(config=config)
    pipeline.add_strategy(SummaryCompressor())

    entries = [
        create_test_entry("Pinned memory", tags=["pinned"]),
        create_test_entry("Regular memory 1"),
        create_test_entry("Regular memory 2"),
    ]

    result = await pipeline.compress(entries)

    # Pinned memory should be preserved unchanged
    pinned_found = any(
        e.content == "Pinned memory"
        for e in result.compressed_entries
    )
    assert pinned_found


@pytest.mark.asyncio
async def test_compression_respects_age():
    """Test that recent memories are not compressed."""
    config = CompressionConfig(min_age_hours=24)
    pipeline = CompressionPipeline(config=config)
    pipeline.add_strategy(SummaryCompressor())

    entries = [
        create_test_entry("Recent memory", age_hours=1),  # Too recent
        create_test_entry("Old memory 1", age_hours=48),
        create_test_entry("Old memory 2", age_hours=72),
    ]

    eligible, preserved = pipeline.filter_eligible(entries)

    assert len(preserved) == 1  # Recent memory preserved
    assert len(eligible) == 2  # Old memories eligible


@pytest.mark.asyncio
async def test_summary_compressor():
    """Test summary compression strategy."""
    compressor = SummaryCompressor()
    config = CompressionConfig(level=CompressionLevel.MODERATE)

    entries = [
        create_test_entry("First point about topic A"),
        create_test_entry("Second point about topic A"),
        create_test_entry("Third point about topic A"),
    ]

    compressed = await compressor.compress(entries, config)

    assert len(compressed) <= len(entries)
    # Should have compressed tag
    assert any("compressed" in e.tags for e in compressed)


@pytest.mark.asyncio
async def test_time_window_compressor():
    """Test time window compression."""
    compressor = TimeWindowCompressor(window_hours=24)
    config = CompressionConfig(level=CompressionLevel.MODERATE)

    # Create entries in the same day but with enough span for the window
    now = datetime.now(timezone.utc)
    entries = [
        MemoryEntry(
            content=f"Event {i} on day {i}",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.5),
            created_at=now - timedelta(days=i),
            updated_at=now - timedelta(days=i),
        )
        for i in range(3)
    ]

    # Verify entries span multiple windows
    assert compressor.can_compress(entries)

    compressed = await compressor.compress(entries, config)

    # Should produce valid output
    assert len(compressed) > 0
    assert len(compressed) <= len(entries)


@pytest.mark.asyncio
async def test_deduplication_compressor():
    """Test deduplication strategy."""
    compressor = DeduplicationCompressor()
    config = CompressionConfig()

    entries = [
        create_test_entry("Duplicate content", importance=0.5),
        create_test_entry("Duplicate content", importance=0.7),  # Higher importance
        create_test_entry("Unique content"),
    ]

    compressed = await compressor.compress(entries, config)

    assert len(compressed) == 2  # One duplicate removed
    # Should keep the higher importance version
    duplicate_entry = next(e for e in compressed if "Duplicate" in e.content)
    assert duplicate_entry.importance.base_score == 0.7


@pytest.mark.asyncio
async def test_compression_result_metrics():
    """Test compression result metrics."""
    pipeline = CompressionPipeline()
    pipeline.add_strategy(SummaryCompressor())

    entries = [
        create_test_entry("Short content"),
        create_test_entry("A" * 1000),  # Long content
    ]

    result = await pipeline.compress(entries)

    assert result.original_count == 2
    assert result.original_tokens > 0
    assert result.compression_ratio <= 1.0


@pytest.mark.asyncio
async def test_auto_compress():
    """Test automatic compression to size limit."""
    config = CompressionConfig(
        min_age_hours=1,
        importance_threshold=1.0,  # Allow all to be compressed
    )
    pipeline = CompressionPipeline(config=config)
    pipeline.add_strategy(SummaryCompressor())

    entries = [
        create_test_entry(f"Memory {i}", importance=0.3)
        for i in range(10)
    ]

    result = await pipeline.auto_compress(entries, max_entries=5)

    assert result.compressed_count <= 5


def test_compression_stats():
    """Test compression statistics."""
    pipeline = CompressionPipeline()
    pipeline.add_strategy(SummaryCompressor())

    stats = pipeline.get_stats()

    assert stats["total_compressions"] == 0
    assert stats["total_entries_processed"] == 0


@pytest.mark.asyncio
async def test_compression_levels():
    """Test different compression levels."""
    for level in CompressionLevel:
        config = CompressionConfig(level=level, min_age_hours=1)
        compressor = SummaryCompressor()

        entries = [
            create_test_entry(f"Memory {i} with some content")
            for i in range(5)
        ]

        compressed = await compressor.compress(entries, config)

        # Should produce valid output for all levels
        assert len(compressed) > 0
        for entry in compressed:
            assert entry.content
