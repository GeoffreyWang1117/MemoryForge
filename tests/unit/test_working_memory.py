"""Tests for working memory implementation."""

import pytest
from uuid import uuid4

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer, MemoryQuery
from memoryforge.memory.working.memory import WorkingMemory


@pytest.fixture
def working_memory():
    """Create a working memory instance for testing."""
    return WorkingMemory(max_entries=10, max_tokens=1000, importance_threshold=0.7)


@pytest.fixture
def sample_entry():
    """Create a sample memory entry."""
    return MemoryEntry(
        content="This is a test memory entry",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
        tags=["test"],
    )


@pytest.mark.asyncio
async def test_store_and_retrieve(working_memory, sample_entry):
    """Test basic store and retrieve functionality."""
    await working_memory.store(sample_entry)

    query = MemoryQuery(query_text="test", target_layers=[MemoryLayer.WORKING])
    result = await working_memory.retrieve(query)

    assert len(result.entries) == 1
    assert result.entries[0].content == sample_entry.content


@pytest.mark.asyncio
async def test_importance_threshold_pinning(working_memory):
    """Test that high-importance entries are pinned."""
    high_importance = MemoryEntry(
        content="Important entry",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.9),
    )

    await working_memory.store(high_importance)

    assert high_importance.id in working_memory._pinned


@pytest.mark.asyncio
async def test_sliding_window_eviction(working_memory):
    """Test that old entries are evicted when window is full."""
    for i in range(15):
        entry = MemoryEntry(
            content=f"Entry {i}",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.3),
        )
        await working_memory.store(entry)

    assert len(working_memory._entries) <= 10


@pytest.mark.asyncio
async def test_pin_and_unpin(working_memory, sample_entry):
    """Test pinning and unpinning entries."""
    await working_memory.store(sample_entry)

    assert working_memory.pin(sample_entry.id)
    assert sample_entry.id in working_memory._pinned

    assert working_memory.unpin(sample_entry.id)
    assert sample_entry.id not in working_memory._pinned


@pytest.mark.asyncio
async def test_delete_entry(working_memory, sample_entry):
    """Test deleting an entry."""
    await working_memory.store(sample_entry)

    result = await working_memory.delete(str(sample_entry.id))
    assert result is True

    query = MemoryQuery(query_text="test", target_layers=[MemoryLayer.WORKING])
    result = await working_memory.retrieve(query)
    assert len(result.entries) == 0


@pytest.mark.asyncio
async def test_clear(working_memory, sample_entry):
    """Test clearing all entries."""
    await working_memory.store(sample_entry)
    await working_memory.clear()

    assert len(working_memory.entries) == 0
    assert working_memory.token_count == 0


@pytest.mark.asyncio
async def test_get_context_window(working_memory):
    """Test getting entries within token budget."""
    for i in range(5):
        entry = MemoryEntry(
            content=f"Entry {i} " * 50,
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.5 + i * 0.1),
        )
        await working_memory.store(entry)

    context = working_memory.get_context_window(max_tokens=200)

    assert len(context) > 0
    assert context[0].importance.base_score >= context[-1].importance.base_score


@pytest.mark.asyncio
async def test_update_importance(working_memory, sample_entry):
    """Test updating entry importance."""
    await working_memory.store(sample_entry)

    result = working_memory.update_importance(sample_entry.id, 0.9)
    assert result is True

    for entry in working_memory.entries:
        if entry.id == sample_entry.id:
            assert entry.importance.base_score == 0.9
