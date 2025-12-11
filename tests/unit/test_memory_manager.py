"""Tests for the memory manager."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memoryforge.core.types import (
    ImportanceScore,
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
)
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.memory.manager import MemoryManager


@pytest.fixture
def working_memory():
    """Create a working memory instance."""
    return WorkingMemory(max_entries=50, max_tokens=4000)


@pytest.fixture
def mock_episodic():
    """Create a mock episodic memory."""
    mock = AsyncMock()
    mock.retrieve = AsyncMock(return_value=MagicMock(
        entries=[],
        scores=[],
        layer_sources=[],
        query_time_ms=0,
        total_candidates=0,
    ))
    mock.store = AsyncMock()
    mock.clear = AsyncMock()
    mock.get_recent_context = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_semantic():
    """Create a mock semantic memory."""
    mock = AsyncMock()
    mock.retrieve = AsyncMock(return_value=MagicMock(
        entries=[],
        scores=[],
        layer_sources=[],
        query_time_ms=0,
        total_candidates=0,
    ))
    mock.store = AsyncMock()
    mock.clear = AsyncMock()
    return mock


@pytest.fixture
def memory_manager(working_memory, mock_episodic, mock_semantic):
    """Create a memory manager with mocked components."""
    return MemoryManager(
        working=working_memory,
        episodic=mock_episodic,
        semantic=mock_semantic,
    )


@pytest.mark.asyncio
async def test_store_to_working(memory_manager):
    """Test storing to working memory."""
    entry = MemoryEntry(
        content="Test content",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
    )

    await memory_manager.store(entry, layer=MemoryLayer.WORKING)

    assert len(memory_manager.working.entries) == 1


@pytest.mark.asyncio
async def test_retrieve_from_working(memory_manager):
    """Test retrieving from working memory."""
    entries = [
        MemoryEntry(
            content="FastAPI development",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.8),
        ),
        MemoryEntry(
            content="PostgreSQL database",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.7),
        ),
    ]

    for entry in entries:
        await memory_manager.store(entry, layer=MemoryLayer.WORKING)

    query = MemoryQuery(
        query_text="database",
        target_layers=[MemoryLayer.WORKING],
        top_k=5,
    )

    result = await memory_manager.retrieve(query)

    assert len(result.entries) == 2


@pytest.mark.asyncio
async def test_retrieve_multi_layer(memory_manager):
    """Test retrieving from multiple layers."""
    entry = MemoryEntry(
        content="Working memory entry",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.6),
    )
    await memory_manager.store(entry, layer=MemoryLayer.WORKING)

    query = MemoryQuery(
        query_text="test",
        target_layers=[MemoryLayer.WORKING, MemoryLayer.EPISODIC],
        top_k=10,
    )

    result = await memory_manager.retrieve(query)

    # Should have called episodic.retrieve
    memory_manager.episodic.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_consolidate(memory_manager):
    """Test consolidating working memory to episodic."""
    # Add high importance entries
    for i in range(5):
        entry = MemoryEntry(
            content=f"Important entry {i}",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.8),
        )
        await memory_manager.store(entry, layer=MemoryLayer.WORKING)

    # Add low importance entries
    for i in range(5):
        entry = MemoryEntry(
            content=f"Less important entry {i}",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.3),
        )
        await memory_manager.store(entry, layer=MemoryLayer.WORKING)

    count = await memory_manager.consolidate()

    # Should consolidate high importance entries
    assert count >= 5


@pytest.mark.asyncio
async def test_get_context(memory_manager):
    """Test getting formatted context."""
    entries = [
        MemoryEntry(
            content="User requirement: Build API",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.9),
        ),
        MemoryEntry(
            content="Decision: Use FastAPI",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.85),
        ),
    ]

    for entry in entries:
        await memory_manager.store(entry, layer=MemoryLayer.WORKING)

    context = await memory_manager.get_context(max_tokens=1000)

    assert "Current Context" in context
    assert "User requirement" in context or "Decision" in context


@pytest.mark.asyncio
async def test_clear_all(memory_manager):
    """Test clearing all memories."""
    entry = MemoryEntry(
        content="Test entry",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
    )
    await memory_manager.store(entry, layer=MemoryLayer.WORKING)

    await memory_manager.clear_all()

    assert len(memory_manager.working.entries) == 0
    memory_manager.episodic.clear.assert_called_once()
    memory_manager.semantic.clear.assert_called_once()


@pytest.mark.asyncio
async def test_result_ranking(memory_manager):
    """Test that results are ranked by score."""
    entries = [
        MemoryEntry(
            content="Low importance",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.3),
        ),
        MemoryEntry(
            content="High importance",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.9),
        ),
        MemoryEntry(
            content="Medium importance",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.6),
        ),
    ]

    for entry in entries:
        await memory_manager.store(entry, layer=MemoryLayer.WORKING)

    query = MemoryQuery(
        query_text="test",
        target_layers=[MemoryLayer.WORKING],
        top_k=10,
    )

    result = await memory_manager.retrieve(query)

    # Verify descending order by score
    for i in range(len(result.scores) - 1):
        assert result.scores[i] >= result.scores[i + 1]
