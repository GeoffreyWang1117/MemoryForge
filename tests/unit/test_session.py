"""Tests for multi-session manager."""

import pytest

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer, MemoryQuery
from memoryforge.session import SessionManager, SessionConfig, SessionState


@pytest.mark.asyncio
async def test_create_session():
    """Test session creation."""
    manager = SessionManager()
    session = await manager.create_session()

    assert session is not None
    assert session.id
    assert session.state == SessionState.ACTIVE
    assert session.working_memory is not None


@pytest.mark.asyncio
async def test_create_named_session():
    """Test creating a session with custom config."""
    manager = SessionManager()
    config = SessionConfig(
        name="Test Session",
        max_entries=50,
        tags=["test", "unit"],
    )
    session = await manager.create_session(config=config)

    assert session.config.name == "Test Session"
    assert session.config.max_entries == 50
    assert "test" in session.config.tags


@pytest.mark.asyncio
async def test_get_session():
    """Test retrieving a session."""
    manager = SessionManager()
    created = await manager.create_session()

    retrieved = await manager.get_session(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id


@pytest.mark.asyncio
async def test_get_nonexistent_session():
    """Test getting a session that doesn't exist."""
    manager = SessionManager()

    session = await manager.get_session("nonexistent")

    assert session is None


@pytest.mark.asyncio
async def test_switch_session():
    """Test switching between sessions."""
    manager = SessionManager()
    session1 = await manager.create_session()
    session2 = await manager.create_session()

    await manager.switch_session(session1.id)
    assert manager.current_session.id == session1.id

    await manager.switch_session(session2.id)
    assert manager.current_session.id == session2.id


@pytest.mark.asyncio
async def test_store_memory_in_session():
    """Test storing memory in a session."""
    manager = SessionManager()
    session = await manager.create_session()
    await manager.switch_session(session.id)

    entry = MemoryEntry(
        content="Test memory",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
    )

    await manager.store_memory(entry)

    assert len(session.working_memory.entries) == 1
    assert session.memory_operations == 1


@pytest.mark.asyncio
async def test_retrieve_memory_from_session():
    """Test retrieving memory from a session."""
    manager = SessionManager()
    session = await manager.create_session()
    await manager.switch_session(session.id)

    entry = MemoryEntry(
        content="Test memory content",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
    )
    await manager.store_memory(entry)

    query = MemoryQuery(
        query_text="test",
        top_k=10,
    )
    result = await manager.retrieve_memory(query)

    assert len(result.entries) > 0


@pytest.mark.asyncio
async def test_session_isolation():
    """Test that sessions are isolated."""
    manager = SessionManager()
    session1 = await manager.create_session()
    session2 = await manager.create_session()

    entry1 = MemoryEntry(
        content="Session 1 memory",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
    )
    entry2 = MemoryEntry(
        content="Session 2 memory",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
    )

    await manager.store_memory(entry1, session_id=session1.id)
    await manager.store_memory(entry2, session_id=session2.id)

    assert len(session1.working_memory.entries) == 1
    assert len(session2.working_memory.entries) == 1
    assert session1.working_memory.entries[0].content == "Session 1 memory"
    assert session2.working_memory.entries[0].content == "Session 2 memory"


@pytest.mark.asyncio
async def test_clear_session():
    """Test clearing a session."""
    manager = SessionManager()
    session = await manager.create_session()
    await manager.switch_session(session.id)

    entry = MemoryEntry(
        content="Test memory",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
    )
    await manager.store_memory(entry)

    await manager.clear_session()

    assert len(session.working_memory.entries) == 0


@pytest.mark.asyncio
async def test_pause_resume_session():
    """Test pausing and resuming a session."""
    manager = SessionManager()
    session = await manager.create_session()

    await manager.pause_session(session.id)
    assert session.state == SessionState.PAUSED

    await manager.resume_session(session.id)
    assert session.state == SessionState.ACTIVE


@pytest.mark.asyncio
async def test_archive_session():
    """Test archiving a session."""
    manager = SessionManager()
    session = await manager.create_session()
    session_id = session.id

    await manager.archive_session(session_id)

    assert session_id not in manager._sessions


@pytest.mark.asyncio
async def test_delete_session():
    """Test deleting a session."""
    manager = SessionManager()
    session = await manager.create_session()
    session_id = session.id

    result = await manager.delete_session(session_id)

    assert result is True
    assert session_id not in manager._sessions


@pytest.mark.asyncio
async def test_list_sessions():
    """Test listing sessions."""
    manager = SessionManager()
    await manager.create_session(config=SessionConfig(name="Session 1"))
    await manager.create_session(config=SessionConfig(name="Session 2"))

    sessions = manager.list_sessions()

    assert len(sessions) == 2
    names = [s["name"] for s in sessions]
    assert "Session 1" in names
    assert "Session 2" in names


@pytest.mark.asyncio
async def test_list_sessions_by_state():
    """Test listing sessions filtered by state."""
    manager = SessionManager()
    session1 = await manager.create_session()
    session2 = await manager.create_session()

    await manager.pause_session(session2.id)

    active = manager.list_sessions(state=SessionState.ACTIVE)
    paused = manager.list_sessions(state=SessionState.PAUSED)

    assert len(active) == 1
    assert len(paused) == 1


@pytest.mark.asyncio
async def test_max_sessions_limit():
    """Test maximum sessions limit."""
    manager = SessionManager(max_sessions=2)

    await manager.create_session()
    await manager.create_session()

    with pytest.raises(ValueError, match="Maximum sessions"):
        await manager.create_session()


@pytest.mark.asyncio
async def test_session_stats():
    """Test session manager statistics."""
    manager = SessionManager()
    session = await manager.create_session()
    await manager.switch_session(session.id)

    entry = MemoryEntry(
        content="Test memory",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
    )
    await manager.store_memory(entry)

    stats = manager.get_stats()

    assert stats["total_sessions"] == 1
    assert stats["total_memories"] == 1
    assert stats["total_operations"] == 1


@pytest.mark.asyncio
async def test_export_session():
    """Test exporting a session."""
    manager = SessionManager()
    session = await manager.create_session(
        config=SessionConfig(name="Export Test")
    )
    await manager.switch_session(session.id)

    entry = MemoryEntry(
        content="Exportable memory",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.7),
    )
    await manager.store_memory(entry)

    data = await manager.export_session(session.id)

    assert data["name"] == "Export Test"
    assert len(data["memories"]) == 1
    assert data["memories"][0]["content"] == "Exportable memory"


@pytest.mark.asyncio
async def test_import_session():
    """Test importing a session."""
    manager = SessionManager()

    data = {
        "id": "imported-session",
        "name": "Imported Session",
        "config": {"max_entries": 50},
        "memories": [
            {"content": "Imported memory 1", "importance": 0.5},
            {"content": "Imported memory 2", "importance": 0.8},
        ],
    }

    session = await manager.import_session(data)

    assert session.id == "imported-session"
    assert session.config.name == "Imported Session"
    assert len(session.working_memory.entries) == 2


def test_session_to_dict():
    """Test session serialization."""
    config = SessionConfig(name="Test", tags=["a", "b"])
    session = SessionState.ACTIVE

    # Create minimal session for testing
    from memoryforge.session.manager import Session
    s = Session(config=config)

    data = s.to_dict()

    assert data["name"] == "Test"
    assert data["state"] == "active"
    assert data["config"]["tags"] == ["a", "b"]
