"""Unit tests for SQLite storage backend."""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer
from memoryforge.storage.sqlite import SQLiteMemoryStore


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def store(temp_db):
    """Create a SQLite store with temporary database."""
    return SQLiteMemoryStore(temp_db)


def create_entry(
    content: str = "Test content",
    importance: float = 0.5,
    layer: MemoryLayer = MemoryLayer.WORKING,
    tags: list[str] | None = None,
) -> MemoryEntry:
    """Helper to create test entries."""
    return MemoryEntry(
        content=content,
        layer=layer,
        importance=ImportanceScore(base_score=importance),
        tags=tags or [],
    )


class TestSQLiteMemoryStore:
    """Tests for SQLite storage operations."""

    def test_init_creates_db(self, temp_db):
        """Test that initialization creates database file."""
        store = SQLiteMemoryStore(temp_db)
        assert temp_db.exists()

    def test_store_and_get(self, store):
        """Test storing and retrieving a single entry."""
        entry = create_entry("Test memory content", importance=0.8)
        store.store(entry)

        retrieved = store.get(str(entry.id))
        assert retrieved is not None
        assert retrieved.content == "Test memory content"
        assert retrieved.importance.base_score == 0.8

    def test_get_nonexistent(self, store):
        """Test getting non-existent entry returns None."""
        result = store.get(str(uuid4()))
        assert result is None

    def test_store_with_session(self, store):
        """Test storing entry with session ID."""
        entry = create_entry("Session memory")
        store.store(entry, session_id="test-session")

        entries = store.get_all(session_id="test-session")
        assert len(entries) == 1
        assert entries[0].content == "Session memory"

    def test_get_all_basic(self, store):
        """Test getting all entries."""
        entries = [
            create_entry("First"),
            create_entry("Second"),
            create_entry("Third"),
        ]
        for entry in entries:
            store.store(entry)

        retrieved = store.get_all()
        assert len(retrieved) == 3

    def test_get_all_with_limit(self, store):
        """Test getting entries with limit."""
        for i in range(10):
            store.store(create_entry(f"Entry {i}"))

        retrieved = store.get_all(limit=5)
        assert len(retrieved) == 5

    def test_get_all_with_offset(self, store):
        """Test getting entries with offset."""
        for i in range(10):
            store.store(create_entry(f"Entry {i}"))

        retrieved = store.get_all(limit=5, offset=5)
        assert len(retrieved) == 5

    def test_get_all_filter_by_layer(self, store):
        """Test filtering by memory layer."""
        store.store(create_entry("Working 1", layer=MemoryLayer.WORKING))
        store.store(create_entry("Working 2", layer=MemoryLayer.WORKING))
        store.store(create_entry("Episodic", layer=MemoryLayer.EPISODIC))

        working = store.get_all(layer=MemoryLayer.WORKING)
        assert len(working) == 2

        episodic = store.get_all(layer=MemoryLayer.EPISODIC)
        assert len(episodic) == 1

    def test_search_basic(self, store):
        """Test basic text search."""
        store.store(create_entry("Python programming tutorial"))
        store.store(create_entry("JavaScript basics"))
        store.store(create_entry("Python web frameworks"))

        results = store.search("Python")
        assert len(results) == 2
        assert all("Python" in entry.content for entry, _ in results)

    def test_search_returns_scores(self, store):
        """Test that search returns relevance scores."""
        store.store(create_entry("Python programming", importance=0.8))
        results = store.search("Python")

        assert len(results) == 1
        entry, score = results[0]
        assert score > 0

    def test_search_with_layer_filter(self, store):
        """Test search with layer filter."""
        store.store(create_entry("Working Python", layer=MemoryLayer.WORKING))
        store.store(create_entry("Episodic Python", layer=MemoryLayer.EPISODIC))

        results = store.search("Python", layer=MemoryLayer.WORKING)
        assert len(results) == 1
        assert results[0][0].layer == MemoryLayer.WORKING

    def test_delete_entry(self, store):
        """Test deleting an entry."""
        entry = create_entry("To be deleted")
        store.store(entry)

        assert store.get(str(entry.id)) is not None
        result = store.delete(str(entry.id))
        assert result is True
        assert store.get(str(entry.id)) is None

    def test_delete_nonexistent(self, store):
        """Test deleting non-existent entry."""
        result = store.delete(str(uuid4()))
        assert result is False

    def test_clear_all(self, store):
        """Test clearing all entries."""
        for i in range(5):
            store.store(create_entry(f"Entry {i}"))

        count = store.clear()
        assert count == 5
        assert len(store.get_all()) == 0

    def test_clear_by_session(self, store):
        """Test clearing entries by session."""
        store.store(create_entry("Session A"), session_id="session-a")
        store.store(create_entry("Session A 2"), session_id="session-a")
        store.store(create_entry("Session B"), session_id="session-b")

        count = store.clear(session_id="session-a")
        assert count == 2
        assert len(store.get_all(session_id="session-a")) == 0
        assert len(store.get_all(session_id="session-b")) == 1

    def test_count_total(self, store):
        """Test counting total entries."""
        for i in range(7):
            store.store(create_entry(f"Entry {i}"))

        assert store.count() == 7

    def test_count_by_layer(self, store):
        """Test counting by layer."""
        store.store(create_entry("W1", layer=MemoryLayer.WORKING))
        store.store(create_entry("W2", layer=MemoryLayer.WORKING))
        store.store(create_entry("E1", layer=MemoryLayer.EPISODIC))

        assert store.count(layer=MemoryLayer.WORKING) == 2
        assert store.count(layer=MemoryLayer.EPISODIC) == 1
        assert store.count(layer=MemoryLayer.SEMANTIC) == 0

    def test_count_by_session(self, store):
        """Test counting by session."""
        store.store(create_entry("A1"), session_id="a")
        store.store(create_entry("A2"), session_id="a")
        store.store(create_entry("B1"), session_id="b")

        assert store.count(session_id="a") == 2
        assert store.count(session_id="b") == 1


class TestSQLiteSessionManagement:
    """Tests for session management."""

    def test_create_session(self, store):
        """Test creating a session."""
        store.create_session("test-session", name="Test Session")
        sessions = store.get_sessions()

        assert len(sessions) == 1
        assert sessions[0]["id"] == "test-session"
        assert sessions[0]["name"] == "Test Session"

    def test_get_sessions_with_memory_count(self, store):
        """Test getting sessions includes memory count."""
        store.create_session("session-1")
        store.store(create_entry("Memory 1"), session_id="session-1")
        store.store(create_entry("Memory 2"), session_id="session-1")

        sessions = store.get_sessions()
        assert sessions[0]["memory_count"] == 2

    def test_delete_session(self, store):
        """Test deleting a session removes memories."""
        store.create_session("to-delete")
        store.store(create_entry("Memory"), session_id="to-delete")

        result = store.delete_session("to-delete")
        assert result is True
        assert len(store.get_sessions()) == 0
        assert len(store.get_all(session_id="to-delete")) == 0

    def test_delete_nonexistent_session(self, store):
        """Test deleting non-existent session."""
        result = store.delete_session("nonexistent")
        assert result is False


class TestSQLiteExportImport:
    """Tests for export/import functionality."""

    def test_export_basic(self, store):
        """Test basic export."""
        store.store(create_entry("Memory 1", importance=0.8, tags=["tag1"]))
        store.store(create_entry("Memory 2", importance=0.6))

        data = store.export_to_json()
        assert data["version"] == "1.0"
        assert data["count"] == 2
        assert len(data["memories"]) == 2

    def test_export_by_session(self, store):
        """Test export filtered by session."""
        store.store(create_entry("Session A"), session_id="a")
        store.store(create_entry("Session B"), session_id="b")

        data = store.export_to_json(session_id="a")
        assert data["count"] == 1
        assert data["session_id"] == "a"

    def test_import_basic(self, store):
        """Test basic import."""
        data = {
            "memories": [
                {"content": "Imported 1", "layer": "working", "importance": 0.7},
                {"content": "Imported 2", "layer": "episodic", "tags": ["imported"]},
            ]
        }

        count = store.import_from_json(data)
        assert count == 2

        entries = store.get_all()
        assert len(entries) == 2

    def test_import_with_session(self, store):
        """Test import with session ID."""
        data = {
            "memories": [
                {"content": "Session memory", "layer": "working"},
            ]
        }

        store.import_from_json(data, session_id="imported-session")
        entries = store.get_all(session_id="imported-session")
        assert len(entries) == 1

    def test_roundtrip_export_import(self, store, temp_db):
        """Test that export/import round-trip preserves data."""
        import tempfile

        original = create_entry("Round trip test", importance=0.75, tags=["test"])
        store.store(original)

        exported = store.export_to_json()

        # Create new store with unique temp file and import
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store2_path = Path(f.name)
        try:
            store2 = SQLiteMemoryStore(store2_path)
            store2.import_from_json(exported)

            imported = store2.get_all()
            assert len(imported) == 1
            assert imported[0].content == "Round trip test"
            assert imported[0].importance.base_score == 0.75
            assert "test" in imported[0].tags
        finally:
            if store2_path.exists():
                store2_path.unlink()


class TestSQLiteStats:
    """Tests for statistics."""

    def test_get_stats_empty(self, store):
        """Test stats on empty database."""
        stats = store.get_stats()
        assert stats["total_memories"] == 0
        assert stats["session_count"] == 0

    def test_get_stats_with_data(self, store):
        """Test stats with data."""
        store.store(create_entry("W1", layer=MemoryLayer.WORKING, importance=0.8))
        store.store(create_entry("W2", layer=MemoryLayer.WORKING, importance=0.6))
        store.store(create_entry("E1", layer=MemoryLayer.EPISODIC, importance=0.4))
        store.create_session("session-1")

        stats = store.get_stats()
        assert stats["total_memories"] == 3
        assert stats["by_layer"]["working"] == 2
        assert stats["by_layer"]["episodic"] == 1
        assert stats["session_count"] == 1
        assert 0.5 < stats["avg_importance"] < 0.7


class TestSQLiteEdgeCases:
    """Tests for edge cases."""

    def test_store_with_embedding(self, store):
        """Test storing entry with embedding."""
        entry = create_entry("With embedding")
        entry.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        store.store(entry)

        retrieved = store.get(str(entry.id))
        assert retrieved.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_store_with_metadata(self, store):
        """Test storing entry with metadata."""
        entry = create_entry("With metadata")
        entry.metadata = {"key": "value", "nested": {"a": 1}}
        store.store(entry)

        retrieved = store.get(str(entry.id))
        assert retrieved.metadata["key"] == "value"
        assert retrieved.metadata["nested"]["a"] == 1

    def test_store_with_source_id(self, store):
        """Test storing entry with source reference."""
        source = create_entry("Source")
        store.store(source)

        child = create_entry("Child")
        child.source_id = source.id
        store.store(child)

        retrieved = store.get(str(child.id))
        assert retrieved.source_id == source.id

    def test_update_existing_entry(self, store):
        """Test updating an existing entry."""
        entry = create_entry("Original", importance=0.5)
        store.store(entry)

        # Update with same ID
        entry.content = "Updated"
        entry.importance = ImportanceScore(base_score=0.9)
        store.store(entry)

        retrieved = store.get(str(entry.id))
        assert retrieved.content == "Updated"
        assert retrieved.importance.base_score == 0.9

    def test_special_characters_in_content(self, store):
        """Test handling special characters."""
        content = "Special: 'quotes', \"double\", emoji 🚀, unicode: café"
        entry = create_entry(content)
        store.store(entry)

        retrieved = store.get(str(entry.id))
        assert retrieved.content == content

    def test_empty_content(self, store):
        """Test handling empty content."""
        entry = create_entry("")
        store.store(entry)

        retrieved = store.get(str(entry.id))
        assert retrieved.content == ""

    def test_very_long_content(self, store):
        """Test handling very long content."""
        content = "x" * 100000
        entry = create_entry(content)
        store.store(entry)

        retrieved = store.get(str(entry.id))
        assert len(retrieved.content) == 100000
