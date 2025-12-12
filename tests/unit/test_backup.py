"""Unit tests for backup manager module."""

import gzip
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer
from memoryforge.backup.manager import (
    BackupConfig,
    BackupManager,
    BackupMetadata,
    RestoreResult,
)
from memoryforge.storage.sqlite import SQLiteMemoryStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary database file."""
    return temp_dir / "test.db"


@pytest.fixture
def store(temp_db):
    """Create a SQLite store with temporary database."""
    return SQLiteMemoryStore(temp_db)


@pytest.fixture
def backup_config(temp_dir):
    """Create a backup configuration."""
    return BackupConfig(
        backup_dir=temp_dir / "backups",
        compress=True,
        max_backups=5,
    )


@pytest.fixture
def manager(backup_config):
    """Create a backup manager."""
    return BackupManager(config=backup_config)


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


class TestBackupConfig:
    """Tests for BackupConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = BackupConfig()

        assert config.backup_dir == Path("backups")
        assert config.compress is True
        assert config.include_embeddings is False
        assert config.max_backups == 10
        assert config.file_prefix == "memoryforge_backup"

    def test_custom_config(self):
        """Test custom configuration."""
        custom_dir = Path("/custom/backup/dir")
        config = BackupConfig(
            backup_dir=custom_dir,
            compress=False,
            include_embeddings=True,
            max_backups=5,
            file_prefix="custom_prefix",
        )

        assert config.backup_dir == custom_dir
        assert config.compress is False
        assert config.include_embeddings is True
        assert config.max_backups == 5
        assert config.file_prefix == "custom_prefix"


class TestBackupMetadata:
    """Tests for BackupMetadata dataclass."""

    def test_creation(self):
        """Test creating metadata."""
        metadata = BackupMetadata(
            memory_count=100,
            session_count=5,
            file_size=1024,
            compressed=True,
            source_db="/path/to/db",
        )

        assert metadata.memory_count == 100
        assert metadata.session_count == 5
        assert metadata.file_size == 1024
        assert metadata.compressed is True
        assert metadata.source_db == "/path/to/db"
        assert metadata.id is not None
        assert metadata.created_at is not None

    def test_default_values(self):
        """Test default values."""
        metadata = BackupMetadata()

        assert metadata.version == "1.0"
        assert metadata.memory_count == 0
        assert metadata.session_count == 0
        assert metadata.file_size == 0
        assert metadata.compressed is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = BackupMetadata(
            memory_count=50,
            session_count=3,
            source_db="test.db",
        )

        result = metadata.to_dict()

        assert result["memory_count"] == 50
        assert result["session_count"] == 3
        assert result["source_db"] == "test.db"
        assert "id" in result
        assert "created_at" in result
        assert "version" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "memory_count": 25,
            "session_count": 2,
            "file_size": 512,
            "compressed": False,
            "source_db": "source.db",
            "checksum": "abc123",
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.memory_count == 25
        assert metadata.session_count == 2
        assert metadata.file_size == 512
        assert metadata.compressed is False
        assert metadata.source_db == "source.db"
        assert metadata.checksum == "abc123"


class TestRestoreResult:
    """Tests for RestoreResult dataclass."""

    def test_success_result(self):
        """Test successful restore result."""
        result = RestoreResult(
            success=True,
            memories_restored=10,
            sessions_restored=2,
        )

        assert result.success is True
        assert result.memories_restored == 10
        assert result.sessions_restored == 2
        assert result.errors == []

    def test_failure_result(self):
        """Test failed restore result."""
        result = RestoreResult(
            success=False,
            errors=["Error 1", "Error 2"],
        )

        assert result.success is False
        assert len(result.errors) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = BackupMetadata(memory_count=5)
        result = RestoreResult(
            success=True,
            memories_restored=5,
            sessions_restored=1,
            backup_metadata=metadata,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["memories_restored"] == 5
        assert data["backup_metadata"] is not None


class TestBackupManager:
    """Tests for BackupManager."""

    def test_init_default(self, temp_dir):
        """Test default initialization."""
        config = BackupConfig(backup_dir=temp_dir / "backups")
        manager = BackupManager(config=config)

        assert manager._config.backup_dir.exists()

    def test_init_creates_backup_dir(self, temp_dir):
        """Test that init creates backup directory."""
        backup_dir = temp_dir / "new_backup_dir"
        config = BackupConfig(backup_dir=backup_dir)
        manager = BackupManager(config=config)

        assert backup_dir.exists()

    def test_create_backup_compressed(self, manager, store):
        """Test creating a compressed backup."""
        # Add some data
        store.store(create_entry("Memory 1", importance=0.8))
        store.store(create_entry("Memory 2", importance=0.6))
        store.create_session("session-1", name="Test Session")

        path, metadata = manager.create_backup(store, description="Test backup")

        assert path.exists()
        assert str(path).endswith(".json.gz")
        assert metadata.memory_count == 2
        assert metadata.compressed is True
        assert metadata.file_size > 0

    def test_create_backup_uncompressed(self, temp_dir, store):
        """Test creating an uncompressed backup."""
        config = BackupConfig(
            backup_dir=temp_dir / "backups",
            compress=False,
        )
        manager = BackupManager(config=config)

        store.store(create_entry("Test memory"))

        path, metadata = manager.create_backup(store)

        assert path.exists()
        assert str(path).endswith(".json")
        assert not str(path).endswith(".json.gz")
        assert metadata.compressed is False

    def test_create_backup_with_embeddings(self, temp_dir, store):
        """Test backup includes embeddings when configured."""
        config = BackupConfig(
            backup_dir=temp_dir / "backups",
            include_embeddings=True,
            compress=False,
        )
        manager = BackupManager(config=config)

        entry = create_entry("With embedding")
        entry.embedding = [0.1, 0.2, 0.3]
        store.store(entry)

        path, _ = manager.create_backup(store)

        # Read and verify embedding is included
        data = json.loads(path.read_text())
        assert "embedding" in data["memories"][0]
        assert data["memories"][0]["embedding"] == [0.1, 0.2, 0.3]

    def test_restore_backup(self, manager, store, temp_dir):
        """Test restoring from backup."""
        # Create backup with data
        store.store(create_entry("Memory to restore", importance=0.9))
        store.store(create_entry("Another memory", importance=0.7, tags=["tag1"]))
        store.create_session("session-1", name="Test Session")
        store.store(create_entry("Session memory"), session_id="session-1")

        path, _ = manager.create_backup(store)

        # Create new store and restore
        new_db = temp_dir / "restored.db"
        new_store = SQLiteMemoryStore(new_db)

        result = manager.restore_backup(path, new_store)

        assert result.success is True
        assert result.memories_restored == 3
        assert result.backup_metadata is not None

        # Verify data
        memories = new_store.get_all()
        assert len(memories) == 3

    def test_restore_backup_clear_existing(self, manager, store, temp_dir):
        """Test restoring with clear_existing option."""
        # Create backup
        store.store(create_entry("Backup memory"))
        path, _ = manager.create_backup(store)

        # New store with existing data
        new_db = temp_dir / "existing.db"
        new_store = SQLiteMemoryStore(new_db)
        new_store.store(create_entry("Existing memory"))

        result = manager.restore_backup(path, new_store, clear_existing=True)

        assert result.success is True
        memories = new_store.get_all()
        assert len(memories) == 1
        assert memories[0].content == "Backup memory"

    def test_restore_nonexistent_backup(self, manager, store):
        """Test restoring from non-existent file."""
        result = manager.restore_backup("/nonexistent/backup.json", store)

        assert result.success is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()

    def test_list_backups(self, manager, store):
        """Test listing available backups."""
        # Create a backup and verify listing
        store.store(create_entry("Memory"))
        manager.create_backup(store, description="Backup 1")

        backups = manager.list_backups()

        assert len(backups) >= 1
        assert "path" in backups[0]
        assert "filename" in backups[0]
        assert "size" in backups[0]
        assert "created" in backups[0]
        assert "compressed" in backups[0]

    def test_list_backups_empty(self, manager):
        """Test listing when no backups exist."""
        backups = manager.list_backups()
        assert backups == []

    def test_get_backup_info(self, manager, store):
        """Test getting backup metadata."""
        store.store(create_entry("Test"))
        path, original_meta = manager.create_backup(store)

        info = manager.get_backup_info(path)

        assert info is not None
        assert info.memory_count == 1
        assert info.version == "1.0"

    def test_get_backup_info_nonexistent(self, manager):
        """Test getting info for non-existent backup."""
        info = manager.get_backup_info("/nonexistent/backup.json")
        assert info is None

    def test_delete_backup(self, manager, store):
        """Test deleting a backup."""
        store.store(create_entry("Test"))
        path, _ = manager.create_backup(store)

        assert path.exists()
        result = manager.delete_backup(path)

        assert result is True
        assert not path.exists()

    def test_delete_nonexistent_backup(self, manager):
        """Test deleting non-existent backup."""
        result = manager.delete_backup("/nonexistent/backup.json")
        assert result is False

    def test_verify_backup_valid(self, manager, store):
        """Test verifying a valid backup."""
        store.store(create_entry("Valid memory"))
        path, _ = manager.create_backup(store)

        is_valid, message = manager.verify_backup(path)

        assert is_valid is True
        assert "Valid backup" in message
        assert "1 memories" in message

    def test_verify_backup_nonexistent(self, manager):
        """Test verifying non-existent backup."""
        is_valid, message = manager.verify_backup("/nonexistent/backup.json")

        assert is_valid is False
        assert "not found" in message.lower()

    def test_verify_backup_invalid_json(self, manager, temp_dir):
        """Test verifying backup with invalid JSON."""
        invalid_backup = temp_dir / "backups" / "memoryforge_backup_invalid.json.gz"
        invalid_backup.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(invalid_backup, "wt") as f:
            f.write("not valid json{")

        is_valid, message = manager.verify_backup(invalid_backup)

        assert is_valid is False
        assert "Invalid JSON" in message or "error" in message.lower()

    def test_verify_backup_missing_metadata(self, manager, temp_dir):
        """Test verifying backup without metadata."""
        invalid_backup = temp_dir / "backups" / "memoryforge_backup_no_meta.json"
        invalid_backup.parent.mkdir(parents=True, exist_ok=True)

        data = {"memories": []}  # Missing metadata
        invalid_backup.write_text(json.dumps(data))

        is_valid, message = manager.verify_backup(invalid_backup)

        assert is_valid is False
        assert "metadata" in message.lower()

    def test_backup_rotation(self, temp_dir, store):
        """Test that old backups are rotated by manually creating backup files."""
        from datetime import datetime, timezone, timedelta

        config = BackupConfig(
            backup_dir=temp_dir / "backups",
            max_backups=3,
        )
        manager = BackupManager(config=config)

        # Manually create backup files to simulate rotation scenario
        backup_dir = config.backup_dir
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create 5 fake backup files with different timestamps
        for i in range(5):
            timestamp = (datetime.now(timezone.utc) - timedelta(hours=5-i)).strftime("%Y%m%d_%H%M%S")
            filename = f"{config.file_prefix}_{timestamp}_{i}.json.gz"
            backup_path = backup_dir / filename

            # Create valid backup content
            data = {
                "metadata": {
                    "version": "1.0",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                "memories": [],
                "sessions": [],
            }
            import gzip
            import json
            with gzip.open(backup_path, "wt", encoding="utf-8") as f:
                json.dump(data, f)

        # Verify we have 5 backups
        initial_backups = manager.list_backups()
        assert len(initial_backups) == 5

        # Now create a real backup which should trigger rotation
        store.store(create_entry("Test"))
        manager.create_backup(store)

        # After rotation, should be limited to max_backups
        backups = manager.list_backups()
        assert len(backups) == 3

    def test_get_stats(self, manager, store):
        """Test getting backup statistics."""
        store.store(create_entry("Test"))
        manager.create_backup(store)

        stats = manager.get_stats()

        assert "backup_count" in stats
        assert "total_size" in stats
        assert "backup_dir" in stats
        assert "max_backups" in stats
        assert "compression_enabled" in stats
        assert stats["backup_count"] >= 1

    def test_get_stats_empty(self, manager):
        """Test getting stats with no backups."""
        stats = manager.get_stats()

        assert stats["backup_count"] == 0
        assert stats["total_size"] == 0


class TestBackupManagerEdgeCases:
    """Tests for edge cases."""

    def test_backup_with_special_characters(self, manager, store):
        """Test backup with special characters in content."""
        store.store(create_entry("Special: 'quotes', \"double\", 中文, emoji 🚀"))
        path, _ = manager.create_backup(store)

        # Restore and verify
        new_store = SQLiteMemoryStore(path.parent / "test_restore.db")
        result = manager.restore_backup(path, new_store)

        assert result.success is True
        memories = new_store.get_all()
        assert "中文" in memories[0].content

    def test_backup_with_metadata(self, manager, store):
        """Test backup with entry metadata."""
        entry = create_entry("With metadata")
        entry.metadata = {"key": "value", "nested": {"a": 1}}
        store.store(entry)

        path, _ = manager.create_backup(store)

        new_store = SQLiteMemoryStore(path.parent / "test_restore.db")
        result = manager.restore_backup(path, new_store)

        assert result.success is True
        restored = new_store.get_all()[0]
        assert restored.metadata["key"] == "value"
        assert restored.metadata["nested"]["a"] == 1

    def test_backup_all_memory_layers(self, manager, store):
        """Test backup with all memory layers."""
        store.store(create_entry("Working", layer=MemoryLayer.WORKING))
        store.store(create_entry("Episodic", layer=MemoryLayer.EPISODIC))
        store.store(create_entry("Semantic", layer=MemoryLayer.SEMANTIC))

        path, metadata = manager.create_backup(store)

        assert metadata.memory_count == 3

        new_store = SQLiteMemoryStore(path.parent / "test_restore.db")
        result = manager.restore_backup(path, new_store)

        assert result.memories_restored == 3

        # Verify layers
        layers = {m.layer for m in new_store.get_all()}
        assert MemoryLayer.WORKING in layers
        assert MemoryLayer.EPISODIC in layers
        assert MemoryLayer.SEMANTIC in layers

    def test_backup_empty_store(self, manager, store):
        """Test backing up empty store."""
        path, metadata = manager.create_backup(store)

        assert path.exists()
        assert metadata.memory_count == 0

    def test_restore_preserves_importance(self, manager, store):
        """Test that restore preserves importance scores."""
        entry = create_entry("Important", importance=0.95)
        store.store(entry)

        path, _ = manager.create_backup(store)

        new_store = SQLiteMemoryStore(path.parent / "test_restore.db")
        manager.restore_backup(path, new_store)

        restored = new_store.get_all()[0]
        assert restored.importance.base_score == 0.95

    def test_restore_preserves_tags(self, manager, store):
        """Test that restore preserves tags."""
        entry = create_entry("Tagged", tags=["tag1", "tag2", "tag3"])
        store.store(entry)

        path, _ = manager.create_backup(store)

        new_store = SQLiteMemoryStore(path.parent / "test_restore.db")
        manager.restore_backup(path, new_store)

        restored = new_store.get_all()[0]
        assert set(restored.tags) == {"tag1", "tag2", "tag3"}

    def test_checksum_calculation(self, manager, store):
        """Test that checksum is calculated."""
        store.store(create_entry("Test"))
        path, metadata = manager.create_backup(store)

        assert metadata.checksum != ""
        assert len(metadata.checksum) == 16  # First 16 chars of sha256

    def test_multiple_sessions_backup(self, manager, store):
        """Test backup with multiple sessions."""
        store.create_session("session-1", name="Session 1")
        store.create_session("session-2", name="Session 2")
        store.store(create_entry("Memory 1"), session_id="session-1")
        store.store(create_entry("Memory 2"), session_id="session-2")

        path, metadata = manager.create_backup(store)

        assert metadata.session_count == 2
        assert metadata.memory_count == 2
