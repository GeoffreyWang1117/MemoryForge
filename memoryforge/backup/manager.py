"""Backup and restore manager for memory data."""

import gzip
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

import structlog

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer
from memoryforge.storage.sqlite import SQLiteMemoryStore

logger = structlog.get_logger()


@dataclass
class BackupConfig:
    """Configuration for backup operations."""

    # Backup directory
    backup_dir: Path = field(default_factory=lambda: Path("backups"))

    # Compression
    compress: bool = True

    # Include embeddings in backup
    include_embeddings: bool = False

    # Maximum backups to keep
    max_backups: int = 10

    # Backup file prefix
    file_prefix: str = "memoryforge_backup"


@dataclass
class BackupMetadata:
    """Metadata for a backup file."""

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"
    memory_count: int = 0
    session_count: int = 0
    file_size: int = 0
    compressed: bool = True
    source_db: str = ""
    checksum: str = ""

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "memory_count": self.memory_count,
            "session_count": self.session_count,
            "file_size": self.file_size,
            "compressed": self.compressed,
            "source_db": self.source_db,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BackupMetadata":
        return cls(
            id=UUID(data["id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            version=data.get("version", "1.0"),
            memory_count=data.get("memory_count", 0),
            session_count=data.get("session_count", 0),
            file_size=data.get("file_size", 0),
            compressed=data.get("compressed", True),
            source_db=data.get("source_db", ""),
            checksum=data.get("checksum", ""),
        )


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    success: bool
    memories_restored: int = 0
    sessions_restored: int = 0
    errors: list[str] = field(default_factory=list)
    backup_metadata: BackupMetadata | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "memories_restored": self.memories_restored,
            "sessions_restored": self.sessions_restored,
            "errors": self.errors,
            "backup_metadata": self.backup_metadata.to_dict() if self.backup_metadata else None,
        }


class BackupManager:
    """Manages backup and restore operations for MemoryForge.

    Provides:
    - Full database backups
    - Incremental backups
    - Compressed storage
    - Backup rotation
    - Point-in-time restore
    """

    def __init__(
        self,
        config: BackupConfig | None = None,
    ):
        """Initialize backup manager.

        Args:
            config: Backup configuration
        """
        self._config = config or BackupConfig()
        self._ensure_backup_dir()

    def _ensure_backup_dir(self) -> None:
        """Ensure backup directory exists."""
        self._config.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(
        self,
        store: SQLiteMemoryStore,
        description: str = "",
    ) -> tuple[Path, BackupMetadata]:
        """Create a full backup of the memory store.

        Args:
            store: SQLite store to backup
            description: Optional backup description

        Returns:
            Tuple of (backup_path, metadata)
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self._config.file_prefix}_{timestamp}"

        if self._config.compress:
            filename += ".json.gz"
        else:
            filename += ".json"

        backup_path = self._config.backup_dir / filename

        # Collect data
        memories = store.get_all(limit=100000)
        sessions = store.get_sessions()

        backup_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "description": description,
                "source_db": str(store._db_path),
            },
            "memories": [
                self._serialize_memory(m)
                for m in memories
            ],
            "sessions": sessions,
        }

        # Write backup
        json_data = json.dumps(backup_data, indent=2, ensure_ascii=False)

        if self._config.compress:
            with gzip.open(backup_path, "wt", encoding="utf-8") as f:
                f.write(json_data)
        else:
            backup_path.write_text(json_data, encoding="utf-8")

        # Create metadata
        metadata = BackupMetadata(
            memory_count=len(memories),
            session_count=len(sessions),
            file_size=backup_path.stat().st_size,
            compressed=self._config.compress,
            source_db=str(store._db_path),
            checksum=self._calculate_checksum(backup_path),
        )

        # Rotate old backups
        self._rotate_backups()

        logger.info(
            "Backup created",
            path=str(backup_path),
            memories=metadata.memory_count,
            size=metadata.file_size,
        )

        return backup_path, metadata

    def restore_backup(
        self,
        backup_path: Path | str,
        target_store: SQLiteMemoryStore,
        clear_existing: bool = False,
    ) -> RestoreResult:
        """Restore from a backup file.

        Args:
            backup_path: Path to backup file
            target_store: Store to restore into
            clear_existing: Clear existing data before restore

        Returns:
            Restore result
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            return RestoreResult(
                success=False,
                errors=[f"Backup file not found: {backup_path}"],
            )

        try:
            # Load backup data
            if backup_path.suffix == ".gz" or str(backup_path).endswith(".json.gz"):
                with gzip.open(backup_path, "rt", encoding="utf-8") as f:
                    backup_data = json.load(f)
            else:
                backup_data = json.loads(backup_path.read_text(encoding="utf-8"))

            # Clear existing if requested
            if clear_existing:
                target_store.clear()
                for session in target_store.get_sessions():
                    target_store.delete_session(session["id"])

            # Restore sessions first
            sessions_restored = 0
            for session in backup_data.get("sessions", []):
                try:
                    target_store.create_session(
                        session["id"],
                        session.get("name", ""),
                    )
                    sessions_restored += 1
                except Exception as e:
                    logger.warning(f"Failed to restore session: {e}")

            # Restore memories
            memories_restored = 0
            errors = []

            for mem_data in backup_data.get("memories", []):
                try:
                    entry = self._deserialize_memory(mem_data)
                    target_store.store(entry, session_id=mem_data.get("session_id"))
                    memories_restored += 1
                except Exception as e:
                    errors.append(f"Memory {mem_data.get('id', 'unknown')}: {str(e)}")

            # Build metadata from backup
            metadata = BackupMetadata(
                created_at=datetime.fromisoformat(
                    backup_data["metadata"]["created_at"]
                ),
                version=backup_data["metadata"].get("version", "1.0"),
                memory_count=len(backup_data.get("memories", [])),
                session_count=len(backup_data.get("sessions", [])),
                file_size=backup_path.stat().st_size,
                compressed=backup_path.suffix == ".gz",
            )

            result = RestoreResult(
                success=True,
                memories_restored=memories_restored,
                sessions_restored=sessions_restored,
                errors=errors,
                backup_metadata=metadata,
            )

            logger.info(
                "Backup restored",
                memories=memories_restored,
                sessions=sessions_restored,
                errors=len(errors),
            )

            return result

        except Exception as e:
            logger.error("Restore failed", error=str(e))
            return RestoreResult(
                success=False,
                errors=[str(e)],
            )

    def list_backups(self) -> list[dict]:
        """List available backups.

        Returns:
            List of backup info dictionaries
        """
        backups = []

        for path in self._config.backup_dir.glob(f"{self._config.file_prefix}*"):
            try:
                stat = path.stat()
                backups.append({
                    "path": str(path),
                    "filename": path.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "compressed": path.suffix == ".gz" or str(path).endswith(".json.gz"),
                })
            except Exception as e:
                logger.warning(f"Error reading backup {path}: {e}")

        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b["created"], reverse=True)
        return backups

    def get_backup_info(self, backup_path: Path | str) -> BackupMetadata | None:
        """Get metadata for a backup file.

        Args:
            backup_path: Path to backup file

        Returns:
            Backup metadata or None
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            return None

        try:
            if backup_path.suffix == ".gz" or str(backup_path).endswith(".json.gz"):
                with gzip.open(backup_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(backup_path.read_text(encoding="utf-8"))

            metadata = BackupMetadata(
                created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
                version=data["metadata"].get("version", "1.0"),
                memory_count=len(data.get("memories", [])),
                session_count=len(data.get("sessions", [])),
                file_size=backup_path.stat().st_size,
                compressed=backup_path.suffix == ".gz",
                source_db=data["metadata"].get("source_db", ""),
            )

            return metadata

        except Exception as e:
            logger.error(f"Error reading backup info: {e}")
            return None

    def delete_backup(self, backup_path: Path | str) -> bool:
        """Delete a backup file.

        Args:
            backup_path: Path to backup file

        Returns:
            True if deleted
        """
        backup_path = Path(backup_path)

        if backup_path.exists():
            backup_path.unlink()
            logger.info(f"Backup deleted: {backup_path}")
            return True

        return False

    def verify_backup(self, backup_path: Path | str) -> tuple[bool, str]:
        """Verify a backup file's integrity.

        Args:
            backup_path: Path to backup file

        Returns:
            Tuple of (is_valid, message)
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            return False, "Backup file not found"

        try:
            if backup_path.suffix == ".gz" or str(backup_path).endswith(".json.gz"):
                with gzip.open(backup_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(backup_path.read_text(encoding="utf-8"))

            # Verify structure
            if "metadata" not in data:
                return False, "Missing metadata section"

            if "memories" not in data:
                return False, "Missing memories section"

            # Verify memories can be deserialized
            for mem in data["memories"][:5]:  # Check first 5
                self._deserialize_memory(mem)

            return True, f"Valid backup with {len(data['memories'])} memories"

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Verification error: {e}"

    def _serialize_memory(self, entry: MemoryEntry) -> dict:
        """Serialize a memory entry for backup."""
        data = {
            "id": str(entry.id),
            "content": entry.content,
            "layer": entry.layer.value,
            "importance": {
                "base_score": entry.importance.base_score,
                "recency_weight": entry.importance.recency_weight,
                "access_count": entry.importance.access_count,
                "last_accessed": entry.importance.last_accessed.isoformat(),
            },
            "tags": entry.tags,
            "metadata": entry.metadata,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
        }

        if self._config.include_embeddings and entry.embedding:
            data["embedding"] = entry.embedding

        if entry.source_id:
            data["source_id"] = str(entry.source_id)

        return data

    def _deserialize_memory(self, data: dict) -> MemoryEntry:
        """Deserialize a memory entry from backup."""
        importance = data.get("importance", {})

        return MemoryEntry(
            id=UUID(data["id"]),
            content=data["content"],
            layer=MemoryLayer(data["layer"]),
            importance=ImportanceScore(
                base_score=importance.get("base_score", 0.5),
                recency_weight=importance.get("recency_weight", 1.0),
                access_count=importance.get("access_count", 0),
                last_accessed=datetime.fromisoformat(
                    importance.get("last_accessed", datetime.now(timezone.utc).isoformat())
                ),
            ),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            source_id=UUID(data["source_id"]) if data.get("source_id") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def _rotate_backups(self) -> int:
        """Remove old backups exceeding max_backups.

        Returns:
            Number of backups removed
        """
        backups = self.list_backups()
        removed = 0

        while len(backups) > self._config.max_backups:
            oldest = backups.pop()
            Path(oldest["path"]).unlink()
            removed += 1
            logger.info(f"Rotated old backup: {oldest['filename']}")

        return removed

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate checksum for a backup file."""
        import hashlib

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()[:16]

    def get_stats(self) -> dict:
        """Get backup statistics."""
        backups = self.list_backups()
        total_size = sum(b["size"] for b in backups)

        return {
            "backup_count": len(backups),
            "total_size": total_size,
            "backup_dir": str(self._config.backup_dir),
            "max_backups": self._config.max_backups,
            "compression_enabled": self._config.compress,
        }
