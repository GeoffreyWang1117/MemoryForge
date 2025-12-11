"""Memory versioning and change history tracking."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import structlog

from memoryforge.core.types import MemoryEntry

logger = structlog.get_logger()


class ChangeType(Enum):
    """Types of changes to memory entries."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RESTORED = "restored"
    IMPORTANCE_CHANGED = "importance_changed"
    TAGS_CHANGED = "tags_changed"
    METADATA_CHANGED = "metadata_changed"
    CONSOLIDATED = "consolidated"


@dataclass
class VersionDiff:
    """Difference between two versions."""

    field: str
    old_value: Any
    new_value: Any

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "old_value": self._serialize(self.old_value),
            "new_value": self._serialize(self.new_value),
        }

    def _serialize(self, value: Any) -> Any:
        """Serialize value for JSON."""
        if isinstance(value, (list, dict, str, int, float, bool, type(None))):
            return value
        return str(value)


@dataclass
class MemoryVersion:
    """A version snapshot of a memory entry."""

    id: UUID = field(default_factory=uuid4)
    memory_id: UUID = field(default_factory=uuid4)
    version_number: int = 1
    change_type: ChangeType = ChangeType.CREATED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Snapshot of the entry at this version
    content: str = ""
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Change details
    diffs: list[VersionDiff] = field(default_factory=list)
    changed_by: str = "system"
    comment: str = ""

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "memory_id": str(self.memory_id),
            "version_number": self.version_number,
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "importance": self.importance,
            "tags": self.tags,
            "metadata": self.metadata,
            "diffs": [d.to_dict() for d in self.diffs],
            "changed_by": self.changed_by,
            "comment": self.comment,
        }

    @classmethod
    def from_entry(
        cls,
        entry: MemoryEntry,
        change_type: ChangeType,
        version_number: int = 1,
        diffs: list[VersionDiff] | None = None,
        changed_by: str = "system",
        comment: str = "",
    ) -> "MemoryVersion":
        """Create a version from a memory entry."""
        return cls(
            memory_id=entry.id,
            version_number=version_number,
            change_type=change_type,
            content=entry.content,
            importance=entry.importance.base_score,
            tags=entry.tags.copy(),
            metadata=entry.metadata.copy(),
            diffs=diffs or [],
            changed_by=changed_by,
            comment=comment,
        )


class VersionManager:
    """Manages version history for memory entries.

    Provides:
    - Automatic version tracking
    - Change diff calculation
    - Version rollback
    - History browsing
    """

    def __init__(
        self,
        max_versions_per_entry: int = 50,
        track_all_changes: bool = True,
    ):
        """Initialize version manager.

        Args:
            max_versions_per_entry: Maximum versions to keep per entry
            track_all_changes: Track all changes or only significant ones
        """
        self._max_versions = max_versions_per_entry
        self._track_all = track_all_changes

        # Version storage: memory_id -> list of versions
        self._versions: dict[UUID, list[MemoryVersion]] = defaultdict(list)

        # Statistics
        self._total_versions = 0
        self._total_rollbacks = 0

    def track_create(
        self,
        entry: MemoryEntry,
        changed_by: str = "system",
        comment: str = "",
    ) -> MemoryVersion:
        """Track creation of a new entry.

        Args:
            entry: The created entry
            changed_by: Who made the change
            comment: Optional comment

        Returns:
            The created version
        """
        version = MemoryVersion.from_entry(
            entry=entry,
            change_type=ChangeType.CREATED,
            version_number=1,
            changed_by=changed_by,
            comment=comment or "Initial creation",
        )

        self._add_version(entry.id, version)
        return version

    def track_update(
        self,
        old_entry: MemoryEntry,
        new_entry: MemoryEntry,
        changed_by: str = "system",
        comment: str = "",
    ) -> MemoryVersion | None:
        """Track an update to an entry.

        Args:
            old_entry: Entry before update
            new_entry: Entry after update
            changed_by: Who made the change
            comment: Optional comment

        Returns:
            The created version or None if no changes
        """
        diffs = self._calculate_diffs(old_entry, new_entry)

        if not diffs and not self._track_all:
            return None

        # Determine change type
        change_type = self._determine_change_type(diffs)

        current_version = len(self._versions[old_entry.id])
        version = MemoryVersion.from_entry(
            entry=new_entry,
            change_type=change_type,
            version_number=current_version + 1,
            diffs=diffs,
            changed_by=changed_by,
            comment=comment,
        )

        self._add_version(new_entry.id, version)
        return version

    def track_delete(
        self,
        entry: MemoryEntry,
        changed_by: str = "system",
        comment: str = "",
    ) -> MemoryVersion:
        """Track deletion of an entry.

        Args:
            entry: The deleted entry
            changed_by: Who made the change
            comment: Optional comment

        Returns:
            The created version
        """
        current_version = len(self._versions[entry.id])
        version = MemoryVersion.from_entry(
            entry=entry,
            change_type=ChangeType.DELETED,
            version_number=current_version + 1,
            changed_by=changed_by,
            comment=comment or "Entry deleted",
        )

        self._add_version(entry.id, version)
        return version

    def _add_version(self, memory_id: UUID, version: MemoryVersion) -> None:
        """Add a version to storage."""
        versions = self._versions[memory_id]
        versions.append(version)

        # Enforce max versions limit
        if len(versions) > self._max_versions:
            # Keep first (creation) and last versions
            versions[:] = [versions[0]] + versions[-(self._max_versions - 1):]

        self._total_versions += 1
        logger.debug(
            "Version tracked",
            memory_id=str(memory_id)[:8],
            version=version.version_number,
            type=version.change_type.value,
        )

    def _calculate_diffs(
        self,
        old_entry: MemoryEntry,
        new_entry: MemoryEntry,
    ) -> list[VersionDiff]:
        """Calculate differences between two entries."""
        diffs = []

        if old_entry.content != new_entry.content:
            diffs.append(VersionDiff(
                field="content",
                old_value=old_entry.content,
                new_value=new_entry.content,
            ))

        if old_entry.importance.base_score != new_entry.importance.base_score:
            diffs.append(VersionDiff(
                field="importance",
                old_value=old_entry.importance.base_score,
                new_value=new_entry.importance.base_score,
            ))

        if set(old_entry.tags) != set(new_entry.tags):
            diffs.append(VersionDiff(
                field="tags",
                old_value=old_entry.tags,
                new_value=new_entry.tags,
            ))

        if old_entry.metadata != new_entry.metadata:
            diffs.append(VersionDiff(
                field="metadata",
                old_value=old_entry.metadata,
                new_value=new_entry.metadata,
            ))

        return diffs

    def _determine_change_type(self, diffs: list[VersionDiff]) -> ChangeType:
        """Determine the primary change type from diffs."""
        if not diffs:
            return ChangeType.UPDATED

        fields = {d.field for d in diffs}

        if "content" in fields:
            return ChangeType.UPDATED
        elif "importance" in fields:
            return ChangeType.IMPORTANCE_CHANGED
        elif "tags" in fields:
            return ChangeType.TAGS_CHANGED
        elif "metadata" in fields:
            return ChangeType.METADATA_CHANGED

        return ChangeType.UPDATED

    def get_history(
        self,
        memory_id: UUID,
        limit: int = 20,
    ) -> list[MemoryVersion]:
        """Get version history for an entry.

        Args:
            memory_id: ID of the memory entry
            limit: Maximum versions to return

        Returns:
            List of versions (newest first)
        """
        versions = self._versions.get(memory_id, [])
        return list(reversed(versions[-limit:]))

    def get_version(
        self,
        memory_id: UUID,
        version_number: int,
    ) -> MemoryVersion | None:
        """Get a specific version.

        Args:
            memory_id: ID of the memory entry
            version_number: Version number to retrieve

        Returns:
            The version or None if not found
        """
        versions = self._versions.get(memory_id, [])
        for version in versions:
            if version.version_number == version_number:
                return version
        return None

    def get_latest_version(self, memory_id: UUID) -> MemoryVersion | None:
        """Get the latest version of an entry.

        Args:
            memory_id: ID of the memory entry

        Returns:
            Latest version or None
        """
        versions = self._versions.get(memory_id, [])
        return versions[-1] if versions else None

    def rollback(
        self,
        memory_id: UUID,
        to_version: int,
    ) -> MemoryVersion | None:
        """Get entry state to rollback to.

        Note: This returns the version state but doesn't apply it.
        The caller is responsible for applying the rollback.

        Args:
            memory_id: ID of the memory entry
            to_version: Version number to rollback to

        Returns:
            The version to restore or None
        """
        version = self.get_version(memory_id, to_version)
        if version:
            self._total_rollbacks += 1
            logger.info(
                "Rollback requested",
                memory_id=str(memory_id)[:8],
                to_version=to_version,
            )
        return version

    def compare_versions(
        self,
        memory_id: UUID,
        version_a: int,
        version_b: int,
    ) -> list[VersionDiff]:
        """Compare two versions of an entry.

        Args:
            memory_id: ID of the memory entry
            version_a: First version number
            version_b: Second version number

        Returns:
            List of differences
        """
        v_a = self.get_version(memory_id, version_a)
        v_b = self.get_version(memory_id, version_b)

        if not v_a or not v_b:
            return []

        diffs = []

        if v_a.content != v_b.content:
            diffs.append(VersionDiff("content", v_a.content, v_b.content))

        if v_a.importance != v_b.importance:
            diffs.append(VersionDiff("importance", v_a.importance, v_b.importance))

        if v_a.tags != v_b.tags:
            diffs.append(VersionDiff("tags", v_a.tags, v_b.tags))

        if v_a.metadata != v_b.metadata:
            diffs.append(VersionDiff("metadata", v_a.metadata, v_b.metadata))

        return diffs

    def get_changes_since(
        self,
        since: datetime,
        limit: int = 100,
    ) -> list[MemoryVersion]:
        """Get all changes since a timestamp.

        Args:
            since: Start timestamp
            limit: Maximum versions to return

        Returns:
            List of versions since timestamp
        """
        all_versions = []
        for versions in self._versions.values():
            for version in versions:
                if version.timestamp >= since:
                    all_versions.append(version)

        # Sort by timestamp (newest first)
        all_versions.sort(key=lambda v: v.timestamp, reverse=True)
        return all_versions[:limit]

    def get_deleted_entries(self, limit: int = 50) -> list[MemoryVersion]:
        """Get recently deleted entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of deleted entry versions
        """
        deleted = []
        for versions in self._versions.values():
            for version in versions:
                if version.change_type == ChangeType.DELETED:
                    deleted.append(version)

        deleted.sort(key=lambda v: v.timestamp, reverse=True)
        return deleted[:limit]

    def clear_history(self, memory_id: UUID) -> int:
        """Clear version history for an entry.

        Args:
            memory_id: ID of the memory entry

        Returns:
            Number of versions cleared
        """
        if memory_id in self._versions:
            count = len(self._versions[memory_id])
            del self._versions[memory_id]
            return count
        return 0

    def get_stats(self) -> dict:
        """Get version manager statistics."""
        total_entries = len(self._versions)
        total_versions = sum(len(v) for v in self._versions.values())

        change_type_counts = defaultdict(int)
        for versions in self._versions.values():
            for version in versions:
                change_type_counts[version.change_type.value] += 1

        return {
            "tracked_entries": total_entries,
            "total_versions": total_versions,
            "total_rollbacks": self._total_rollbacks,
            "max_versions_per_entry": self._max_versions,
            "change_types": dict(change_type_counts),
            "avg_versions_per_entry": total_versions / total_entries if total_entries else 0,
        }

    def export_history(self, memory_id: UUID) -> dict:
        """Export full history for an entry.

        Args:
            memory_id: ID of the memory entry

        Returns:
            Dictionary with full version history
        """
        versions = self._versions.get(memory_id, [])
        return {
            "memory_id": str(memory_id),
            "version_count": len(versions),
            "versions": [v.to_dict() for v in versions],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
