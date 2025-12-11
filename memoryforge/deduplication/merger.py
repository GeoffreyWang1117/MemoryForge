"""Merge strategies for duplicate memory entries."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable
from uuid import UUID, uuid4

import structlog

from memoryforge.core.types import MemoryEntry, ImportanceScore, MemoryLayer

logger = structlog.get_logger()


class MergeStrategy(Enum):
    """Strategies for merging duplicate entries."""

    KEEP_FIRST = "keep_first"  # Keep the first entry
    KEEP_LATEST = "keep_latest"  # Keep the most recent entry
    KEEP_HIGHEST_IMPORTANCE = "keep_highest_importance"  # Keep highest importance
    MERGE_CONTENT = "merge_content"  # Combine content
    MERGE_ALL = "merge_all"  # Combine content, tags, and metadata


@dataclass
class MergeResult:
    """Result of a merge operation."""

    merged_entry: MemoryEntry
    source_ids: list[UUID]
    strategy_used: MergeStrategy
    merged_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "merged_entry_id": str(self.merged_entry.id),
            "source_ids": [str(id) for id in self.source_ids],
            "strategy_used": self.strategy_used.value,
            "merged_at": self.merged_at.isoformat(),
        }


class MemoryMerger:
    """Merges duplicate memory entries using various strategies.

    Provides:
    - Multiple merge strategies
    - Content combination
    - Metadata merging
    - Importance score handling
    """

    def __init__(
        self,
        default_strategy: MergeStrategy = MergeStrategy.KEEP_LATEST,
    ):
        """Initialize merger.

        Args:
            default_strategy: Default merge strategy
        """
        self._default_strategy = default_strategy

        # Custom merge functions
        self._custom_mergers: dict[str, Callable[[list[MemoryEntry]], MemoryEntry]] = {}

        # Statistics
        self._total_merges = 0
        self._entries_eliminated = 0

    def merge(
        self,
        entries: list[MemoryEntry],
        strategy: MergeStrategy | None = None,
    ) -> MergeResult:
        """Merge duplicate entries.

        Args:
            entries: Entries to merge (duplicates)
            strategy: Merge strategy (uses default if not specified)

        Returns:
            Merge result
        """
        if not entries:
            raise ValueError("No entries to merge")

        if len(entries) == 1:
            return MergeResult(
                merged_entry=entries[0],
                source_ids=[entries[0].id],
                strategy_used=strategy or self._default_strategy,
            )

        strategy = strategy or self._default_strategy
        source_ids = [e.id for e in entries]

        # Apply merge strategy
        if strategy == MergeStrategy.KEEP_FIRST:
            merged = self._keep_first(entries)
        elif strategy == MergeStrategy.KEEP_LATEST:
            merged = self._keep_latest(entries)
        elif strategy == MergeStrategy.KEEP_HIGHEST_IMPORTANCE:
            merged = self._keep_highest_importance(entries)
        elif strategy == MergeStrategy.MERGE_CONTENT:
            merged = self._merge_content(entries)
        elif strategy == MergeStrategy.MERGE_ALL:
            merged = self._merge_all(entries)
        else:
            merged = self._keep_first(entries)

        # Update statistics
        self._total_merges += 1
        self._entries_eliminated += len(entries) - 1

        logger.debug(
            "Entries merged",
            source_count=len(entries),
            strategy=strategy.value,
        )

        return MergeResult(
            merged_entry=merged,
            source_ids=source_ids,
            strategy_used=strategy,
        )

    def _keep_first(self, entries: list[MemoryEntry]) -> MemoryEntry:
        """Keep the first entry (by creation time).

        Args:
            entries: Entries to merge

        Returns:
            First entry
        """
        sorted_entries = sorted(entries, key=lambda e: e.created_at)
        return sorted_entries[0]

    def _keep_latest(self, entries: list[MemoryEntry]) -> MemoryEntry:
        """Keep the latest entry (by update time).

        Args:
            entries: Entries to merge

        Returns:
            Latest entry
        """
        sorted_entries = sorted(entries, key=lambda e: e.updated_at, reverse=True)
        return sorted_entries[0]

    def _keep_highest_importance(self, entries: list[MemoryEntry]) -> MemoryEntry:
        """Keep entry with highest importance score.

        Args:
            entries: Entries to merge

        Returns:
            Highest importance entry
        """
        sorted_entries = sorted(
            entries,
            key=lambda e: e.importance.base_score,
            reverse=True,
        )
        return sorted_entries[0]

    def _merge_content(self, entries: list[MemoryEntry]) -> MemoryEntry:
        """Merge content from all entries.

        Args:
            entries: Entries to merge

        Returns:
            New entry with merged content
        """
        # Sort by creation time
        sorted_entries = sorted(entries, key=lambda e: e.created_at)

        # Combine unique content
        seen_content = set()
        content_parts = []

        for entry in sorted_entries:
            normalized = entry.content.strip()
            if normalized not in seen_content:
                content_parts.append(entry.content)
                seen_content.add(normalized)

        merged_content = "\n\n".join(content_parts)

        # Use base entry and update
        base = sorted_entries[0]

        return MemoryEntry(
            id=uuid4(),
            content=merged_content,
            layer=base.layer,
            importance=self._merge_importance(entries),
            tags=base.tags,
            metadata={
                **base.metadata,
                "merged_from": [str(e.id) for e in entries],
                "merged_at": datetime.now(timezone.utc).isoformat(),
            },
            embedding=base.embedding,
            source_id=base.id,
            created_at=base.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def _merge_all(self, entries: list[MemoryEntry]) -> MemoryEntry:
        """Merge content, tags, and metadata from all entries.

        Args:
            entries: Entries to merge

        Returns:
            New entry with all merged data
        """
        # Sort by creation time
        sorted_entries = sorted(entries, key=lambda e: e.created_at)

        # Combine unique content
        seen_content = set()
        content_parts = []

        for entry in sorted_entries:
            normalized = entry.content.strip()
            if normalized not in seen_content:
                content_parts.append(entry.content)
                seen_content.add(normalized)

        merged_content = "\n\n".join(content_parts)

        # Merge tags (preserve order, remove duplicates)
        all_tags = []
        seen_tags = set()
        for entry in sorted_entries:
            for tag in entry.tags:
                if tag not in seen_tags:
                    all_tags.append(tag)
                    seen_tags.add(tag)

        # Merge metadata
        merged_metadata = {}
        for entry in sorted_entries:
            merged_metadata.update(entry.metadata)

        merged_metadata["merged_from"] = [str(e.id) for e in entries]
        merged_metadata["merged_at"] = datetime.now(timezone.utc).isoformat()

        # Use oldest entry as base
        base = sorted_entries[0]

        return MemoryEntry(
            id=uuid4(),
            content=merged_content,
            layer=self._select_layer(entries),
            importance=self._merge_importance(entries),
            tags=all_tags,
            metadata=merged_metadata,
            embedding=base.embedding,
            source_id=base.id,
            created_at=base.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def _merge_importance(self, entries: list[MemoryEntry]) -> ImportanceScore:
        """Merge importance scores from entries.

        Args:
            entries: Entries to merge

        Returns:
            Merged importance score
        """
        # Use highest base score
        max_score = max(e.importance.base_score for e in entries)

        # Sum access counts
        total_access = sum(e.importance.access_count for e in entries)

        # Use most recent access
        latest_access = max(e.importance.last_accessed for e in entries)

        # Average recency weight
        avg_recency = sum(e.importance.recency_weight for e in entries) / len(entries)

        return ImportanceScore(
            base_score=max_score,
            recency_weight=avg_recency,
            access_count=total_access,
            last_accessed=latest_access,
        )

    def _select_layer(self, entries: list[MemoryEntry]) -> MemoryLayer:
        """Select appropriate layer for merged entry.

        Args:
            entries: Entries being merged

        Returns:
            Selected memory layer
        """
        # Priority: Semantic > Episodic > Working
        layer_priority = {
            MemoryLayer.SEMANTIC: 3,
            MemoryLayer.EPISODIC: 2,
            MemoryLayer.WORKING: 1,
        }

        return max(entries, key=lambda e: layer_priority.get(e.layer, 0)).layer

    def register_custom_merger(
        self,
        name: str,
        merger_func: Callable[[list[MemoryEntry]], MemoryEntry],
    ) -> None:
        """Register a custom merge function.

        Args:
            name: Merger name
            merger_func: Function that takes entries and returns merged entry
        """
        self._custom_mergers[name] = merger_func

    def merge_custom(
        self,
        entries: list[MemoryEntry],
        merger_name: str,
    ) -> MergeResult:
        """Merge using a custom merger.

        Args:
            entries: Entries to merge
            merger_name: Name of registered merger

        Returns:
            Merge result
        """
        if merger_name not in self._custom_mergers:
            raise ValueError(f"Unknown custom merger: {merger_name}")

        merger_func = self._custom_mergers[merger_name]
        merged = merger_func(entries)

        self._total_merges += 1
        self._entries_eliminated += len(entries) - 1

        return MergeResult(
            merged_entry=merged,
            source_ids=[e.id for e in entries],
            strategy_used=MergeStrategy.MERGE_ALL,  # Custom
        )

    def deduplicate_list(
        self,
        entries: list[MemoryEntry],
        groups: list[list[UUID]],
        strategy: MergeStrategy | None = None,
    ) -> list[MemoryEntry]:
        """Deduplicate a list of entries using provided groups.

        Args:
            entries: All entries
            groups: Groups of duplicate entry IDs
            strategy: Merge strategy

        Returns:
            Deduplicated list
        """
        # Build ID to entry mapping
        id_to_entry = {e.id: e for e in entries}

        # Track which entries to remove
        to_remove = set()
        merged_entries = []

        for group_ids in groups:
            group_entries = [id_to_entry[id] for id in group_ids if id in id_to_entry]

            if len(group_entries) <= 1:
                continue

            # Merge the group
            result = self.merge(group_entries, strategy)
            merged_entries.append(result.merged_entry)

            # Mark originals for removal
            to_remove.update(group_ids)

        # Build result list
        result = [e for e in entries if e.id not in to_remove]
        result.extend(merged_entries)

        return result

    def get_stats(self) -> dict:
        """Get merger statistics.

        Returns:
            Statistics dict
        """
        return {
            "total_merges": self._total_merges,
            "entries_eliminated": self._entries_eliminated,
            "default_strategy": self._default_strategy.value,
            "custom_mergers": list(self._custom_mergers.keys()),
        }


def create_merger(
    strategy: MergeStrategy = MergeStrategy.KEEP_LATEST,
) -> MemoryMerger:
    """Create a memory merger with specified strategy.

    Args:
        strategy: Default merge strategy

    Returns:
        Configured MemoryMerger
    """
    return MemoryMerger(default_strategy=strategy)
