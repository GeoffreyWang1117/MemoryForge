"""Working Memory: Current task context with sliding window and importance scoring."""

from collections import deque
from datetime import datetime, timezone
from typing import Deque
from uuid import UUID

from memoryforge.core.base import BaseMemory
from memoryforge.core.types import (
    ImportanceScore,
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
    MemoryResult,
)


class WorkingMemory(BaseMemory):
    """Working memory with sliding window and importance-based retention.

    Maintains current task context in memory, using a sliding window
    to limit size while preserving important entries based on
    importance scores.
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_tokens: int = 8000,
        importance_threshold: float = 0.5,
    ):
        super().__init__(MemoryLayer.WORKING)
        self._max_entries = max_entries
        self._max_tokens = max_tokens
        self._importance_threshold = importance_threshold
        self._entries: Deque[MemoryEntry] = deque(maxlen=max_entries)
        self._pinned: dict[UUID, MemoryEntry] = {}
        self._token_count = 0

    @property
    def entries(self) -> list[MemoryEntry]:
        """Get all current entries."""
        return list(self._pinned.values()) + list(self._entries)

    @property
    def token_count(self) -> int:
        return self._token_count

    async def store(self, entry: MemoryEntry) -> None:
        """Store an entry in working memory."""
        entry.layer = MemoryLayer.WORKING
        entry.updated_at = datetime.now(timezone.utc)

        if entry.importance.effective_score >= self._importance_threshold:
            self._pinned[entry.id] = entry
        else:
            evicted = None
            if len(self._entries) >= self._max_entries:
                evicted = self._entries[0]
            self._entries.append(entry)

            if evicted:
                await self._on_eviction(evicted)

        self._update_token_count()

    async def retrieve(self, query: MemoryQuery) -> MemoryResult:
        """Retrieve entries from working memory."""
        start_time = datetime.now(timezone.utc)
        all_entries = self.entries

        filtered = []
        scores = []

        for entry in all_entries:
            if entry.importance.effective_score < query.min_importance:
                continue

            if query.tags_filter:
                if not any(tag in entry.tags for tag in query.tags_filter):
                    continue

            if query.time_range:
                if not (query.time_range[0] <= entry.created_at <= query.time_range[1]):
                    continue

            filtered.append(entry)
            scores.append(entry.importance.effective_score)

        sorted_pairs = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)
        filtered = [p[0] for p in sorted_pairs[: query.top_k]]
        scores = [p[1] for p in sorted_pairs[: query.top_k]]

        query_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return MemoryResult(
            entries=filtered,
            scores=scores,
            layer_sources=[MemoryLayer.WORKING] * len(filtered),
            query_time_ms=query_time,
            total_candidates=len(all_entries),
        )

    async def update(self, entry: MemoryEntry) -> None:
        """Update an existing entry."""
        entry.updated_at = datetime.now(timezone.utc)

        if entry.id in self._pinned:
            self._pinned[entry.id] = entry
            return

        for i, existing in enumerate(self._entries):
            if existing.id == entry.id:
                self._entries[i] = entry
                return

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        try:
            uuid_id = UUID(entry_id)
        except ValueError:
            return False

        if uuid_id in self._pinned:
            del self._pinned[uuid_id]
            self._update_token_count()
            return True

        for i, entry in enumerate(self._entries):
            if entry.id == uuid_id:
                del self._entries[i]
                self._update_token_count()
                return True

        return False

    async def clear(self) -> None:
        """Clear all working memory."""
        self._entries.clear()
        self._pinned.clear()
        self._token_count = 0

    def pin(self, entry_id: UUID) -> bool:
        """Pin an entry to prevent eviction."""
        for entry in self._entries:
            if entry.id == entry_id:
                self._pinned[entry_id] = entry
                self._entries.remove(entry)
                return True
        return False

    def unpin(self, entry_id: UUID) -> bool:
        """Unpin an entry."""
        if entry_id in self._pinned:
            entry = self._pinned.pop(entry_id)
            self._entries.append(entry)
            return True
        return False

    def _update_token_count(self) -> None:
        """Estimate token count for all entries."""
        total = 0
        for entry in self.entries:
            total += len(entry.content) // 4
        self._token_count = total

    async def _on_eviction(self, entry: MemoryEntry) -> None:
        """Handle entry eviction. Override for custom behavior."""
        pass

    def get_context_window(self, max_tokens: int | None = None) -> list[MemoryEntry]:
        """Get entries that fit within token budget."""
        budget = max_tokens or self._max_tokens
        result = []
        token_count = 0

        for entry in sorted(
            self.entries,
            key=lambda e: e.importance.effective_score,
            reverse=True,
        ):
            entry_tokens = len(entry.content) // 4
            if token_count + entry_tokens > budget:
                break
            result.append(entry)
            token_count += entry_tokens

        return result

    def update_importance(self, entry_id: UUID, score: float) -> bool:
        """Update the importance score of an entry."""
        for entry in self.entries:
            if entry.id == entry_id:
                entry.importance = ImportanceScore(
                    base_score=score,
                    recency_weight=entry.importance.recency_weight,
                    access_count=entry.importance.access_count + 1,
                    last_accessed=datetime.now(timezone.utc),
                )
                return True
        return False
