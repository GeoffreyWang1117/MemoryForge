"""Batch operations for memory management."""

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import structlog

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer, MemoryQuery
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.scoring.importance import RuleBasedScorer, ScoringContext

logger = structlog.get_logger()


@dataclass
class BatchStoreResult:
    """Result of a batch store operation."""

    total: int
    stored: int
    failed: int
    errors: list[str]


@dataclass
class BatchQueryResult:
    """Result of a batch query operation."""

    queries: int
    total_results: int
    avg_query_time_ms: float


class BatchMemoryOperations:
    """Batch operations for efficient memory management."""

    def __init__(self, memory: WorkingMemory, batch_size: int = 50):
        self._memory = memory
        self._batch_size = batch_size
        self._scorer = RuleBasedScorer()

    async def store_batch(
        self,
        contents: list[str],
        auto_score: bool = True,
        default_importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> BatchStoreResult:
        """Store multiple memory entries at once.

        Args:
            contents: List of content strings to store
            auto_score: Whether to automatically score importance
            default_importance: Default importance if not scoring
            tags: Tags to apply to all entries

        Returns:
            BatchStoreResult with statistics
        """
        stored = 0
        failed = 0
        errors = []

        for i, content in enumerate(contents):
            try:
                # Score importance
                if auto_score:
                    ctx = ScoringContext(content=content)
                    result = await self._scorer.score(ctx)
                    importance = result.score
                else:
                    importance = default_importance

                # Create entry
                entry = MemoryEntry(
                    content=content,
                    layer=MemoryLayer.WORKING,
                    importance=ImportanceScore(base_score=importance),
                    tags=tags or [],
                    metadata={"batch_index": i},
                )

                await self._memory.store(entry)
                stored += 1

            except Exception as e:
                failed += 1
                errors.append(f"Item {i}: {str(e)}")
                logger.error("Batch store failed", index=i, error=str(e))

        logger.info(
            "Batch store completed",
            total=len(contents),
            stored=stored,
            failed=failed,
        )

        return BatchStoreResult(
            total=len(contents),
            stored=stored,
            failed=failed,
            errors=errors,
        )

    async def store_stream(
        self,
        content_stream: AsyncIterator[str],
        auto_score: bool = True,
    ) -> BatchStoreResult:
        """Store memories from an async stream.

        Args:
            content_stream: Async iterator of content strings
            auto_score: Whether to automatically score importance

        Returns:
            BatchStoreResult with statistics
        """
        batch = []
        total_stored = 0
        total_failed = 0
        all_errors = []

        async for content in content_stream:
            batch.append(content)

            if len(batch) >= self._batch_size:
                result = await self.store_batch(batch, auto_score=auto_score)
                total_stored += result.stored
                total_failed += result.failed
                all_errors.extend(result.errors)
                batch = []

        # Process remaining items
        if batch:
            result = await self.store_batch(batch, auto_score=auto_score)
            total_stored += result.stored
            total_failed += result.failed
            all_errors.extend(result.errors)

        return BatchStoreResult(
            total=total_stored + total_failed,
            stored=total_stored,
            failed=total_failed,
            errors=all_errors,
        )

    async def query_batch(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> tuple[list[list[MemoryEntry]], BatchQueryResult]:
        """Execute multiple queries at once.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            Tuple of (results per query, statistics)
        """
        all_results = []
        total_time = 0
        total_results = 0

        for query_text in queries:
            query = MemoryQuery(
                query_text=query_text,
                target_layers=[MemoryLayer.WORKING],
                top_k=top_k,
            )

            result = await self._memory.retrieve(query)
            all_results.append(result.entries)
            total_time += result.query_time_ms
            total_results += len(result.entries)

        stats = BatchQueryResult(
            queries=len(queries),
            total_results=total_results,
            avg_query_time_ms=total_time / len(queries) if queries else 0,
        )

        return all_results, stats

    async def deduplicate(self, similarity_threshold: float = 0.9) -> int:
        """Remove duplicate or highly similar memories.

        Args:
            similarity_threshold: Content similarity threshold for deduplication

        Returns:
            Number of entries removed
        """
        entries = self._memory.entries
        to_remove = set()

        for i, entry1 in enumerate(entries):
            if entry1.id in to_remove:
                continue

            for entry2 in entries[i + 1 :]:
                if entry2.id in to_remove:
                    continue

                # Simple content similarity
                similarity = self._content_similarity(entry1.content, entry2.content)

                if similarity >= similarity_threshold:
                    # Remove the one with lower importance
                    if entry1.importance.effective_score >= entry2.importance.effective_score:
                        to_remove.add(entry2.id)
                    else:
                        to_remove.add(entry1.id)
                        break

        # Remove duplicates
        removed = 0
        for entry_id in to_remove:
            if await self._memory.delete(str(entry_id)):
                removed += 1

        logger.info("Deduplication completed", removed=removed)
        return removed

    def _content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple content similarity using Jaccard index."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def prune_low_importance(self, threshold: float = 0.3) -> int:
        """Remove entries below importance threshold.

        Args:
            threshold: Minimum importance score to keep

        Returns:
            Number of entries removed
        """
        entries = self._memory.entries
        removed = 0

        for entry in entries:
            if entry.importance.effective_score < threshold:
                if await self._memory.delete(str(entry.id)):
                    removed += 1

        logger.info("Pruning completed", threshold=threshold, removed=removed)
        return removed

    async def export_to_list(self) -> list[dict]:
        """Export all memories to a list of dictionaries."""
        return [
            {
                "id": str(e.id),
                "content": e.content,
                "importance": e.importance.effective_score,
                "tags": e.tags,
                "created_at": e.created_at.isoformat(),
            }
            for e in self._memory.entries
        ]

    async def import_from_list(
        self,
        data: list[dict],
        overwrite: bool = False,
    ) -> BatchStoreResult:
        """Import memories from a list of dictionaries.

        Args:
            data: List of memory dictionaries with 'content' key
            overwrite: Whether to clear existing memories first

        Returns:
            BatchStoreResult with statistics
        """
        if overwrite:
            await self._memory.clear()

        contents = [item.get("content", "") for item in data if item.get("content")]
        return await self.store_batch(contents, auto_score=True)

    def get_stats(self) -> dict:
        """Get batch operation statistics."""
        entries = self._memory.entries

        if not entries:
            return {
                "total_entries": 0,
                "avg_importance": 0,
                "pinned_count": len(self._memory._pinned),
            }

        importances = [e.importance.effective_score for e in entries]

        return {
            "total_entries": len(entries),
            "avg_importance": sum(importances) / len(importances),
            "min_importance": min(importances),
            "max_importance": max(importances),
            "pinned_count": len(self._memory._pinned),
            "token_count": self._memory.token_count,
        }
