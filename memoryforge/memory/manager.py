"""Unified Memory Manager: Routes queries to appropriate memory layers."""

from datetime import datetime, timezone

import structlog

from memoryforge.core.types import MemoryEntry, MemoryLayer, MemoryQuery, MemoryResult
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.memory.episodic.memory import EpisodicMemory
from memoryforge.memory.semantic.memory import SemanticMemory

logger = structlog.get_logger()


class MemoryManager:
    """Unified interface for the three-layer memory system.

    Routes queries to appropriate memory layers based on query type
    and merges results for optimal retrieval.
    """

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
    ):
        self._working = working
        self._episodic = episodic
        self._semantic = semantic
        self._layers = {
            MemoryLayer.WORKING: working,
            MemoryLayer.EPISODIC: episodic,
            MemoryLayer.SEMANTIC: semantic,
        }

    @property
    def working(self) -> WorkingMemory:
        return self._working

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    @property
    def semantic(self) -> SemanticMemory:
        return self._semantic

    async def store(self, entry: MemoryEntry, layer: MemoryLayer | None = None) -> None:
        """Store an entry in the specified layer or auto-route."""
        target_layer = layer or entry.layer
        await self._layers[target_layer].store(entry)

        logger.debug(
            "Stored memory entry",
            entry_id=str(entry.id),
            layer=target_layer.value,
        )

    async def retrieve(self, query: MemoryQuery) -> MemoryResult:
        """Retrieve memories from target layers and merge results."""
        start_time = datetime.now(timezone.utc)

        all_entries: list[MemoryEntry] = []
        all_scores: list[float] = []
        all_layers: list[MemoryLayer] = []
        total_candidates = 0

        for layer in query.target_layers:
            if layer not in self._layers:
                continue

            result = await self._layers[layer].retrieve(query)
            all_entries.extend(result.entries)
            all_scores.extend(result.scores)
            all_layers.extend(result.layer_sources)
            total_candidates += result.total_candidates

        sorted_results = sorted(
            zip(all_entries, all_scores, all_layers),
            key=lambda x: x[1],
            reverse=True,
        )

        top_results = sorted_results[: query.top_k]

        query_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return MemoryResult(
            entries=[r[0] for r in top_results],
            scores=[r[1] for r in top_results],
            layer_sources=[r[2] for r in top_results],
            query_time_ms=query_time,
            total_candidates=total_candidates,
        )

    async def promote(self, entry_id: str, from_layer: MemoryLayer, to_layer: MemoryLayer) -> bool:
        """Promote an entry from one layer to another."""
        source_query = MemoryQuery(
            query_text="",
            target_layers=[from_layer],
            top_k=1000,
        )
        result = await self._layers[from_layer].retrieve(source_query)

        for entry in result.entries:
            if str(entry.id) == entry_id:
                entry.layer = to_layer
                await self._layers[to_layer].store(entry)
                logger.info(
                    "Promoted memory entry",
                    entry_id=entry_id,
                    from_layer=from_layer.value,
                    to_layer=to_layer.value,
                )
                return True

        return False

    async def consolidate(self) -> int:
        """Consolidate working memory into episodic memory.

        Returns the number of entries consolidated.
        """
        working_entries = self._working.entries
        consolidated = 0

        for entry in working_entries:
            if entry.importance.effective_score >= 0.7:
                entry.layer = MemoryLayer.EPISODIC
                await self._episodic.store(entry)
                consolidated += 1

        logger.info("Consolidated memories", count=consolidated)
        return consolidated

    async def get_context(self, max_tokens: int = 4000) -> str:
        """Get formatted context from all memory layers for LLM prompt."""
        sections = []

        working_entries = self._working.get_context_window(max_tokens // 2)
        if working_entries:
            working_text = "\n".join(f"- {e.content}" for e in working_entries)
            sections.append(f"## Current Context\n{working_text}")

        recent_summaries = await self._episodic.get_recent_context(3)
        if recent_summaries:
            episodic_text = "\n".join(f"- {s.summary}" for s in recent_summaries)
            sections.append(f"## Recent History\n{episodic_text}")

        return "\n\n".join(sections)

    async def clear_all(self) -> None:
        """Clear all memory layers."""
        await self._working.clear()
        await self._episodic.clear()
        await self._semantic.clear()
        logger.info("Cleared all memory layers")
