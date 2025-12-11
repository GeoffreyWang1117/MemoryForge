"""Episodic Memory: Conversation history with LLM summaries and vector retrieval."""

from datetime import datetime, timezone
from uuid import UUID

import structlog

from memoryforge.core.base import BaseEmbedder, BaseMemory, BaseSummarizer, BaseVectorStore
from memoryforge.core.types import (
    ConversationTurn,
    EpisodicSummary,
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
    MemoryResult,
)

logger = structlog.get_logger()


class EpisodicMemory(BaseMemory):
    """Episodic memory with LLM-generated summaries and semantic retrieval.

    Stores conversation history as compressed summaries, supporting
    time-based decay and semantic search via vector embeddings.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        summarizer: BaseSummarizer,
        summary_threshold: int = 10,
        decay_rate: float = 0.95,
    ):
        super().__init__(MemoryLayer.EPISODIC)
        self._vector_store = vector_store
        self._embedder = embedder
        self._summarizer = summarizer
        self._summary_threshold = summary_threshold
        self._decay_rate = decay_rate
        self._pending_turns: list[ConversationTurn] = []
        self._summaries: dict[UUID, EpisodicSummary] = {}

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry with embedding."""
        entry.layer = MemoryLayer.EPISODIC
        entry.updated_at = datetime.now(timezone.utc)

        if entry.embedding is None:
            entry.embedding = await self._embedder.embed(entry.content)

        await self._vector_store.upsert(
            ids=[str(entry.id)],
            embeddings=[entry.embedding],
            payloads=[
                {
                    "content": entry.content,
                    "created_at": entry.created_at.isoformat(),
                    "importance": entry.importance.effective_score,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                }
            ],
        )

    async def retrieve(self, query: MemoryQuery) -> MemoryResult:
        """Retrieve memories using semantic search."""
        start_time = datetime.now(timezone.utc)

        query_embedding = await self._embedder.embed(query.query_text)

        filter_conditions = {}
        if query.min_importance > 0:
            filter_conditions["importance"] = {"$gte": query.min_importance}
        if query.tags_filter:
            filter_conditions["tags"] = {"$in": query.tags_filter}

        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=query.top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

        entries = []
        scores = []

        for entry_id, score, payload in results:
            entry = MemoryEntry(
                id=UUID(entry_id),
                content=payload.get("content", ""),
                layer=MemoryLayer.EPISODIC,
                tags=payload.get("tags", []),
                metadata=payload.get("metadata", {}),
                embedding=None if not query.include_embeddings else None,
            )
            entries.append(entry)
            scores.append(score)

        query_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return MemoryResult(
            entries=entries,
            scores=scores,
            layer_sources=[MemoryLayer.EPISODIC] * len(entries),
            query_time_ms=query_time,
            total_candidates=len(results),
        )

    async def update(self, entry: MemoryEntry) -> None:
        """Update an existing memory entry."""
        entry.updated_at = datetime.now(timezone.utc)
        if entry.embedding is None:
            entry.embedding = await self._embedder.embed(entry.content)
        await self.store(entry)

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            await self._vector_store.delete([entry_id])
            return True
        except Exception as e:
            logger.error("Failed to delete entry", entry_id=entry_id, error=str(e))
            return False

    async def clear(self) -> None:
        """Clear all episodic memory."""
        self._pending_turns.clear()
        self._summaries.clear()

    async def add_turn(self, turn: ConversationTurn) -> EpisodicSummary | None:
        """Add a conversation turn, potentially triggering summarization."""
        self._pending_turns.append(turn)

        if len(self._pending_turns) >= self._summary_threshold:
            return await self._create_summary()

        return None

    async def _create_summary(self) -> EpisodicSummary:
        """Create a summary from pending turns."""
        if not self._pending_turns:
            raise ValueError("No pending turns to summarize")

        content = "\n".join(
            f"{turn.role}: {turn.content}" for turn in self._pending_turns
        )

        summary_text = await self._summarizer.summarize(content)
        key_facts = await self._summarizer.extract_key_facts(content)

        time_range = (
            self._pending_turns[0].timestamp,
            self._pending_turns[-1].timestamp,
        )

        summary = EpisodicSummary(
            summary=summary_text,
            key_facts=key_facts,
            time_range=time_range,
            compression_ratio=len(summary_text) / len(content),
        )

        self._summaries[summary.id] = summary

        entry = MemoryEntry(
            id=summary.id,
            content=summary_text,
            layer=MemoryLayer.EPISODIC,
            tags=["summary"],
            metadata={
                "key_facts": key_facts,
                "time_range": [t.isoformat() for t in time_range],
                "compression_ratio": summary.compression_ratio,
            },
        )
        await self.store(entry)

        self._pending_turns.clear()

        logger.info(
            "Created episodic summary",
            summary_id=str(summary.id),
            compression_ratio=summary.compression_ratio,
        )

        return summary

    async def get_recent_context(self, max_summaries: int = 5) -> list[EpisodicSummary]:
        """Get recent summaries for context."""
        sorted_summaries = sorted(
            self._summaries.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )
        return sorted_summaries[:max_summaries]

    async def search_by_time(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[EpisodicSummary]:
        """Search summaries within a time range."""
        return [
            s
            for s in self._summaries.values()
            if s.time_range[0] >= start_time and s.time_range[1] <= end_time
        ]
