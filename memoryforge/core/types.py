"""Core type definitions for the memory system."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class MemoryLayer(str, Enum):
    """Memory layer types in the hierarchical system."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class ImportanceScore(BaseModel):
    """Importance score for memory entries with decay."""

    base_score: float = Field(ge=0.0, le=1.0, description="Base importance score")
    recency_weight: float = Field(default=1.0, ge=0.0, description="Time-based decay factor")
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    last_accessed: datetime = Field(default_factory=utcnow)

    @property
    def effective_score(self) -> float:
        """Calculate effective importance with decay."""
        now = utcnow()
        # Handle timezone-naive datetimes
        last = self.last_accessed
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        hours_since_access = (now - last).total_seconds() / 3600
        decay = self.recency_weight * (0.95 ** hours_since_access)
        access_bonus = min(0.2, self.access_count * 0.02)
        return min(1.0, self.base_score * decay + access_bonus)


class MemoryEntry(BaseModel):
    """A single memory entry in the system."""

    model_config = ConfigDict(
        ser_json_timedelta="iso8601",
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        },
    )

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(description="The actual memory content")
    layer: MemoryLayer = Field(description="Which memory layer this belongs to")
    importance: ImportanceScore = Field(default_factory=ImportanceScore)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    source_id: UUID | None = Field(default=None, description="Reference to source entry")
    tags: list[str] = Field(default_factory=list)


class MemoryQuery(BaseModel):
    """Query for retrieving memories."""

    query_text: str = Field(description="Natural language query")
    target_layers: list[MemoryLayer] = Field(
        default_factory=lambda: list(MemoryLayer),
        description="Which layers to search",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    time_range: tuple[datetime, datetime] | None = Field(default=None)
    tags_filter: list[str] | None = Field(default=None)
    include_embeddings: bool = Field(default=False)


class MemoryResult(BaseModel):
    """Result from a memory query."""

    entries: list[MemoryEntry] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    layer_sources: list[MemoryLayer] = Field(default_factory=list)
    query_time_ms: float = Field(default=0.0)
    total_candidates: int = Field(default=0)


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: str = Field(description="Role: user, assistant, or system")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int | None = Field(default=None)


class EpisodicSummary(BaseModel):
    """Compressed summary of conversation history."""

    id: UUID = Field(default_factory=uuid4)
    summary: str = Field(description="LLM-generated summary")
    key_facts: list[str] = Field(default_factory=list)
    decisions_made: list[str] = Field(default_factory=list)
    time_range: tuple[datetime, datetime]
    source_turn_ids: list[UUID] = Field(default_factory=list)
    compression_ratio: float = Field(default=1.0)
    created_at: datetime = Field(default_factory=utcnow)


class SemanticEntity(BaseModel):
    """Entity in the semantic knowledge graph."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Type: function, class, api, decision, etc.")
    properties: dict[str, Any] = Field(default_factory=dict)
    source_file: str | None = Field(default=None)
    line_number: int | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class SemanticRelation(BaseModel):
    """Relation between entities in the knowledge graph."""

    source_id: UUID
    target_id: UUID
    relation_type: str = Field(description="Type: calls, imports, extends, depends_on, etc.")
    properties: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, ge=0.0)
    created_at: datetime = Field(default_factory=utcnow)
