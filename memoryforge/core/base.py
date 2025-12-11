"""Base classes and interfaces for memory components."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from memoryforge.core.types import (
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
    MemoryResult,
)


class BaseMemory(ABC):
    """Abstract base class for memory layers."""

    def __init__(self, layer: MemoryLayer):
        self._layer = layer

    @property
    def layer(self) -> MemoryLayer:
        return self._layer

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        pass

    @abstractmethod
    async def retrieve(self, query: MemoryQuery) -> MemoryResult:
        """Retrieve memories matching the query."""
        pass

    @abstractmethod
    async def update(self, entry: MemoryEntry) -> None:
        """Update an existing memory entry."""
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories in this layer."""
        pass

    async def stream(self, query: MemoryQuery) -> AsyncIterator[MemoryEntry]:
        """Stream memory entries matching query. Default implementation."""
        result = await self.retrieve(query)
        for entry in result.entries:
            yield entry


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class BaseSummarizer(ABC):
    """Abstract base class for summarization."""

    @abstractmethod
    async def summarize(self, content: str, max_tokens: int | None = None) -> str:
        """Generate a summary of the content."""
        pass

    @abstractmethod
    async def extract_key_facts(self, content: str) -> list[str]:
        """Extract key facts from content."""
        pass


class BaseGraphStore(ABC):
    """Abstract base class for graph storage."""

    @abstractmethod
    async def add_node(self, node_id: str, labels: list[str], properties: dict) -> None:
        """Add a node to the graph."""
        pass

    @abstractmethod
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict | None = None,
    ) -> None:
        """Add an edge between nodes."""
        pass

    @abstractmethod
    async def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query."""
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        relation_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]:
        """Get neighboring nodes."""
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector storage."""

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        payloads: list[dict] | None = None,
    ) -> None:
        """Insert or update vectors."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_conditions: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by IDs."""
        pass
