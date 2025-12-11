"""Semantic Memory: Project knowledge graph with code structure and relationships."""

from datetime import datetime, timezone
from uuid import UUID

import structlog

from memoryforge.core.base import BaseGraphStore, BaseMemory
from memoryforge.core.types import (
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
    MemoryResult,
    SemanticEntity,
    SemanticRelation,
)

logger = structlog.get_logger()


class SemanticMemory(BaseMemory):
    """Semantic memory backed by a knowledge graph.

    Stores project-level knowledge including code structure,
    API relationships, type dependencies, and historical decisions.
    """

    def __init__(self, graph_store: BaseGraphStore):
        super().__init__(MemoryLayer.SEMANTIC)
        self._graph_store = graph_store
        self._entities: dict[UUID, SemanticEntity] = {}
        self._relations: list[SemanticRelation] = []

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry as a graph node."""
        entry.layer = MemoryLayer.SEMANTIC
        entry.updated_at = datetime.now(timezone.utc)

        labels = ["MemoryEntry"]
        if entry.tags:
            labels.extend(entry.tags)

        await self._graph_store.add_node(
            node_id=str(entry.id),
            labels=labels,
            properties={
                "content": entry.content,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "importance": entry.importance.effective_score,
                **entry.metadata,
            },
        )

    async def retrieve(self, query: MemoryQuery) -> MemoryResult:
        """Retrieve memories using graph traversal."""
        start_time = datetime.now(timezone.utc)

        cypher = """
        MATCH (n:MemoryEntry)
        WHERE n.content CONTAINS $query_text
        RETURN n
        ORDER BY n.importance DESC
        LIMIT $top_k
        """

        results = await self._graph_store.query(
            cypher,
            params={"query_text": query.query_text, "top_k": query.top_k},
        )

        entries = []
        scores = []

        for record in results:
            node = record.get("n", {})
            entry = MemoryEntry(
                id=UUID(node.get("id", str(UUID()))),
                content=node.get("content", ""),
                layer=MemoryLayer.SEMANTIC,
                metadata=node,
            )
            entries.append(entry)
            scores.append(node.get("importance", 0.5))

        query_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return MemoryResult(
            entries=entries,
            scores=scores,
            layer_sources=[MemoryLayer.SEMANTIC] * len(entries),
            query_time_ms=query_time,
            total_candidates=len(results),
        )

    async def update(self, entry: MemoryEntry) -> None:
        """Update an existing memory entry."""
        await self.store(entry)

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry from the graph."""
        try:
            await self._graph_store.query(
                "MATCH (n {id: $id}) DETACH DELETE n",
                params={"id": entry_id},
            )
            return True
        except Exception as e:
            logger.error("Failed to delete entry", entry_id=entry_id, error=str(e))
            return False

    async def clear(self) -> None:
        """Clear all semantic memory."""
        await self._graph_store.query("MATCH (n) DETACH DELETE n")
        self._entities.clear()
        self._relations.clear()

    async def add_entity(self, entity: SemanticEntity) -> None:
        """Add a semantic entity to the knowledge graph."""
        self._entities[entity.id] = entity

        labels = ["Entity", entity.entity_type]

        await self._graph_store.add_node(
            node_id=str(entity.id),
            labels=labels,
            properties={
                "name": entity.name,
                "entity_type": entity.entity_type,
                "source_file": entity.source_file,
                "line_number": entity.line_number,
                "created_at": entity.created_at.isoformat(),
                **entity.properties,
            },
        )

    async def add_relation(self, relation: SemanticRelation) -> None:
        """Add a relation between entities."""
        self._relations.append(relation)

        await self._graph_store.add_edge(
            source_id=str(relation.source_id),
            target_id=str(relation.target_id),
            relation_type=relation.relation_type,
            properties={
                "weight": relation.weight,
                "created_at": relation.created_at.isoformat(),
                **relation.properties,
            },
        )

    async def get_entity(self, entity_id: UUID) -> SemanticEntity | None:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    async def find_entities_by_type(self, entity_type: str) -> list[SemanticEntity]:
        """Find all entities of a given type."""
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    async def get_related_entities(
        self,
        entity_id: UUID,
        relation_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[SemanticEntity]:
        """Get entities related to a given entity."""
        neighbors = await self._graph_store.get_neighbors(
            node_id=str(entity_id),
            relation_types=relation_types,
            depth=depth,
        )

        entities = []
        for neighbor in neighbors:
            entity_id_str = neighbor.get("id")
            if entity_id_str:
                entity = self._entities.get(UUID(entity_id_str))
                if entity:
                    entities.append(entity)

        return entities

    async def find_path(
        self,
        source_id: UUID,
        target_id: UUID,
        max_depth: int = 5,
    ) -> list[SemanticEntity]:
        """Find shortest path between two entities."""
        cypher = """
        MATCH path = shortestPath(
            (source {id: $source_id})-[*1..$max_depth]-(target {id: $target_id})
        )
        RETURN nodes(path) as nodes
        """

        results = await self._graph_store.query(
            cypher,
            params={
                "source_id": str(source_id),
                "target_id": str(target_id),
                "max_depth": max_depth,
            },
        )

        if not results:
            return []

        path_nodes = results[0].get("nodes", [])
        entities = []

        for node in path_nodes:
            entity_id = node.get("id")
            if entity_id:
                entity = self._entities.get(UUID(entity_id))
                if entity:
                    entities.append(entity)

        return entities

    async def get_subgraph(
        self,
        center_id: UUID,
        radius: int = 2,
    ) -> tuple[list[SemanticEntity], list[SemanticRelation]]:
        """Get a subgraph centered on an entity."""
        neighbors = await self._graph_store.get_neighbors(
            node_id=str(center_id),
            depth=radius,
        )

        entity_ids = {center_id}
        for neighbor in neighbors:
            entity_id_str = neighbor.get("id")
            if entity_id_str:
                entity_ids.add(UUID(entity_id_str))

        entities = [self._entities[eid] for eid in entity_ids if eid in self._entities]

        relations = [
            r
            for r in self._relations
            if r.source_id in entity_ids and r.target_id in entity_ids
        ]

        return entities, relations
