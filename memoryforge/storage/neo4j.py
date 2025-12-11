"""Neo4j graph store implementation."""

from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from memoryforge.core.base import BaseGraphStore

logger = structlog.get_logger()


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j-backed graph store for semantic memory."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        self._uri = uri
        self._database = database
        self._driver: AsyncDriver | None = None
        self._username = username
        self._password = password

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
        )
        logger.info("Connected to Neo4j", uri=self._uri)

    async def close(self) -> None:
        """Close the connection."""
        if self._driver:
            await self._driver.close()
            logger.info("Closed Neo4j connection")

    async def add_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any],
    ) -> None:
        """Add a node to the graph."""
        if not self._driver:
            await self.connect()

        labels_str = ":".join(labels)
        properties["id"] = node_id

        cypher = f"""
        MERGE (n:{labels_str} {{id: $id}})
        SET n += $properties
        """

        async with self._driver.session(database=self._database) as session:
            await session.run(cypher, id=node_id, properties=properties)

        logger.debug("Added node", node_id=node_id, labels=labels)

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add an edge between nodes."""
        if not self._driver:
            await self.connect()

        props = properties or {}
        cypher = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        MERGE (a)-[r:{relation_type}]->(b)
        SET r += $properties
        """

        async with self._driver.session(database=self._database) as session:
            await session.run(
                cypher,
                source_id=source_id,
                target_id=target_id,
                properties=props,
            )

        logger.debug(
            "Added edge",
            source=source_id,
            target=target_id,
            relation=relation_type,
        )

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query."""
        if not self._driver:
            await self.connect()

        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, params or {})
            records = await result.data()

        return records

    async def get_neighbors(
        self,
        node_id: str,
        relation_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes."""
        if not self._driver:
            await self.connect()

        if relation_types:
            rel_pattern = "|".join(relation_types)
            rel_clause = f"[:{rel_pattern}*1..{depth}]"
        else:
            rel_clause = f"[*1..{depth}]"

        cypher = f"""
        MATCH (start {{id: $node_id}})-{rel_clause}-(neighbor)
        RETURN DISTINCT neighbor
        """

        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, node_id=node_id)
            records = await result.data()

        return [dict(r["neighbor"]) for r in records]

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its relationships."""
        if not self._driver:
            await self.connect()

        cypher = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        """

        async with self._driver.session(database=self._database) as session:
            await session.run(cypher, node_id=node_id)

        logger.debug("Deleted node", node_id=node_id)

    async def clear(self) -> None:
        """Clear all nodes and relationships."""
        if not self._driver:
            await self.connect()

        async with self._driver.session(database=self._database) as session:
            await session.run("MATCH (n) DETACH DELETE n")

        logger.info("Cleared graph database")
