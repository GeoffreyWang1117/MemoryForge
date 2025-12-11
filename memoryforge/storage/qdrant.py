"""Qdrant vector store implementation."""

from typing import Any

import structlog
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from memoryforge.core.base import BaseVectorStore

logger = structlog.get_logger()


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed vector store for episodic memory."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "memoryforge_episodic",
        embedding_dim: int = 384,
        grpc_port: int | None = None,
        api_key: str | None = None,
        url: str | None = None,
    ):
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                api_key=api_key,
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        try:
            self._client.get_collection(self._collection_name)
            logger.info("Using existing collection", collection=self._collection_name)
        except (UnexpectedResponse, Exception):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Created collection", collection=self._collection_name)

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        payloads: list[dict] | None = None,
    ) -> None:
        """Insert or update vectors."""
        points = []
        for i, (id_, embedding) in enumerate(zip(ids, embeddings)):
            payload = payloads[i] if payloads else {}
            points.append(
                models.PointStruct(
                    id=id_,
                    vector=embedding,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

        logger.debug("Upserted vectors", count=len(points))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_conditions: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors."""
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            (str(hit.id), hit.score, hit.payload or {})
            for hit in results
        ]

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by IDs."""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=ids),
        )
        logger.debug("Deleted vectors", count=len(ids))

    def _build_filter(self, conditions: dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from conditions."""
        must_conditions = []

        for key, value in conditions.items():
            if isinstance(value, dict):
                if "$gte" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(gte=value["$gte"]),
                        )
                    )
                if "$lte" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(lte=value["$lte"]),
                        )
                    )
                if "$in" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value["$in"]),
                        )
                    )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        return models.Filter(must=must_conditions) if must_conditions else None

    async def count(self) -> int:
        """Get total number of vectors in collection."""
        info = self._client.get_collection(self._collection_name)
        return info.points_count

    async def clear(self) -> None:
        """Clear all vectors from collection."""
        self._client.delete_collection(self._collection_name)
        self._ensure_collection()
        logger.info("Cleared collection", collection=self._collection_name)
