"""Embedding providers for vector similarity search."""

import structlog
from sentence_transformers import SentenceTransformer

from memoryforge.core.base import BaseEmbedder

logger = structlog.get_logger()


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding provider using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Loaded embedding model",
            model=model_name,
            dimension=self._dimension,
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class OpenAIEmbedder(BaseEmbedder):
    """Embedding provider using OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        logger.info("Initialized OpenAI embedder", model=model)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimensions.get(self._model, 1536)
