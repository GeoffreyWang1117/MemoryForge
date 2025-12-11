"""Configuration management for MemoryForge."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    """Qdrant vector database settings."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = "localhost"
    port: int = 6333
    grpc_port: int | None = None
    api_key: str | None = None
    url: str | None = None
    collection_name: str = "memoryforge_episodic"


class Neo4jSettings(BaseSettings):
    """Neo4j graph database settings."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_")

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: str = Field(default="openai", description="LLM provider: openai or anthropic")
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    summarization_model: str | None = None


class MemorySettings(BaseSettings):
    """Memory layer settings."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    working_max_entries: int = 100
    working_max_tokens: int = 8000
    working_importance_threshold: float = 0.5

    episodic_summary_threshold: int = 10
    episodic_decay_rate: float = 0.95

    embedding_dimension: int = 384


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MemoryForge"
    debug: bool = False
    log_level: str = "INFO"

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
