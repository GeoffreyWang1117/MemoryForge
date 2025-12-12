"""Configuration management for MemoryForge."""

from functools import lru_cache

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

    # Working memory
    working_max_entries: int = 100
    working_max_tokens: int = 8000
    working_importance_threshold: float = 0.5

    # Episodic memory
    episodic_summary_threshold: int = 10
    episodic_decay_rate: float = 0.95

    # Embedding
    embedding_dimension: int = 384


class CompressionSettings(BaseSettings):
    """Memory compression settings."""

    model_config = SettingsConfigDict(env_prefix="COMPRESSION_")

    # Age-based compression
    min_age_hours: float = 24.0
    importance_threshold: float = 0.9
    target_ratio: float = 0.3

    # Strategy ratios
    moderate_keep_ratio: float = 0.7
    aggressive_keep_ratio: float = 0.5

    # Content limits
    truncate_short: int = 100
    truncate_long: int = 200
    preview_length: int = 100
    summary_preview_length: int = 200

    # Time window
    window_hours: int = 24

    # Deduplication
    similarity_threshold: float = 0.95


class RetrievalSettings(BaseSettings):
    """Retrieval and search settings."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    # Semantic search
    min_similarity: float = 0.5
    semantic_weight: float = 0.7
    recency_boost: float = 0.1
    importance_boost: float = 0.15
    recency_decay_hours: int = 168  # 1 week


class ScoringSettings(BaseSettings):
    """Importance scoring settings."""

    model_config = SettingsConfigDict(env_prefix="SCORING_")

    base_score: float = 0.5
    keyword_max_boost: float = 0.3
    pattern_max_boost: float = 0.15
    confidence_base: float = 0.5
    confidence_multiplier: float = 0.05
    llm_threshold: float = 0.5


class SummarizationSettings(BaseSettings):
    """Summarization settings."""

    model_config = SettingsConfigDict(env_prefix="SUMMARIZATION_")

    brief_max_tokens: int = 100
    standard_max_tokens: int = 200
    detailed_max_tokens: int = 500
    comprehensive_max_tokens: int = 1000
    min_content_length: int = 100


class ContextSettings(BaseSettings):
    """Context builder settings."""

    model_config = SettingsConfigDict(env_prefix="CONTEXT_")

    memories_budget: int = 3000
    history_budget: int = 2000
    user_budget: int = 1000
    custom_budget: int = 1000
    min_content_threshold: int = 100


class APISettings(BaseSettings):
    """API server settings."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8000
    database_path: str = "memoryforge.db"
    default_model: str = "gpt-4o-mini"

    # Rate limiting
    requests_per_hour: int = 1000
    rate_window_seconds: int = 60

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class AuthSettings(BaseSettings):
    """Authentication settings."""

    model_config = SettingsConfigDict(env_prefix="AUTH_")

    token_expiry_hours: int = 24
    token_refresh_hours: int = 168  # 7 days
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 100
    rate_limit_per_day: int = 1000


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MemoryForge"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Sub-settings
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    summarization: SummarizationSettings = Field(default_factory=SummarizationSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    api: APISettings = Field(default_factory=APISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)


@lru_cache
def get_settings() -> Settings:
    """Get application settings singleton (cached)."""
    return Settings()
