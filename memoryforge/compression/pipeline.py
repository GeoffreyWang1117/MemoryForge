"""Memory compression pipeline for automatic memory consolidation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable
from uuid import UUID, uuid4

import structlog

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer

logger = structlog.get_logger()


class CompressionLevel(Enum):
    """Compression aggressiveness levels."""

    LIGHT = "light"  # Preserve most detail
    MODERATE = "moderate"  # Balance between detail and compression
    AGGRESSIVE = "aggressive"  # Maximum compression


@dataclass
class CompressionConfig:
    """Configuration for compression pipeline."""

    # Age threshold for compression eligibility
    min_age_hours: float = 24.0

    # Importance threshold - memories above this are never compressed
    importance_threshold: float = 0.9

    # Maximum memories per compression batch
    batch_size: int = 20

    # Compression level
    level: CompressionLevel = CompressionLevel.MODERATE

    # Preserve tagged memories
    preserve_tags: list[str] = field(default_factory=lambda: ["important", "pinned"])

    # Target compression ratio (1.0 = no compression)
    target_ratio: float = 0.3


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    id: UUID = field(default_factory=uuid4)
    original_count: int = 0
    compressed_count: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    preserved_count: int = 0
    compressed_entries: list[MemoryEntry] = field(default_factory=list)
    removed_ids: list[UUID] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def compression_ratio(self) -> float:
        """Calculate the compression ratio achieved."""
        if self.original_count == 0:
            return 1.0
        return self.compressed_count / self.original_count

    @property
    def token_savings(self) -> int:
        """Calculate token savings."""
        return self.original_tokens - self.compressed_tokens

    @property
    def token_reduction_pct(self) -> float:
        """Calculate token reduction percentage."""
        if self.original_tokens == 0:
            return 0.0
        return (self.token_savings / self.original_tokens) * 100


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""

    @abstractmethod
    async def compress(
        self,
        entries: list[MemoryEntry],
        config: CompressionConfig,
    ) -> list[MemoryEntry]:
        """Compress a list of memory entries.

        Args:
            entries: Memory entries to compress
            config: Compression configuration

        Returns:
            Compressed memory entries
        """
        pass

    @abstractmethod
    def can_compress(self, entries: list[MemoryEntry]) -> bool:
        """Check if this strategy can compress the given entries."""
        pass


class CompressionPipeline:
    """Pipeline for compressing and consolidating memories.

    The pipeline supports multiple compression strategies that can be
    applied in sequence or selected based on memory characteristics.
    """

    def __init__(
        self,
        config: CompressionConfig | None = None,
        strategies: list[CompressionStrategy] | None = None,
    ):
        self._config = config or CompressionConfig()
        self._strategies: list[CompressionStrategy] = strategies or []
        self._compression_history: list[CompressionResult] = []

    def add_strategy(self, strategy: CompressionStrategy) -> None:
        """Add a compression strategy to the pipeline."""
        self._strategies.append(strategy)

    def set_config(self, config: CompressionConfig) -> None:
        """Update the compression configuration."""
        self._config = config

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        # Rough estimate: ~4 characters per token
        return len(content) // 4

    def _is_eligible_for_compression(self, entry: MemoryEntry) -> bool:
        """Check if a memory entry is eligible for compression."""
        now = datetime.now(timezone.utc)
        age = now - entry.created_at
        min_age = timedelta(hours=self._config.min_age_hours)

        # Check age
        if age < min_age:
            return False

        # Check importance threshold
        if entry.importance.effective_score >= self._config.importance_threshold:
            return False

        # Check preserved tags
        for tag in self._config.preserve_tags:
            if tag in entry.tags:
                return False

        return True

    def filter_eligible(
        self,
        entries: list[MemoryEntry],
    ) -> tuple[list[MemoryEntry], list[MemoryEntry]]:
        """Filter entries into eligible and preserved.

        Returns:
            Tuple of (eligible_entries, preserved_entries)
        """
        eligible = []
        preserved = []

        for entry in entries:
            if self._is_eligible_for_compression(entry):
                eligible.append(entry)
            else:
                preserved.append(entry)

        return eligible, preserved

    async def compress(
        self,
        entries: list[MemoryEntry],
        strategy: CompressionStrategy | None = None,
    ) -> CompressionResult:
        """Compress a batch of memory entries.

        Args:
            entries: Memory entries to compress
            strategy: Specific strategy to use (or auto-select)

        Returns:
            Compression result with new entries and statistics
        """
        result = CompressionResult(
            original_count=len(entries),
            original_tokens=sum(self._estimate_tokens(e.content) for e in entries),
        )

        # Filter eligible entries
        eligible, preserved = self.filter_eligible(entries)
        result.preserved_count = len(preserved)

        if not eligible:
            logger.info("No eligible entries for compression")
            result.compressed_entries = preserved
            result.compressed_count = len(preserved)
            result.compressed_tokens = result.original_tokens
            return result

        # Select or use provided strategy
        compressor = strategy
        if compressor is None:
            compressor = self._select_strategy(eligible)

        if compressor is None:
            logger.warning("No suitable compression strategy found")
            result.compressed_entries = entries
            result.compressed_count = len(entries)
            result.compressed_tokens = result.original_tokens
            return result

        # Apply compression
        try:
            compressed = await compressor.compress(eligible, self._config)

            # Track removed IDs
            compressed_ids = {e.id for e in compressed}
            for entry in eligible:
                if entry.id not in compressed_ids:
                    result.removed_ids.append(entry.id)

            # Combine with preserved entries
            result.compressed_entries = preserved + compressed
            result.compressed_count = len(result.compressed_entries)
            result.compressed_tokens = sum(
                self._estimate_tokens(e.content)
                for e in result.compressed_entries
            )

            logger.info(
                "Compression complete",
                original=result.original_count,
                compressed=result.compressed_count,
                ratio=f"{result.compression_ratio:.2%}",
                token_savings=result.token_savings,
            )

        except Exception as e:
            logger.error("Compression failed", error=str(e))
            result.compressed_entries = entries
            result.compressed_count = len(entries)
            result.compressed_tokens = result.original_tokens

        # Store in history
        self._compression_history.append(result)

        return result

    def _select_strategy(
        self,
        entries: list[MemoryEntry],
    ) -> CompressionStrategy | None:
        """Select the best compression strategy for the entries."""
        for strategy in self._strategies:
            if strategy.can_compress(entries):
                return strategy
        return None

    async def auto_compress(
        self,
        entries: list[MemoryEntry],
        max_entries: int | None = None,
    ) -> CompressionResult:
        """Automatically compress entries to meet size constraints.

        Args:
            entries: All memory entries
            max_entries: Maximum number of entries to keep

        Returns:
            Compression result
        """
        if max_entries is None:
            max_entries = int(len(entries) * self._config.target_ratio)

        result = CompressionResult(original_count=len(entries))

        # Filter eligible entries first
        eligible, preserved = self.filter_eligible(entries)

        if not eligible or len(entries) <= max_entries:
            result.compressed_entries = entries
            result.compressed_count = len(entries)
            result.compressed_tokens = sum(
                self._estimate_tokens(e.content) for e in entries
            )
            result.original_tokens = result.compressed_tokens
            return result

        # Compress all eligible entries at once
        if self._strategies:
            strategy = self._select_strategy(eligible)
            if strategy:
                compressed_eligible = await strategy.compress(eligible, self._config)
            else:
                compressed_eligible = eligible
        else:
            compressed_eligible = eligible

        # Combine preserved + compressed
        result.compressed_entries = preserved + compressed_eligible
        result.compressed_count = len(result.compressed_entries)

        # Track removed IDs
        compressed_ids = {e.id for e in compressed_eligible}
        for entry in eligible:
            if entry.id not in compressed_ids:
                result.removed_ids.append(entry.id)

        result.compressed_tokens = sum(
            self._estimate_tokens(e.content) for e in result.compressed_entries
        )
        result.original_tokens = sum(
            self._estimate_tokens(e.content) for e in entries
        )

        return result

    def get_compression_history(self) -> list[CompressionResult]:
        """Get the history of compression operations."""
        return self._compression_history.copy()

    def get_stats(self) -> dict:
        """Get compression pipeline statistics."""
        if not self._compression_history:
            return {
                "total_compressions": 0,
                "total_entries_processed": 0,
                "total_entries_removed": 0,
                "total_token_savings": 0,
                "avg_compression_ratio": 0,
            }

        total_original = sum(r.original_count for r in self._compression_history)
        total_compressed = sum(r.compressed_count for r in self._compression_history)
        total_removed = sum(len(r.removed_ids) for r in self._compression_history)
        total_token_savings = sum(r.token_savings for r in self._compression_history)

        return {
            "total_compressions": len(self._compression_history),
            "total_entries_processed": total_original,
            "total_entries_removed": total_removed,
            "total_token_savings": total_token_savings,
            "avg_compression_ratio": total_compressed / total_original if total_original > 0 else 1.0,
            "strategies": [type(s).__name__ for s in self._strategies],
        }
