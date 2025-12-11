"""Memory compression module for reducing memory footprint."""

from memoryforge.compression.pipeline import (
    CompressionPipeline,
    CompressionStrategy,
    CompressionResult,
    CompressionConfig,
    CompressionLevel,
)
from memoryforge.compression.strategies import (
    SummaryCompressor,
    ClusterCompressor,
    TimeWindowCompressor,
    DeduplicationCompressor,
)

__all__ = [
    "CompressionPipeline",
    "CompressionStrategy",
    "CompressionResult",
    "CompressionConfig",
    "CompressionLevel",
    "SummaryCompressor",
    "ClusterCompressor",
    "TimeWindowCompressor",
    "DeduplicationCompressor",
]
