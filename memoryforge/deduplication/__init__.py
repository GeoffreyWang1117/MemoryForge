"""Memory deduplication services."""

from memoryforge.deduplication.detector import (
    DuplicateDetector,
    DeduplicationConfig,
    DuplicateMatch,
    DeduplicationResult,
)
from memoryforge.deduplication.merger import (
    MemoryMerger,
    MergeStrategy,
    MergeResult,
)

__all__ = [
    "DuplicateDetector",
    "DeduplicationConfig",
    "DuplicateMatch",
    "DeduplicationResult",
    "MemoryMerger",
    "MergeStrategy",
    "MergeResult",
]
