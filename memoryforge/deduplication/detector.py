"""Duplicate detection for memory entries."""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Callable
from uuid import UUID

import structlog

from memoryforge.core.types import MemoryEntry

logger = structlog.get_logger()


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""

    # Similarity threshold (0.0 - 1.0)
    similarity_threshold: float = 0.85

    # Enable exact match detection
    enable_exact_match: bool = True

    # Enable fuzzy matching
    enable_fuzzy_match: bool = True

    # Enable semantic similarity (requires embeddings)
    enable_semantic_match: bool = False

    # Minimum content length to process
    min_content_length: int = 10

    # Maximum entries to compare (for performance)
    max_comparison_entries: int = 1000

    # Normalize content before comparison
    normalize_content: bool = True

    # Consider tags in similarity
    consider_tags: bool = True

    # Tag similarity weight
    tag_weight: float = 0.2


@dataclass
class DuplicateMatch:
    """A pair of duplicate or similar entries."""

    entry1_id: UUID
    entry2_id: UUID
    similarity_score: float
    match_type: str  # "exact", "fuzzy", "semantic"
    matched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "entry1_id": str(self.entry1_id),
            "entry2_id": str(self.entry2_id),
            "similarity_score": round(self.similarity_score, 4),
            "match_type": self.match_type,
            "matched_at": self.matched_at.isoformat(),
        }


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""

    total_entries: int
    duplicates_found: int
    matches: list[DuplicateMatch]
    processing_time_ms: float
    unique_groups: int = 0

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "duplicates_found": self.duplicates_found,
            "unique_groups": self.unique_groups,
            "matches": [m.to_dict() for m in self.matches],
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class DuplicateDetector:
    """Detects duplicate and similar memory entries.

    Provides:
    - Exact hash-based matching
    - Fuzzy text matching
    - Semantic similarity (with embeddings)
    - Configurable thresholds
    """

    def __init__(
        self,
        config: DeduplicationConfig | None = None,
    ):
        """Initialize duplicate detector.

        Args:
            config: Deduplication configuration
        """
        self._config = config or DeduplicationConfig()

        # Hash index for exact matching
        self._hash_index: dict[str, list[UUID]] = defaultdict(list)

        # Statistics
        self._total_processed = 0
        self._duplicates_found = 0

    def detect_duplicates(
        self,
        entries: list[MemoryEntry],
    ) -> DeduplicationResult:
        """Detect duplicates in a list of entries.

        Args:
            entries: Entries to check for duplicates

        Returns:
            Deduplication result
        """
        start_time = datetime.now(timezone.utc)
        matches: list[DuplicateMatch] = []

        # Filter entries
        valid_entries = [
            e for e in entries
            if len(e.content) >= self._config.min_content_length
        ]

        # Limit for performance
        if len(valid_entries) > self._config.max_comparison_entries:
            valid_entries = valid_entries[:self._config.max_comparison_entries]

        # Exact matching
        if self._config.enable_exact_match:
            exact_matches = self._find_exact_matches(valid_entries)
            matches.extend(exact_matches)

        # Fuzzy matching (for non-exact duplicates)
        if self._config.enable_fuzzy_match:
            exact_ids = {m.entry1_id for m in matches} | {m.entry2_id for m in matches}
            fuzzy_matches = self._find_fuzzy_matches(valid_entries, exact_ids)
            matches.extend(fuzzy_matches)

        # Calculate processing time
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Count unique duplicate groups
        unique_groups = self._count_groups(matches)

        # Update statistics
        self._total_processed += len(valid_entries)
        self._duplicates_found += len(matches)

        result = DeduplicationResult(
            total_entries=len(valid_entries),
            duplicates_found=len(matches),
            matches=matches,
            processing_time_ms=elapsed,
            unique_groups=unique_groups,
        )

        logger.debug(
            "Duplicate detection complete",
            entries=len(valid_entries),
            duplicates=len(matches),
        )

        return result

    def _find_exact_matches(
        self,
        entries: list[MemoryEntry],
    ) -> list[DuplicateMatch]:
        """Find exact duplicate matches.

        Args:
            entries: Entries to check

        Returns:
            List of exact matches
        """
        matches = []
        hash_to_entries: dict[str, list[MemoryEntry]] = defaultdict(list)

        for entry in entries:
            content_hash = self._hash_content(entry.content)
            hash_to_entries[content_hash].append(entry)

        # Find groups with multiple entries
        for content_hash, group in hash_to_entries.items():
            if len(group) > 1:
                # Create pairwise matches
                for i, entry1 in enumerate(group):
                    for entry2 in group[i + 1:]:
                        matches.append(DuplicateMatch(
                            entry1_id=entry1.id,
                            entry2_id=entry2.id,
                            similarity_score=1.0,
                            match_type="exact",
                        ))

        return matches

    def _find_fuzzy_matches(
        self,
        entries: list[MemoryEntry],
        exclude_ids: set[UUID],
    ) -> list[DuplicateMatch]:
        """Find fuzzy (similar but not exact) matches.

        Args:
            entries: Entries to check
            exclude_ids: IDs to exclude (already matched)

        Returns:
            List of fuzzy matches
        """
        matches = []
        compared: set[tuple[UUID, UUID]] = set()

        for i, entry1 in enumerate(entries):
            if entry1.id in exclude_ids:
                continue

            for entry2 in entries[i + 1:]:
                if entry2.id in exclude_ids:
                    continue

                # Skip if already compared
                pair = tuple(sorted([entry1.id, entry2.id]))
                if pair in compared:
                    continue
                compared.add(pair)

                # Calculate similarity
                similarity = self._calculate_similarity(entry1, entry2)

                if similarity >= self._config.similarity_threshold:
                    matches.append(DuplicateMatch(
                        entry1_id=entry1.id,
                        entry2_id=entry2.id,
                        similarity_score=similarity,
                        match_type="fuzzy",
                    ))

        return matches

    def _hash_content(self, content: str) -> str:
        """Create hash of content for exact matching.

        Args:
            content: Content to hash

        Returns:
            Content hash
        """
        normalized = self._normalize(content) if self._config.normalize_content else content
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _normalize(self, content: str) -> str:
        """Normalize content for comparison.

        Args:
            content: Content to normalize

        Returns:
            Normalized content
        """
        # Lowercase
        text = content.lower()

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove punctuation for comparison
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def _calculate_similarity(
        self,
        entry1: MemoryEntry,
        entry2: MemoryEntry,
    ) -> float:
        """Calculate similarity between two entries.

        Args:
            entry1: First entry
            entry2: Second entry

        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Content similarity
        content1 = self._normalize(entry1.content) if self._config.normalize_content else entry1.content
        content2 = self._normalize(entry2.content) if self._config.normalize_content else entry2.content

        content_similarity = SequenceMatcher(None, content1, content2).ratio()

        # Tag similarity
        if self._config.consider_tags and (entry1.tags or entry2.tags):
            tags1 = set(entry1.tags)
            tags2 = set(entry2.tags)

            if tags1 or tags2:
                tag_overlap = len(tags1 & tags2)
                tag_union = len(tags1 | tags2)
                tag_similarity = tag_overlap / tag_union if tag_union > 0 else 0.0
            else:
                tag_similarity = 0.0

            # Weighted combination
            weight = self._config.tag_weight
            similarity = (1 - weight) * content_similarity + weight * tag_similarity
        else:
            similarity = content_similarity

        return similarity

    def _count_groups(self, matches: list[DuplicateMatch]) -> int:
        """Count unique duplicate groups using union-find.

        Args:
            matches: List of duplicate matches

        Returns:
            Number of unique groups
        """
        if not matches:
            return 0

        # Build union-find structure
        parent: dict[UUID, UUID] = {}

        def find(x: UUID) -> UUID:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: UUID, y: UUID) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union all matched pairs
        for match in matches:
            union(match.entry1_id, match.entry2_id)

        # Count unique roots
        roots = {find(id) for id in parent}
        return len(roots)

    def is_duplicate(
        self,
        entry: MemoryEntry,
        existing_entries: list[MemoryEntry],
    ) -> tuple[bool, DuplicateMatch | None]:
        """Check if entry is a duplicate of any existing entry.

        Args:
            entry: Entry to check
            existing_entries: Existing entries to compare against

        Returns:
            Tuple of (is_duplicate, matching_entry)
        """
        if len(entry.content) < self._config.min_content_length:
            return False, None

        entry_hash = self._hash_content(entry.content)

        # Check exact matches first
        if self._config.enable_exact_match:
            for existing in existing_entries:
                existing_hash = self._hash_content(existing.content)
                if entry_hash == existing_hash:
                    match = DuplicateMatch(
                        entry1_id=entry.id,
                        entry2_id=existing.id,
                        similarity_score=1.0,
                        match_type="exact",
                    )
                    return True, match

        # Check fuzzy matches
        if self._config.enable_fuzzy_match:
            for existing in existing_entries:
                similarity = self._calculate_similarity(entry, existing)
                if similarity >= self._config.similarity_threshold:
                    match = DuplicateMatch(
                        entry1_id=entry.id,
                        entry2_id=existing.id,
                        similarity_score=similarity,
                        match_type="fuzzy",
                    )
                    return True, match

        return False, None

    def find_similar(
        self,
        entry: MemoryEntry,
        candidates: list[MemoryEntry],
        top_k: int = 5,
    ) -> list[tuple[MemoryEntry, float]]:
        """Find most similar entries.

        Args:
            entry: Entry to find similar entries for
            candidates: Candidate entries
            top_k: Number of results

        Returns:
            List of (entry, similarity) tuples
        """
        similarities = []

        for candidate in candidates:
            if candidate.id == entry.id:
                continue

            similarity = self._calculate_similarity(entry, candidate)
            similarities.append((candidate, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_stats(self) -> dict:
        """Get detector statistics.

        Returns:
            Statistics dict
        """
        return {
            "total_processed": self._total_processed,
            "duplicates_found": self._duplicates_found,
            "config": {
                "similarity_threshold": self._config.similarity_threshold,
                "enable_exact_match": self._config.enable_exact_match,
                "enable_fuzzy_match": self._config.enable_fuzzy_match,
                "normalize_content": self._config.normalize_content,
            },
        }


def create_detector(
    threshold: float = 0.85,
    **config_kwargs,
) -> DuplicateDetector:
    """Create a duplicate detector with specified threshold.

    Args:
        threshold: Similarity threshold
        **config_kwargs: Additional config options

    Returns:
        Configured DuplicateDetector
    """
    config = DeduplicationConfig(
        similarity_threshold=threshold,
        **config_kwargs,
    )
    return DuplicateDetector(config)
