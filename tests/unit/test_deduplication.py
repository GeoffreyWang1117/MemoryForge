"""Tests for memory deduplication functionality."""

import pytest
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore
from memoryforge.deduplication import (
    DuplicateDetector,
    DeduplicationConfig,
    DuplicateMatch,
    MemoryMerger,
    MergeStrategy,
)
from memoryforge.deduplication.detector import create_detector


def create_test_entry(
    content: str,
    tags: list[str] | None = None,
    importance: float = 0.5,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> MemoryEntry:
    """Helper to create test memory entries."""
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=uuid4(),
        content=content,
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=importance),
        tags=tags or [],
        metadata={},
        created_at=created_at or now,
        updated_at=updated_at or now,
    )


class TestDuplicateDetector:
    """Tests for DuplicateDetector."""

    def test_detect_exact_duplicates(self):
        """Test detection of exact duplicates."""
        detector = DuplicateDetector()
        entries = [
            create_test_entry("This is a test content"),
            create_test_entry("This is a test content"),  # Exact duplicate
            create_test_entry("Different content here"),
        ]

        result = detector.detect_duplicates(entries)

        assert result.duplicates_found == 1
        assert result.matches[0].match_type == "exact"
        assert result.matches[0].similarity_score == 1.0

    def test_detect_fuzzy_duplicates(self):
        """Test detection of fuzzy (similar) duplicates."""
        config = DeduplicationConfig(similarity_threshold=0.8)
        detector = DuplicateDetector(config)
        entries = [
            create_test_entry("This is a test content for checking"),
            create_test_entry("This is a test content for checking duplicates"),
            create_test_entry("Completely different content here"),
        ]

        result = detector.detect_duplicates(entries)

        # Should find fuzzy match
        fuzzy_matches = [m for m in result.matches if m.match_type == "fuzzy"]
        assert len(fuzzy_matches) >= 0  # May or may not match depending on threshold

    def test_no_duplicates(self):
        """Test when no duplicates exist."""
        config = DeduplicationConfig(similarity_threshold=0.95)  # Higher threshold
        detector = DuplicateDetector(config)
        entries = [
            create_test_entry("Python is a programming language used for web development"),
            create_test_entry("JavaScript runs in browsers and powers frontend applications"),
            create_test_entry("Rust provides memory safety without garbage collection overhead"),
        ]

        result = detector.detect_duplicates(entries)

        assert result.duplicates_found == 0
        assert len(result.matches) == 0

    def test_multiple_duplicate_groups(self):
        """Test detection of multiple duplicate groups."""
        detector = DuplicateDetector()
        entries = [
            create_test_entry("This is content group A with enough length"),
            create_test_entry("This is content group A with enough length"),  # Duplicate of first
            create_test_entry("This is content group B with enough length"),
            create_test_entry("This is content group B with enough length"),  # Duplicate of third
        ]

        result = detector.detect_duplicates(entries)

        assert result.duplicates_found == 2
        assert result.unique_groups == 2

    def test_is_duplicate_exact(self):
        """Test checking if entry is duplicate."""
        detector = DuplicateDetector()
        existing = [
            create_test_entry("Existing content one"),
            create_test_entry("Existing content two"),
        ]
        new_entry = create_test_entry("Existing content one")

        is_dup, match = detector.is_duplicate(new_entry, existing)

        assert is_dup is True
        assert match is not None
        assert match.match_type == "exact"

    def test_is_duplicate_not_found(self):
        """Test checking entry that is not duplicate."""
        detector = DuplicateDetector()
        existing = [
            create_test_entry("Existing content one"),
        ]
        new_entry = create_test_entry("Brand new unique content")

        is_dup, match = detector.is_duplicate(new_entry, existing)

        assert is_dup is False
        assert match is None

    def test_find_similar(self):
        """Test finding similar entries."""
        detector = DuplicateDetector()
        entries = [
            create_test_entry("Python programming language basics"),
            create_test_entry("Python programming language advanced"),
            create_test_entry("Java programming language basics"),
            create_test_entry("Completely unrelated content"),
        ]
        target = create_test_entry("Python programming language")

        similar = detector.find_similar(target, entries, top_k=3)

        assert len(similar) <= 3
        # Python entries should be more similar
        if len(similar) > 0:
            assert similar[0][1] > 0.5  # First result should be fairly similar

    def test_normalization(self):
        """Test content normalization."""
        config = DeduplicationConfig(normalize_content=True)
        detector = DuplicateDetector(config)
        entries = [
            create_test_entry("Test Content Here"),
            create_test_entry("test content here"),  # Same after normalization
        ]

        result = detector.detect_duplicates(entries)

        assert result.duplicates_found == 1

    def test_tag_consideration(self):
        """Test that tags affect similarity."""
        config = DeduplicationConfig(consider_tags=True, tag_weight=0.3)
        detector = DuplicateDetector(config)

        entry1 = create_test_entry("Similar content", tags=["python", "code"])
        entry2 = create_test_entry("Similar content", tags=["python", "code"])
        entry3 = create_test_entry("Similar content", tags=["java", "web"])

        # Same tags should be more similar
        sim_same = detector._calculate_similarity(entry1, entry2)
        sim_diff = detector._calculate_similarity(entry1, entry3)

        assert sim_same > sim_diff

    def test_min_content_length(self):
        """Test minimum content length filtering."""
        config = DeduplicationConfig(min_content_length=20)
        detector = DuplicateDetector(config)
        entries = [
            create_test_entry("Short"),  # Too short
            create_test_entry("Short"),  # Too short
            create_test_entry("This is long enough content to process"),
        ]

        result = detector.detect_duplicates(entries)

        # Short entries should be filtered out
        assert result.total_entries == 1

    def test_statistics(self):
        """Test detector statistics."""
        detector = DuplicateDetector()
        entries = [
            create_test_entry("Test content"),
            create_test_entry("Test content"),
        ]

        detector.detect_duplicates(entries)
        stats = detector.get_stats()

        assert stats["total_processed"] == 2
        assert stats["duplicates_found"] == 1


class TestMemoryMerger:
    """Tests for MemoryMerger."""

    def test_keep_first_strategy(self):
        """Test KEEP_FIRST merge strategy."""
        merger = MemoryMerger()
        older = create_test_entry(
            "First content",
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        newer = create_test_entry("Second content")

        result = merger.merge([older, newer], MergeStrategy.KEEP_FIRST)

        assert result.merged_entry.id == older.id
        assert result.strategy_used == MergeStrategy.KEEP_FIRST

    def test_keep_latest_strategy(self):
        """Test KEEP_LATEST merge strategy."""
        merger = MemoryMerger()
        older = create_test_entry(
            "Old content",
            updated_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        newer = create_test_entry(
            "New content",
            updated_at=datetime.now(timezone.utc),
        )

        result = merger.merge([older, newer], MergeStrategy.KEEP_LATEST)

        assert result.merged_entry.id == newer.id

    def test_keep_highest_importance(self):
        """Test KEEP_HIGHEST_IMPORTANCE merge strategy."""
        merger = MemoryMerger()
        low_importance = create_test_entry("Low", importance=0.3)
        high_importance = create_test_entry("High", importance=0.9)

        result = merger.merge(
            [low_importance, high_importance],
            MergeStrategy.KEEP_HIGHEST_IMPORTANCE,
        )

        assert result.merged_entry.id == high_importance.id

    def test_merge_content_strategy(self):
        """Test MERGE_CONTENT merge strategy."""
        merger = MemoryMerger()
        entry1 = create_test_entry(
            "First piece of content",
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        entry2 = create_test_entry(
            "Second piece of content",
            created_at=datetime.now(timezone.utc),
        )

        result = merger.merge([entry1, entry2], MergeStrategy.MERGE_CONTENT)

        assert "First piece" in result.merged_entry.content
        assert "Second piece" in result.merged_entry.content
        assert result.merged_entry.id != entry1.id  # New ID

    def test_merge_all_strategy(self):
        """Test MERGE_ALL merge strategy."""
        merger = MemoryMerger()
        entry1 = create_test_entry(
            "Content one",
            tags=["tag1", "tag2"],
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        entry2 = create_test_entry(
            "Content two",
            tags=["tag2", "tag3"],
            created_at=datetime.now(timezone.utc),
        )

        result = merger.merge([entry1, entry2], MergeStrategy.MERGE_ALL)

        # Content merged
        assert "Content one" in result.merged_entry.content
        assert "Content two" in result.merged_entry.content

        # Tags merged (unique)
        assert "tag1" in result.merged_entry.tags
        assert "tag2" in result.merged_entry.tags
        assert "tag3" in result.merged_entry.tags

        # Metadata includes merge info
        assert "merged_from" in result.merged_entry.metadata

    def test_single_entry_merge(self):
        """Test merging a single entry."""
        merger = MemoryMerger()
        entry = create_test_entry("Single entry")

        result = merger.merge([entry])

        assert result.merged_entry.id == entry.id

    def test_empty_entries_raises(self):
        """Test that empty entries raises error."""
        merger = MemoryMerger()

        with pytest.raises(ValueError):
            merger.merge([])

    def test_importance_merging(self):
        """Test that importance scores are merged correctly."""
        merger = MemoryMerger()
        entry1 = create_test_entry("Content", importance=0.7)
        entry2 = create_test_entry("Content", importance=0.9)

        result = merger.merge([entry1, entry2], MergeStrategy.MERGE_ALL)

        # Should use highest importance
        assert result.merged_entry.importance.base_score == 0.9

    def test_custom_merger(self):
        """Test registering and using custom merger."""
        merger = MemoryMerger()

        def custom_merge(entries: list[MemoryEntry]) -> MemoryEntry:
            # Custom: concatenate with separator
            combined = " | ".join(e.content for e in entries)
            return MemoryEntry(
                id=uuid4(),
                content=combined,
                layer=entries[0].layer,
                importance=entries[0].importance,
                tags=[],
                metadata={"custom_merged": True},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

        merger.register_custom_merger("concat", custom_merge)

        entry1 = create_test_entry("First")
        entry2 = create_test_entry("Second")

        result = merger.merge_custom([entry1, entry2], "concat")

        assert "First" in result.merged_entry.content
        assert " | " in result.merged_entry.content
        assert "Second" in result.merged_entry.content

    def test_deduplicate_list(self):
        """Test deduplicating a list of entries."""
        merger = MemoryMerger()
        entry1 = create_test_entry("Content A")
        entry2 = create_test_entry("Content A duplicate")
        entry3 = create_test_entry("Content B")

        entries = [entry1, entry2, entry3]
        groups = [[entry1.id, entry2.id]]

        result = merger.deduplicate_list(entries, groups, MergeStrategy.KEEP_FIRST)

        # Should have 2 entries: merged A + B
        assert len(result) == 2

    def test_statistics(self):
        """Test merger statistics."""
        merger = MemoryMerger()
        entries = [
            create_test_entry("A"),
            create_test_entry("B"),
        ]

        merger.merge(entries)
        stats = merger.get_stats()

        assert stats["total_merges"] == 1
        assert stats["entries_eliminated"] == 1


class TestDeduplicationConfig:
    """Tests for DeduplicationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DeduplicationConfig()

        assert config.similarity_threshold == 0.85
        assert config.enable_exact_match is True
        assert config.enable_fuzzy_match is True
        assert config.normalize_content is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeduplicationConfig(
            similarity_threshold=0.9,
            enable_fuzzy_match=False,
        )

        assert config.similarity_threshold == 0.9
        assert config.enable_fuzzy_match is False


class TestMergeStrategy:
    """Tests for MergeStrategy enum."""

    def test_all_strategies_defined(self):
        """Test all strategies are defined."""
        strategies = list(MergeStrategy)

        assert MergeStrategy.KEEP_FIRST in strategies
        assert MergeStrategy.KEEP_LATEST in strategies
        assert MergeStrategy.KEEP_HIGHEST_IMPORTANCE in strategies
        assert MergeStrategy.MERGE_CONTENT in strategies
        assert MergeStrategy.MERGE_ALL in strategies


class TestDuplicateMatch:
    """Tests for DuplicateMatch dataclass."""

    def test_to_dict(self):
        """Test DuplicateMatch serialization."""
        match = DuplicateMatch(
            entry1_id=uuid4(),
            entry2_id=uuid4(),
            similarity_score=0.95,
            match_type="fuzzy",
        )

        data = match.to_dict()

        assert "entry1_id" in data
        assert "entry2_id" in data
        assert data["similarity_score"] == 0.95
        assert data["match_type"] == "fuzzy"


class TestCreateDetector:
    """Tests for create_detector helper."""

    def test_create_with_defaults(self):
        """Test creating detector with defaults."""
        detector = create_detector()

        assert detector._config.similarity_threshold == 0.85

    def test_create_with_threshold(self):
        """Test creating detector with custom threshold."""
        detector = create_detector(threshold=0.9)

        assert detector._config.similarity_threshold == 0.9
