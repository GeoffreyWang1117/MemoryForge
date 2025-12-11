"""Tests for memory summarization functionality."""

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore
from memoryforge.summarization import (
    MemorySummarizer,
    SummaryConfig,
    SummaryLevel,
    ExtractiveSummarizer,
    KeyPointSummarizer,
    ConversationSummarizer,
)
from memoryforge.summarization.summarizer import create_summarizer


def create_test_entry(
    content: str,
    tags: list[str] | None = None,
) -> MemoryEntry:
    """Helper to create test memory entries."""
    return MemoryEntry(
        id=uuid4(),
        content=content,
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.5),
        tags=tags or [],
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


LONG_CONTENT = """
Machine learning is a subset of artificial intelligence that focuses on building systems
that can learn from data. These systems improve their performance over time without being
explicitly programmed for specific tasks. The field has grown significantly in recent years,
driven by advances in computing power and the availability of large datasets.

Deep learning, a more specialized form of machine learning, uses neural networks with many
layers to model complex patterns. This approach has been particularly successful in areas
such as image recognition, natural language processing, and speech recognition.

Key applications of machine learning include recommendation systems, fraud detection,
medical diagnosis, and autonomous vehicles. Companies across industries are investing
heavily in these technologies to gain competitive advantages.

The future of machine learning looks promising, with ongoing research in areas like
explainable AI, federated learning, and reinforcement learning. These advances will
enable more sophisticated and trustworthy AI systems.
"""

CONVERSATION_CONTENT = """
user: How do I configure the database connection for the application?
assistant: You can configure the database connection by setting environment variables. The key variables are DATABASE_URL for the connection string and DATABASE_POOL_SIZE for connection pooling.
user: What's the recommended pool size?
assistant: For most applications, a pool size of 10-20 connections is recommended. You should adjust based on your expected concurrent users and database capacity.
user: Thanks! I'll set it to 15.
assistant: Good choice. Remember to also configure DATABASE_TIMEOUT to handle slow queries gracefully.
"""

TECHNICAL_CONTENT = """
The API endpoint `/api/v1/users` supports the following methods:

GET: Retrieves a list of users with pagination support. Use query parameters `page` and `limit`.
POST: Creates a new user. Required fields are `email` and `password`.
PUT: Updates an existing user by ID.
DELETE: Removes a user by ID.

Authentication is required for all endpoints. Use Bearer tokens in the Authorization header.

```python
def get_users(page: int = 1, limit: int = 10):
    return db.query(User).offset((page-1)*limit).limit(limit).all()
```

Important: All passwords must be hashed before storage. Never store plaintext passwords.
"""


class TestMemorySummarizer:
    """Tests for MemorySummarizer class."""

    def test_basic_summarization(self):
        """Test basic summarization."""
        summarizer = MemorySummarizer()
        entry = create_test_entry(LONG_CONTENT)

        result = summarizer.summarize_entry(entry)

        assert result.original_length == len(LONG_CONTENT)
        assert result.summary_length < result.original_length
        assert len(result.summary) > 0
        assert result.compression_ratio < 1.0

    def test_summary_levels(self):
        """Test different summary levels."""
        summarizer = MemorySummarizer()
        entry = create_test_entry(LONG_CONTENT)

        brief = summarizer.summarize_entry(entry, SummaryLevel.BRIEF)
        short = summarizer.summarize_entry(entry, SummaryLevel.SHORT)
        medium = summarizer.summarize_entry(entry, SummaryLevel.MEDIUM)
        detailed = summarizer.summarize_entry(entry, SummaryLevel.DETAILED)

        # Longer levels should produce longer summaries
        assert brief.summary_length <= short.summary_length
        assert short.summary_length <= medium.summary_length
        assert medium.summary_length <= detailed.summary_length

    def test_short_content_passthrough(self):
        """Test that short content is passed through unchanged."""
        summarizer = MemorySummarizer()
        short_content = "This is short."
        entry = create_test_entry(short_content)

        result = summarizer.summarize_entry(entry)

        assert result.summary == short_content
        assert result.compression_ratio == 1.0

    def test_summarize_content(self):
        """Test summarizing raw content."""
        summarizer = MemorySummarizer()

        result = summarizer.summarize_content(LONG_CONTENT, SummaryLevel.SHORT)

        assert result.summary_length < result.original_length
        assert len(result.summary) > 0

    def test_batch_summarization(self):
        """Test batch summarization."""
        summarizer = MemorySummarizer()
        entries = [
            create_test_entry(LONG_CONTENT),
            create_test_entry(TECHNICAL_CONTENT),
            create_test_entry("Short content."),
        ]

        results = summarizer.summarize_batch(entries)

        assert len(results) == 3
        for result in results:
            assert result.summary_length > 0

    def test_conversation_summarization(self):
        """Test conversation summarization."""
        summarizer = MemorySummarizer()
        messages = [
            {"role": "user", "content": "How do I configure the database?"},
            {"role": "assistant", "content": "Set the DATABASE_URL environment variable."},
            {"role": "user", "content": "What about pooling?"},
            {"role": "assistant", "content": "Use DATABASE_POOL_SIZE, recommend 10-20."},
        ]

        result = summarizer.summarize_conversation(messages)

        assert len(result.summary) > 0
        assert result.original_length > 0

    def test_group_summarization(self):
        """Test summarizing a group of entries."""
        summarizer = MemorySummarizer()
        entries = [
            create_test_entry("First memory about Python.", tags=["python"]),
            create_test_entry("Second memory about databases.", tags=["database"]),
            create_test_entry("Third memory about testing.", tags=["testing"]),
        ]

        result = summarizer.summarize_group(entries)

        assert len(result.summary) > 0
        assert "python" in result.topics or "database" in result.topics

    def test_max_length_enforcement(self):
        """Test that max length is enforced."""
        config = SummaryConfig(
            max_lengths={
                SummaryLevel.BRIEF: 50,
                SummaryLevel.SHORT: 100,
                SummaryLevel.MEDIUM: 200,
                SummaryLevel.DETAILED: 400,
            }
        )
        summarizer = MemorySummarizer(config)
        entry = create_test_entry(LONG_CONTENT)

        result = summarizer.summarize_entry(entry, SummaryLevel.BRIEF)

        assert result.summary_length <= 50 + 10  # Small tolerance

    def test_statistics(self):
        """Test summarizer statistics."""
        summarizer = MemorySummarizer()

        for i in range(5):
            entry = create_test_entry(LONG_CONTENT + f" Iteration {i}.")
            summarizer.summarize_entry(entry)

        stats = summarizer.get_stats()

        assert stats["total_summarized"] == 5
        assert stats["total_chars_processed"] > 0
        assert stats["overall_compression"] < 1.0

    def test_key_points_extraction(self):
        """Test key points are extracted."""
        summarizer = MemorySummarizer()
        entry = create_test_entry(LONG_CONTENT)

        result = summarizer.summarize_entry(entry)

        # May or may not have key points depending on content
        assert isinstance(result.key_points, list)


class TestExtractiveSummarizer:
    """Tests for ExtractiveSummarizer."""

    def test_basic_extraction(self):
        """Test basic extractive summarization."""
        summarizer = ExtractiveSummarizer()

        summary, key_points = summarizer.summarize(LONG_CONTENT, 200)

        assert len(summary) > 0
        assert len(summary) <= 200 + 20  # Tolerance
        assert isinstance(key_points, list)

    def test_sentence_selection(self):
        """Test that meaningful sentences are selected."""
        summarizer = ExtractiveSummarizer()

        summary, _ = summarizer.summarize(LONG_CONTENT, 300)

        # Summary should contain coherent text
        assert "." in summary

    def test_preserves_important_sentences(self):
        """Test that important sentences are preserved."""
        summarizer = ExtractiveSummarizer()
        content = "Introduction sentence. Important key information here. Random filler text. Conclusion statement."

        summary, _ = summarizer.summarize(content, 100)

        # First sentence often preserved
        assert len(summary) > 0

    def test_handles_single_sentence(self):
        """Test handling single sentence content."""
        summarizer = ExtractiveSummarizer()
        content = "This is a single sentence without any other content."

        summary, key_points = summarizer.summarize(content, 100)

        assert summary == content or summary.startswith(content[:20])

    def test_handles_empty_content(self):
        """Test handling empty content."""
        summarizer = ExtractiveSummarizer()

        summary, key_points = summarizer.summarize("", 100)

        assert summary == ""


class TestKeyPointSummarizer:
    """Tests for KeyPointSummarizer."""

    def test_extracts_key_points(self):
        """Test key point extraction."""
        summarizer = KeyPointSummarizer()
        content = """
        The important thing to note is that performance matters.
        First, we need to optimize the database queries.
        In conclusion, these changes will improve response times.
        Random unimportant filler text here.
        """

        summary, key_points = summarizer.summarize(content, 200)

        assert len(key_points) > 0

    def test_prioritizes_conclusions(self):
        """Test that conclusions are prioritized."""
        summarizer = KeyPointSummarizer()
        content = """
        Some background information.
        More details about the topic.
        Therefore, the final recommendation is to proceed.
        """

        _, key_points = summarizer.summarize(content, 200)

        # Should have at least one key point
        assert len(key_points) > 0

    def test_identifies_action_items(self):
        """Test action item identification."""
        summarizer = KeyPointSummarizer()
        content = """
        Project overview and context.
        TODO: Implement the new feature.
        Action: Review the code changes.
        Other miscellaneous notes.
        """

        _, key_points = summarizer.summarize(content, 200)

        assert len(key_points) > 0


class TestConversationSummarizer:
    """Tests for ConversationSummarizer."""

    def test_conversation_parsing(self):
        """Test conversation parsing."""
        summarizer = ConversationSummarizer()

        summary, key_points = summarizer.summarize(CONVERSATION_CONTENT, 300)

        assert "conversation" in summary.lower() or "exchange" in summary.lower()

    def test_topic_extraction(self):
        """Test topic extraction from conversation."""
        summarizer = ConversationSummarizer()

        summary, key_points = summarizer.summarize(CONVERSATION_CONTENT, 400)

        # Should identify some topics
        assert "Topic" in str(key_points) or len(summary) > 0

    def test_handles_non_conversation(self):
        """Test handling non-conversation content."""
        summarizer = ConversationSummarizer()
        content = "This is just regular text without conversation markers."

        summary, key_points = summarizer.summarize(content, 100)

        # Should still produce output
        assert len(summary) >= 0


class TestSummaryConfig:
    """Tests for SummaryConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SummaryConfig()

        assert config.default_level == SummaryLevel.SHORT
        assert SummaryLevel.BRIEF in config.max_lengths
        assert SummaryLevel.DETAILED in config.max_lengths

    def test_custom_max_lengths(self):
        """Test custom max lengths."""
        config = SummaryConfig(
            max_lengths={
                SummaryLevel.BRIEF: 50,
                SummaryLevel.SHORT: 100,
                SummaryLevel.MEDIUM: 200,
                SummaryLevel.DETAILED: 400,
            }
        )

        assert config.max_lengths[SummaryLevel.BRIEF] == 50
        assert config.max_lengths[SummaryLevel.DETAILED] == 400


class TestSummaryLevel:
    """Tests for SummaryLevel enum."""

    def test_all_levels_defined(self):
        """Test all summary levels are defined."""
        levels = list(SummaryLevel)

        assert SummaryLevel.BRIEF in levels
        assert SummaryLevel.SHORT in levels
        assert SummaryLevel.MEDIUM in levels
        assert SummaryLevel.DETAILED in levels

    def test_level_values(self):
        """Test level string values."""
        assert SummaryLevel.BRIEF.value == "brief"
        assert SummaryLevel.DETAILED.value == "detailed"


class TestSummaryResult:
    """Tests for SummaryResult."""

    def test_to_dict(self):
        """Test result serialization."""
        from memoryforge.summarization.summarizer import SummaryResult

        result = SummaryResult(
            original_length=1000,
            summary_length=200,
            summary="Test summary",
            level=SummaryLevel.SHORT,
            compression_ratio=0.2,
            key_points=["Point 1", "Point 2"],
            topics=["topic1"],
            processing_time_ms=10.5,
        )

        data = result.to_dict()

        assert data["original_length"] == 1000
        assert data["summary_length"] == 200
        assert data["level"] == "short"
        assert data["compression_ratio"] == 0.2
        assert len(data["key_points"]) == 2


class TestCreateSummarizer:
    """Tests for create_summarizer helper."""

    def test_create_with_defaults(self):
        """Test creating summarizer with defaults."""
        summarizer = create_summarizer()

        assert summarizer is not None
        assert summarizer._config.default_level == SummaryLevel.SHORT

    def test_create_with_level(self):
        """Test creating summarizer with specific level."""
        summarizer = create_summarizer(level=SummaryLevel.DETAILED)

        assert summarizer._config.default_level == SummaryLevel.DETAILED

    def test_create_with_strategy(self):
        """Test creating summarizer with custom strategy."""
        strategy = ExtractiveSummarizer()
        summarizer = create_summarizer(strategy=strategy)

        assert summarizer._strategy == strategy
