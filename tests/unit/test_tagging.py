"""Tests for auto-tagging functionality."""

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore
from memoryforge.tagging import (
    AutoTagger,
    TaggingConfig,
    TagRule,
    KeywordExtractor,
    EntityExtractor,
    TopicExtractor,
)
from memoryforge.tagging.auto_tagger import create_auto_tagger, COMMON_RULES


def create_test_entry(
    content: str,
    tags: list[str] | None = None,
    layer: MemoryLayer = MemoryLayer.WORKING,
    metadata: dict | None = None,
) -> MemoryEntry:
    """Helper to create test memory entries."""
    return MemoryEntry(
        id=uuid4(),
        content=content,
        layer=layer,
        importance=ImportanceScore(base_score=0.5),
        tags=tags or [],
        metadata=metadata or {},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestAutoTagger:
    """Tests for AutoTagger class."""

    def test_basic_tagging(self):
        """Test basic auto-tagging."""
        tagger = AutoTagger()
        entry = create_test_entry(
            "The python function handles database queries efficiently"
        )

        result = tagger.tag_entry(entry)

        assert result.memory_id == str(entry.id)
        assert len(result.all_tags) > 0
        assert result.processing_time_ms >= 0

    def test_keyword_extraction(self):
        """Test keyword extraction."""
        tagger = AutoTagger()
        entry = create_test_entry(
            "Python Python Python function function database"
        )

        result = tagger.tag_entry(entry)

        assert "python" in result.all_tags

    def test_entity_extraction(self):
        """Test entity extraction from content."""
        tagger = AutoTagger()

        # Test code detection
        entry = create_test_entry("```python\ndef hello():\n    pass\n```")
        result = tagger.tag_entry(entry)
        assert "code" in result.all_tags

        # Test URL detection
        entry = create_test_entry("Check https://example.com for details")
        result = tagger.tag_entry(entry)
        assert "url" in result.all_tags

        # Test email detection
        entry = create_test_entry("Contact user@example.com for help")
        result = tagger.tag_entry(entry)
        assert "email" in result.all_tags

    def test_error_detection(self):
        """Test error pattern detection."""
        tagger = AutoTagger()
        entry = create_test_entry(
            "Error: Connection failed with exception"
        )

        result = tagger.tag_entry(entry)

        assert "error" in result.all_tags

    def test_question_detection(self):
        """Test question pattern detection."""
        tagger = AutoTagger()
        entry = create_test_entry(
            "How do I configure the database connection?"
        )

        result = tagger.tag_entry(entry)

        assert "question" in result.all_tags

    def test_preserve_existing_tags(self):
        """Test that existing tags are preserved."""
        tagger = AutoTagger()
        entry = create_test_entry(
            "Simple content",
            tags=["existing-tag", "another-tag"],
        )

        result = tagger.tag_entry(entry, merge_existing=True)

        assert "existing-tag" in result.all_tags
        assert "another-tag" in result.all_tags

    def test_replace_existing_tags(self):
        """Test replacing existing tags."""
        tagger = AutoTagger()
        entry = create_test_entry(
            "Python function for database queries",
            tags=["old-tag"],
        )

        result = tagger.tag_entry(entry, merge_existing=False)

        assert "old-tag" not in result.all_tags

    def test_max_tags_limit(self):
        """Test that max tags limit is respected."""
        config = TaggingConfig(max_tags=3)
        tagger = AutoTagger(config)
        entry = create_test_entry(
            "Python JavaScript TypeScript React Vue Angular database API testing code"
        )

        result = tagger.tag_entry(entry)

        assert len(result.all_tags) <= 3

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        config = TaggingConfig(min_confidence=0.8)
        tagger = AutoTagger(config)
        entry = create_test_entry("Simple test content")

        result = tagger.tag_entry(entry)

        # All tags should meet confidence threshold
        for tag, score in result.confidence_scores.items():
            assert score >= 0.8

    def test_custom_rule(self):
        """Test custom tagging rules."""
        tagger = AutoTagger()
        rule = TagRule(
            name="test_rule",
            pattern=r"\bspecial\s+keyword\b",
            tags=["special", "custom-rule"],
        )
        tagger.add_rule(rule)

        entry = create_test_entry("This contains special keyword pattern")
        result = tagger.tag_entry(entry)

        assert "special" in result.all_tags
        assert "custom-rule" in result.all_tags

    def test_rule_case_sensitivity(self):
        """Test rule case sensitivity."""
        tagger = AutoTagger()

        # Case insensitive rule (default)
        rule1 = TagRule(
            name="insensitive",
            pattern=r"\bTEST\b",
            tags=["case-insensitive"],
            case_sensitive=False,
        )
        tagger.add_rule(rule1)

        entry = create_test_entry("This is a test")
        result = tagger.tag_entry(entry)
        assert "case-insensitive" in result.all_tags

    def test_batch_tagging(self):
        """Test batch tagging multiple entries."""
        tagger = AutoTagger()
        entries = [
            create_test_entry("Python programming language"),
            create_test_entry("JavaScript web development"),
            create_test_entry("SQL database queries"),
        ]

        results = tagger.tag_batch(entries)

        assert len(results) == 3
        for result in results:
            assert len(result.all_tags) >= 0

    def test_suggest_tags(self):
        """Test tag suggestions without applying."""
        tagger = AutoTagger()
        suggestions = tagger.suggest_tags(
            "Python function for API database testing",
            top_n=5,
        )

        assert len(suggestions) <= 5
        for tag, score in suggestions:
            assert isinstance(tag, str)
            assert 0 <= score <= 1

    def test_common_rules(self):
        """Test common predefined rules."""
        tagger = create_auto_tagger(include_common_rules=True)

        # Bug report
        entry = create_test_entry("There's a bug in the authentication")
        result = tagger.tag_entry(entry)
        assert "bug" in result.all_tags or "issue" in result.all_tags

        # Security
        entry = create_test_entry("Fix the security vulnerability in auth")
        result = tagger.tag_entry(entry)
        assert "security" in result.all_tags

        # Performance
        entry = create_test_entry("Need to optimize the slow query")
        result = tagger.tag_entry(entry)
        assert "performance" in result.all_tags

    def test_custom_handler(self):
        """Test custom tag handler."""
        tagger = AutoTagger()

        def custom_handler(entry: MemoryEntry) -> list[tuple[str, float]]:
            if "special" in entry.content.lower():
                return [("custom-detected", 0.95)]
            return []

        tagger.add_handler(custom_handler)

        entry = create_test_entry("This is special content")
        result = tagger.tag_entry(entry)

        assert "custom-detected" in result.all_tags

    def test_statistics(self):
        """Test tagger statistics."""
        tagger = AutoTagger()

        for i in range(5):
            entry = create_test_entry(f"Content {i} with python code")
            tagger.tag_entry(entry)

        stats = tagger.get_stats()

        assert stats["total_tagged"] == 5
        assert stats["tags_added"] >= 0
        assert "config" in stats


class TestKeywordExtractor:
    """Tests for KeywordExtractor."""

    def test_basic_extraction(self):
        """Test basic keyword extraction."""
        extractor = KeywordExtractor()
        tags = extractor.extract("Python programming language Python Python")

        assert len(tags) > 0
        # Python should be extracted due to high frequency
        tag_names = [t[0] for t in tags]
        assert "python" in tag_names

    def test_stop_word_filtering(self):
        """Test that stop words are filtered."""
        extractor = KeywordExtractor()
        tags = extractor.extract("The and or but with from")

        # All stop words should be filtered
        assert len(tags) == 0

    def test_min_length_filtering(self):
        """Test minimum length filtering."""
        extractor = KeywordExtractor(min_length=5)
        tags = extractor.extract("cat dog programming")

        tag_names = [t[0] for t in tags]
        assert "cat" not in tag_names
        assert "dog" not in tag_names
        assert "programming" in tag_names

    def test_top_n_limit(self):
        """Test top N limit."""
        extractor = KeywordExtractor(top_n=3)
        tags = extractor.extract(
            "python java javascript typescript ruby golang swift kotlin"
        )

        assert len(tags) <= 3

    def test_boost_terms(self):
        """Test technical term boosting."""
        extractor = KeywordExtractor()
        extractor.boost_terms.add("specialized")

        tags = extractor.extract("specialized content specialized")

        tag_names = [t[0] for t in tags]
        assert "specialized" in tag_names


class TestEntityExtractor:
    """Tests for EntityExtractor."""

    def test_email_detection(self):
        """Test email pattern detection."""
        extractor = EntityExtractor()
        tags = extractor.extract("Contact user@example.com")

        tag_names = [t[0] for t in tags]
        assert "email" in tag_names

    def test_url_detection(self):
        """Test URL pattern detection."""
        extractor = EntityExtractor()
        tags = extractor.extract("Visit https://example.com")

        tag_names = [t[0] for t in tags]
        assert "url" in tag_names

    def test_code_block_detection(self):
        """Test code block detection."""
        extractor = EntityExtractor()
        tags = extractor.extract("```python\nprint('hello')\n```")

        tag_names = [t[0] for t in tags]
        assert "code-block" in tag_names

    def test_python_function_detection(self):
        """Test Python function detection."""
        extractor = EntityExtractor()
        tags = extractor.extract("def my_function(args):")

        tag_names = [t[0] for t in tags]
        assert "python" in tag_names

    def test_date_detection(self):
        """Test date pattern detection."""
        extractor = EntityExtractor()
        tags = extractor.extract("Date: 2024-01-15")

        tag_names = [t[0] for t in tags]
        assert "date" in tag_names

    def test_error_detection(self):
        """Test error pattern detection."""
        extractor = EntityExtractor()
        tags = extractor.extract("Error: Something went wrong")

        tag_names = [t[0] for t in tags]
        assert "error" in tag_names

    def test_git_command_detection(self):
        """Test git command detection."""
        extractor = EntityExtractor()
        tags = extractor.extract("Run git commit -m 'message'")

        tag_names = [t[0] for t in tags]
        assert "git" in tag_names


class TestTopicExtractor:
    """Tests for TopicExtractor."""

    def test_web_development_topic(self):
        """Test web development topic detection."""
        extractor = TopicExtractor()
        tags = extractor.extract("Using React and JavaScript for the frontend")

        tag_names = [t[0] for t in tags]
        assert "web-development" in tag_names

    def test_database_topic(self):
        """Test database topic detection."""
        extractor = TopicExtractor()
        tags = extractor.extract("Execute SQL query on postgres database")

        tag_names = [t[0] for t in tags]
        assert "database" in tag_names

    def test_devops_topic(self):
        """Test DevOps topic detection."""
        extractor = TopicExtractor()
        tags = extractor.extract("Deploy with Docker and Kubernetes")

        tag_names = [t[0] for t in tags]
        assert "devops" in tag_names

    def test_ml_topic(self):
        """Test machine learning topic detection."""
        extractor = TopicExtractor()
        tags = extractor.extract("Using LLM embeddings with transformer")

        tag_names = [t[0] for t in tags]
        assert "machine-learning" in tag_names

    def test_testing_topic(self):
        """Test testing topic detection."""
        extractor = TopicExtractor()
        tags = extractor.extract("Write pytest test with fixture and mock")

        tag_names = [t[0] for t in tags]
        assert "testing" in tag_names

    def test_min_matches_threshold(self):
        """Test minimum matches threshold."""
        extractor = TopicExtractor(min_matches=3)
        tags = extractor.extract("python")  # Only one match

        # Should not detect any topic with only one match
        assert len([t for t in tags if t[0] in extractor.topic_vocabularies]) == 0


class TestTaggingConfig:
    """Tests for TaggingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TaggingConfig()

        assert config.max_tags == 10
        assert config.min_confidence == 0.3
        assert config.enable_keywords is True
        assert config.enable_entities is True
        assert config.enable_topics is True
        assert config.enable_rules is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TaggingConfig(
            max_tags=5,
            min_confidence=0.5,
            enable_keywords=False,
        )

        assert config.max_tags == 5
        assert config.min_confidence == 0.5
        assert config.enable_keywords is False

    def test_disabled_methods(self):
        """Test disabling tagging methods."""
        config = TaggingConfig(
            enable_keywords=False,
            enable_entities=False,
            enable_topics=False,
            enable_rules=False,
        )
        tagger = AutoTagger(config)

        entry = create_test_entry("Python code with error")
        result = tagger.tag_entry(entry)

        # Should have no method-based tags
        assert len(result.methods_used) == 0


class TestTagRule:
    """Tests for TagRule."""

    def test_rule_matching(self):
        """Test rule pattern matching."""
        rule = TagRule(
            name="test",
            pattern=r"\btest\s+pattern\b",
            tags=["matched"],
        )

        assert rule.matches("This is a test pattern here")
        assert not rule.matches("This is different")

    def test_case_insensitive_matching(self):
        """Test case insensitive matching."""
        rule = TagRule(
            name="test",
            pattern=r"\bTEST\b",
            tags=["matched"],
            case_sensitive=False,
        )

        assert rule.matches("this is a test")
        assert rule.matches("this is a TEST")

    def test_case_sensitive_matching(self):
        """Test case sensitive matching."""
        rule = TagRule(
            name="test",
            pattern=r"\bTEST\b",
            tags=["matched"],
            case_sensitive=True,
        )

        assert not rule.matches("this is a test")
        assert rule.matches("this is a TEST")
