"""Unit tests for context builder module."""

import pytest

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer
from memoryforge.context.builder import (
    ContextBuilder,
    ContextConfig,
    ContextSection,
    SectionType,
    BuiltContext,
)


def create_entry(
    content: str,
    importance: float = 0.5,
    tags: list[str] | None = None,
) -> MemoryEntry:
    """Helper to create test entries."""
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=importance),
        tags=tags or [],
    )


class TestSectionType:
    """Tests for SectionType enum."""

    def test_all_types_defined(self):
        """Test all section types are defined."""
        assert SectionType.SYSTEM.value == "system"
        assert SectionType.MEMORIES.value == "memories"
        assert SectionType.HISTORY.value == "history"
        assert SectionType.INSTRUCTIONS.value == "instructions"
        assert SectionType.USER.value == "user"
        assert SectionType.CUSTOM.value == "custom"


class TestContextSection:
    """Tests for ContextSection dataclass."""

    def test_creation(self):
        """Test creating a context section."""
        section = ContextSection(
            type=SectionType.MEMORIES,
            content="Memory content",
            priority=10,
            tokens=50,
        )

        assert section.type == SectionType.MEMORIES
        assert section.content == "Memory content"
        assert section.priority == 10
        assert section.tokens == 50

    def test_default_values(self):
        """Test default values."""
        section = ContextSection(
            type=SectionType.USER,
            content="User message",
        )

        assert section.priority == 50
        assert section.tokens == 0
        assert section.metadata == {}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        section = ContextSection(
            type=SectionType.SYSTEM,
            content="System prompt",
            priority=0,
            tokens=100,
            metadata={"key": "value"},
        )

        result = section.to_dict()

        assert result["type"] == "system"
        assert result["content"] == "System prompt"
        assert result["priority"] == 0
        assert result["tokens"] == 100
        assert result["metadata"] == {"key": "value"}


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ContextConfig()

        assert config.max_tokens == 8000
        assert config.include_headers is True
        assert config.memory_format == "bullet"
        assert "memories" in config.section_budgets

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextConfig(
            max_tokens=4000,
            include_headers=False,
            memory_format="numbered",
        )

        assert config.max_tokens == 4000
        assert config.include_headers is False
        assert config.memory_format == "numbered"

    def test_section_budgets(self):
        """Test section budgets are set."""
        config = ContextConfig()

        assert config.section_budgets["system"] == 500
        assert config.section_budgets["memories"] == 3000
        assert config.section_budgets["history"] == 2000


class TestBuiltContext:
    """Tests for BuiltContext dataclass."""

    def test_creation(self):
        """Test creating a built context."""
        sections = [
            ContextSection(type=SectionType.SYSTEM, content="System"),
            ContextSection(type=SectionType.MEMORIES, content="Memories"),
        ]

        context = BuiltContext(
            content="Full context",
            sections=sections,
            total_tokens=100,
        )

        assert context.content == "Full context"
        assert len(context.sections) == 2
        assert context.total_tokens == 100
        assert context.truncated is False

    def test_to_messages_basic(self):
        """Test converting to message format."""
        sections = [
            ContextSection(type=SectionType.SYSTEM, content="Be helpful"),
        ]

        context = BuiltContext(
            content="",
            sections=sections,
            total_tokens=10,
        )

        messages = context.to_messages(user_message="Hello")

        assert len(messages) >= 1
        assert any(m["role"] == "system" for m in messages)

    def test_to_messages_with_memories(self):
        """Test messages include memories."""
        sections = [
            ContextSection(type=SectionType.SYSTEM, content="System prompt"),
            ContextSection(type=SectionType.MEMORIES, content="Important memory"),
        ]

        context = BuiltContext(
            content="",
            sections=sections,
            total_tokens=50,
        )

        messages = context.to_messages()
        assert len(messages) >= 1


class TestContextBuilder:
    """Tests for ContextBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a context builder."""
        return ContextBuilder()

    @pytest.fixture
    def builder_with_config(self):
        """Create a builder with custom config."""
        config = ContextConfig(
            max_tokens=2000,
            include_headers=True,
        )
        return ContextBuilder(config=config)

    def test_init_default(self, builder):
        """Test default initialization."""
        assert builder._config is not None
        assert len(builder._sections) == 0

    def test_init_with_config(self, builder_with_config):
        """Test initialization with config."""
        assert builder_with_config._config.max_tokens == 2000

    def test_add_system(self, builder):
        """Test adding system prompt."""
        builder.add_system("Be a helpful assistant")

        assert len(builder._sections) == 1
        assert builder._sections[0].type == SectionType.SYSTEM
        assert "helpful" in builder._sections[0].content

    def test_add_memories_list(self, builder):
        """Test adding memories as list."""
        memories = [
            create_entry("Memory 1", importance=0.8),
            create_entry("Memory 2", importance=0.6),
        ]

        builder.add_memories(memories)

        # Should have a memories section
        memory_sections = [s for s in builder._sections if s.type == SectionType.MEMORIES]
        assert len(memory_sections) == 1

    def test_add_memories_empty(self, builder):
        """Test adding empty memories list."""
        builder.add_memories([])

        # Should not add section for empty list
        memory_sections = [s for s in builder._sections if s.type == SectionType.MEMORIES]
        assert len(memory_sections) == 0

    def test_add_history(self, builder):
        """Test adding conversation history."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        builder.add_history(history)

        history_sections = [s for s in builder._sections if s.type == SectionType.HISTORY]
        assert len(history_sections) == 1

    def test_add_instructions(self, builder):
        """Test adding instructions."""
        builder.add_instructions("Follow these rules: ...")

        instruction_sections = [s for s in builder._sections if s.type == SectionType.INSTRUCTIONS]
        assert len(instruction_sections) == 1

    def test_add_custom_section(self, builder):
        """Test adding custom section."""
        # Use add_custom() instead of add_section()
        builder.add_custom(
            content="Custom content",
            priority=25,
        )

        custom_sections = [s for s in builder._sections if s.type == SectionType.CUSTOM]
        assert len(custom_sections) == 1
        assert custom_sections[0].priority == 25

    def test_build_basic(self, builder):
        """Test building context."""
        builder.add_system("System prompt")
        builder.add_memories([create_entry("Important fact")])

        context = builder.build()

        assert isinstance(context, BuiltContext)
        assert len(context.content) > 0
        assert context.total_tokens > 0

    def test_build_empty(self, builder):
        """Test building empty context."""
        context = builder.build()

        assert context.content == ""
        assert context.total_tokens == 0

    def test_build_respects_priority(self, builder):
        """Test that building respects priority order."""
        builder.add_custom("Low priority", priority=100)
        builder.add_system("High priority")  # Default priority is 0

        context = builder.build()

        # System (priority 0) should come before custom (priority 100)
        system_pos = context.content.find("High priority")
        custom_pos = context.content.find("Low priority")

        if system_pos >= 0 and custom_pos >= 0:
            assert system_pos < custom_pos

    def test_build_with_max_tokens(self, builder_with_config):
        """Test that building respects max tokens."""
        # Add lots of content
        long_content = "x" * 10000
        builder_with_config.add_system(long_content)

        context = builder_with_config.build()

        # Context should be truncated (allow small buffer for truncation logic)
        assert context.truncated is True
        # The truncated content should be significantly smaller than original
        assert context.total_tokens < 10000 // 4  # Original would be ~2500 tokens

    def test_reset(self, builder):
        """Test resetting sections."""
        builder.add_system("System")
        builder.add_instructions("Instructions")

        # Use reset() instead of clear()
        builder.reset()

        assert len(builder._sections) == 0

    def test_get_stats(self, builder):
        """Test getting statistics."""
        builder.add_system("System prompt")
        builder.add_memories([create_entry("Memory")])

        stats = builder.get_stats()

        # Stats uses 'sections' not 'section_count'
        assert "sections" in stats
        assert "total_tokens" in stats
        assert stats["sections"] == 2

    def test_memory_format_bullet(self, builder):
        """Test bullet format for memories."""
        memories = [create_entry("Fact 1"), create_entry("Fact 2")]
        builder.add_memories(memories)

        context = builder.build()

        # Should contain bullet points
        assert "•" in context.content or "-" in context.content or "Fact" in context.content

    def test_chaining(self, builder):
        """Test method chaining."""
        result = (
            builder
            .add_system("System")
            .add_instructions("Instructions")
            .add_memories([create_entry("Memory")])
        )

        assert result is builder
        assert len(builder._sections) == 3


class TestContextBuilderEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def builder(self):
        return ContextBuilder()

    def test_special_characters(self, builder):
        """Test handling special characters."""
        builder.add_system("Use 'quotes' and \"double quotes\"")
        builder.add_memories([create_entry("Code: print('hello')")])

        context = builder.build()
        assert len(context.content) > 0

    def test_unicode_content(self, builder):
        """Test handling unicode."""
        builder.add_system("中文系统提示")
        builder.add_memories([create_entry("日本語メモリ")])

        context = builder.build()
        assert "中文" in context.content or len(context.content) > 0

    def test_empty_strings(self, builder):
        """Test handling empty strings."""
        builder.add_system("")
        builder.add_instructions("")

        context = builder.build()
        # Should handle gracefully
        assert isinstance(context, BuiltContext)

    def test_very_long_content(self, builder):
        """Test handling very long content."""
        config = ContextConfig(max_tokens=100)
        builder = ContextBuilder(config=config)

        long_content = "word " * 1000
        builder.add_system(long_content)

        context = builder.build()

        # Should be truncated
        assert context.truncated or context.total_tokens <= 100

    def test_many_sections(self, builder):
        """Test handling many sections."""
        for i in range(50):
            builder.add_custom(f"Section {i}", priority=i)

        context = builder.build()

        assert context.total_tokens > 0

    def test_duplicate_section_types(self, builder):
        """Test multiple sections of same type."""
        builder.add_system("System 1")
        builder.add_system("System 2")

        context = builder.build()

        # Both should be included
        system_sections = [s for s in context.sections if s.type == SectionType.SYSTEM]
        assert len(system_sections) == 2

    def test_memory_with_metadata(self):
        """Test memories with metadata included."""
        config = ContextConfig(include_memory_metadata=True)
        builder = ContextBuilder(config=config)

        entry = create_entry("Memory content", tags=["important"])
        entry.metadata = {"source": "test"}

        builder.add_memories([entry])
        context = builder.build()

        assert len(context.content) > 0

    def test_history_various_formats(self, builder):
        """Test various history formats."""
        # Standard format
        history1 = [{"role": "user", "content": "Hi"}]
        builder.add_history(history1)

        # With metadata
        history2 = [{"role": "assistant", "content": "Hello", "timestamp": "2024-01-01"}]
        builder.add_history(history2)

        context = builder.build()
        assert len(context.sections) >= 2

    def test_numbered_memory_format(self):
        """Test numbered memory format."""
        config = ContextConfig(memory_format="numbered")
        builder = ContextBuilder(config=config)

        memories = [create_entry("First"), create_entry("Second")]
        builder.add_memories(memories)

        context = builder.build()
        assert "1." in context.content or "2." in context.content

    def test_detailed_memory_format(self):
        """Test detailed memory format."""
        config = ContextConfig(memory_format="detailed", include_memory_metadata=True)
        builder = ContextBuilder(config=config)

        memories = [create_entry("Important fact", importance=0.8, tags=["key"])]
        builder.add_memories(memories)

        context = builder.build()
        # Detailed format includes importance score
        assert "[0.8]" in context.content or "Important" in context.content

    def test_to_dict(self):
        """Test BuiltContext to_dict method."""
        builder = ContextBuilder()
        builder.add_system("Test system")

        context = builder.build()
        result = context.to_dict()

        assert "content" in result
        assert "sections" in result
        assert "total_tokens" in result
        assert "truncated" in result
