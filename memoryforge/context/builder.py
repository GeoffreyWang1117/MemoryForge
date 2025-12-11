"""Context builder for constructing LLM prompts with memory integration."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

import structlog

from memoryforge.core.types import MemoryEntry, MemoryQuery

logger = structlog.get_logger()


class SectionType(Enum):
    """Types of context sections."""

    SYSTEM = "system"
    MEMORIES = "memories"
    HISTORY = "history"
    INSTRUCTIONS = "instructions"
    USER = "user"
    CUSTOM = "custom"


@dataclass
class ContextSection:
    """A section of the context."""

    type: SectionType
    content: str
    priority: int = 50  # 0 = highest priority
    tokens: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "content": self.content,
            "priority": self.priority,
            "tokens": self.tokens,
            "metadata": self.metadata,
        }


@dataclass
class ContextConfig:
    """Configuration for context building."""

    # Maximum total tokens
    max_tokens: int = 8000

    # Token budget per section type
    section_budgets: dict[str, int] = field(default_factory=lambda: {
        "system": 500,
        "memories": 3000,
        "history": 2000,
        "instructions": 500,
        "user": 1000,
        "custom": 1000,
    })

    # Include section headers
    include_headers: bool = True

    # Memory formatting
    memory_format: str = "bullet"  # bullet, numbered, detailed

    # Include metadata in memories
    include_memory_metadata: bool = False

    # Separator between sections
    section_separator: str = "\n\n"

    # Memory separator
    memory_separator: str = "\n"


@dataclass
class BuiltContext:
    """A built context ready for LLM consumption."""

    content: str
    sections: list[ContextSection]
    total_tokens: int
    truncated: bool = False
    metadata: dict = field(default_factory=dict)

    def to_messages(self, user_message: str = "") -> list[dict]:
        """Convert to chat message format.

        Args:
            user_message: Optional user message to append

        Returns:
            List of message dictionaries
        """
        messages = []

        # System message
        system_content = []
        for section in self.sections:
            if section.type == SectionType.SYSTEM:
                system_content.append(section.content)

        if system_content:
            messages.append({
                "role": "system",
                "content": "\n\n".join(system_content),
            })

        # Assistant context (memories, history)
        context_parts = []
        for section in self.sections:
            if section.type in (SectionType.MEMORIES, SectionType.HISTORY):
                context_parts.append(section.content)

        if context_parts:
            messages.append({
                "role": "assistant",
                "content": "\n\n".join(context_parts),
            })

        # User message
        if user_message:
            messages.append({
                "role": "user",
                "content": user_message,
            })

        return messages

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "sections": [s.to_dict() for s in self.sections],
            "total_tokens": self.total_tokens,
            "truncated": self.truncated,
            "metadata": self.metadata,
        }


class ContextBuilder:
    """Builder for constructing LLM context with memory integration.

    Provides:
    - Token-aware context construction
    - Priority-based section ordering
    - Memory formatting options
    - Automatic truncation
    """

    def __init__(
        self,
        config: ContextConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
    ):
        """Initialize the context builder.

        Args:
            config: Context configuration
            token_counter: Function to count tokens (defaults to char/4)
        """
        self._config = config or ContextConfig()
        self._token_counter = token_counter or self._default_token_counter
        self._sections: list[ContextSection] = []

    def _default_token_counter(self, text: str) -> int:
        """Default token counter (rough estimate)."""
        return len(text) // 4

    def reset(self) -> "ContextBuilder":
        """Reset the builder for a new context."""
        self._sections.clear()
        return self

    def add_system(
        self,
        content: str,
        priority: int = 0,
    ) -> "ContextBuilder":
        """Add system instructions.

        Args:
            content: System prompt content
            priority: Priority (lower = higher priority)

        Returns:
            Self for chaining
        """
        self._sections.append(ContextSection(
            type=SectionType.SYSTEM,
            content=content,
            priority=priority,
            tokens=self._token_counter(content),
        ))
        return self

    def add_memories(
        self,
        memories: list[MemoryEntry],
        header: str = "Relevant Memories:",
        priority: int = 30,
    ) -> "ContextBuilder":
        """Add memory entries to context.

        Args:
            memories: Memory entries to include
            header: Section header
            priority: Priority level

        Returns:
            Self for chaining
        """
        if not memories:
            return self

        formatted = self._format_memories(memories)

        content = formatted
        if self._config.include_headers and header:
            content = f"{header}\n{formatted}"

        self._sections.append(ContextSection(
            type=SectionType.MEMORIES,
            content=content,
            priority=priority,
            tokens=self._token_counter(content),
            metadata={"count": len(memories)},
        ))
        return self

    def add_history(
        self,
        history: list[dict],
        header: str = "Conversation History:",
        priority: int = 40,
    ) -> "ContextBuilder":
        """Add conversation history.

        Args:
            history: List of message dicts with 'role' and 'content'
            header: Section header
            priority: Priority level

        Returns:
            Self for chaining
        """
        if not history:
            return self

        formatted_parts = []
        for msg in history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            formatted_parts.append(f"{role}: {content}")

        formatted = "\n".join(formatted_parts)

        content = formatted
        if self._config.include_headers and header:
            content = f"{header}\n{formatted}"

        self._sections.append(ContextSection(
            type=SectionType.HISTORY,
            content=content,
            priority=priority,
            tokens=self._token_counter(content),
            metadata={"turns": len(history)},
        ))
        return self

    def add_instructions(
        self,
        content: str,
        priority: int = 10,
    ) -> "ContextBuilder":
        """Add task-specific instructions.

        Args:
            content: Instructions text
            priority: Priority level

        Returns:
            Self for chaining
        """
        self._sections.append(ContextSection(
            type=SectionType.INSTRUCTIONS,
            content=content,
            priority=priority,
            tokens=self._token_counter(content),
        ))
        return self

    def add_custom(
        self,
        content: str,
        section_type: str = "context",
        priority: int = 50,
    ) -> "ContextBuilder":
        """Add custom section.

        Args:
            content: Section content
            section_type: Type identifier
            priority: Priority level

        Returns:
            Self for chaining
        """
        self._sections.append(ContextSection(
            type=SectionType.CUSTOM,
            content=content,
            priority=priority,
            tokens=self._token_counter(content),
            metadata={"custom_type": section_type},
        ))
        return self

    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format memories based on configuration."""
        parts = []

        for i, memory in enumerate(memories, 1):
            if self._config.memory_format == "bullet":
                line = f"• {memory.content}"
            elif self._config.memory_format == "numbered":
                line = f"{i}. {memory.content}"
            else:  # detailed
                importance = memory.importance.effective_score
                tags = ", ".join(memory.tags) if memory.tags else "none"
                line = f"[{importance:.1f}] {memory.content}"
                if self._config.include_memory_metadata:
                    line += f" (tags: {tags})"

            parts.append(line)

        return self._config.memory_separator.join(parts)

    def build(self) -> BuiltContext:
        """Build the final context.

        Returns:
            Built context with content and metadata
        """
        if not self._sections:
            return BuiltContext(
                content="",
                sections=[],
                total_tokens=0,
            )

        # Sort by priority
        sorted_sections = sorted(self._sections, key=lambda s: s.priority)

        # Apply token budget
        final_sections = []
        total_tokens = 0
        truncated = False

        for section in sorted_sections:
            budget = self._config.section_budgets.get(
                section.type.value,
                self._config.section_budgets.get("custom", 1000),
            )

            # Check if we can fit this section
            if total_tokens + section.tokens > self._config.max_tokens:
                # Try to fit partial content
                remaining = self._config.max_tokens - total_tokens
                if remaining > 100:  # Minimum useful content
                    truncated_content = self._truncate_to_tokens(
                        section.content,
                        remaining,
                    )
                    if truncated_content:
                        truncated_section = ContextSection(
                            type=section.type,
                            content=truncated_content,
                            priority=section.priority,
                            tokens=self._token_counter(truncated_content),
                            metadata=section.metadata,
                        )
                        final_sections.append(truncated_section)
                        total_tokens += truncated_section.tokens
                        truncated = True
                break

            # Apply section budget
            if section.tokens > budget:
                truncated_content = self._truncate_to_tokens(section.content, budget)
                section = ContextSection(
                    type=section.type,
                    content=truncated_content,
                    priority=section.priority,
                    tokens=self._token_counter(truncated_content),
                    metadata=section.metadata,
                )
                truncated = True

            final_sections.append(section)
            total_tokens += section.tokens

        # Build final content
        content_parts = [s.content for s in final_sections]
        content = self._config.section_separator.join(content_parts)

        return BuiltContext(
            content=content,
            sections=final_sections,
            total_tokens=total_tokens,
            truncated=truncated,
            metadata={
                "section_count": len(final_sections),
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        if self._token_counter(text) <= max_tokens:
            return text

        # Binary search for the right length
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self._token_counter(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        truncated = text[:low].rsplit(" ", 1)[0]  # Don't cut words
        return truncated + "..." if len(truncated) < len(text) else truncated

    def get_stats(self) -> dict:
        """Get builder statistics."""
        return {
            "sections": len(self._sections),
            "total_tokens": sum(s.tokens for s in self._sections),
            "by_type": {
                t.value: sum(
                    s.tokens for s in self._sections if s.type == t
                )
                for t in SectionType
            },
            "max_tokens": self._config.max_tokens,
        }


class PromptTemplate:
    """Template for generating prompts with memory context."""

    def __init__(
        self,
        template: str,
        variables: list[str] | None = None,
    ):
        """Initialize template.

        Args:
            template: Template string with {variable} placeholders
            variables: List of expected variables
        """
        self._template = template
        self._variables = variables or []

    def render(self, **kwargs) -> str:
        """Render the template with provided values.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered template string
        """
        return self._template.format(**kwargs)

    def render_with_memories(
        self,
        memories: list[MemoryEntry],
        memory_format: str = "bullet",
        **kwargs,
    ) -> str:
        """Render template with formatted memories.

        Args:
            memories: Memory entries to format
            memory_format: Format style
            **kwargs: Additional variables

        Returns:
            Rendered template
        """
        formatted_memories = self._format_memories(memories, memory_format)
        return self.render(memories=formatted_memories, **kwargs)

    def _format_memories(
        self,
        memories: list[MemoryEntry],
        format_style: str,
    ) -> str:
        """Format memories for template insertion."""
        if not memories:
            return "No relevant memories."

        parts = []
        for i, memory in enumerate(memories, 1):
            if format_style == "bullet":
                parts.append(f"• {memory.content}")
            elif format_style == "numbered":
                parts.append(f"{i}. {memory.content}")
            else:
                parts.append(memory.content)

        return "\n".join(parts)


# Pre-defined templates
TEMPLATES = {
    "default": PromptTemplate(
        template="""You are a helpful assistant with access to the following memories:

{memories}

Current request: {query}

Please respond based on the context provided.""",
        variables=["memories", "query"],
    ),

    "analysis": PromptTemplate(
        template="""Analyze the following information based on stored memories:

Relevant memories:
{memories}

Query: {query}

Provide a detailed analysis.""",
        variables=["memories", "query"],
    ),

    "summary": PromptTemplate(
        template="""Based on the following memories, provide a concise summary:

{memories}

Focus on: {focus}""",
        variables=["memories", "focus"],
    ),

    "qa": PromptTemplate(
        template="""Answer the question using only the provided context.

Context:
{memories}

Question: {question}

If the answer is not in the context, say "I don't have enough information to answer that."
""",
        variables=["memories", "question"],
    ),
}
