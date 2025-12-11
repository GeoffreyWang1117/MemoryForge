"""Memory summarization service."""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Protocol

import structlog

from memoryforge.core.types import MemoryEntry, MemoryLayer

logger = structlog.get_logger()


class SummaryLevel(Enum):
    """Summary length levels."""

    BRIEF = "brief"  # 1-2 sentences
    SHORT = "short"  # 2-3 sentences
    MEDIUM = "medium"  # 1 paragraph
    DETAILED = "detailed"  # Multiple paragraphs


@dataclass
class SummaryConfig:
    """Configuration for summarization."""

    # Default summary level
    default_level: SummaryLevel = SummaryLevel.SHORT

    # Maximum length per level (in characters)
    max_lengths: dict[SummaryLevel, int] = field(default_factory=lambda: {
        SummaryLevel.BRIEF: 100,
        SummaryLevel.SHORT: 200,
        SummaryLevel.MEDIUM: 500,
        SummaryLevel.DETAILED: 1000,
    })

    # Minimum content length to summarize
    min_content_length: int = 100

    # Include metadata in summary
    include_metadata: bool = False

    # Sentence ending markers
    sentence_endings: str = ".!?"


@dataclass
class SummaryResult:
    """Result of a summarization operation."""

    original_length: int
    summary_length: int
    summary: str
    level: SummaryLevel
    compression_ratio: float
    key_points: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "summary": self.summary,
            "level": self.level.value,
            "compression_ratio": round(self.compression_ratio, 3),
            "key_points": self.key_points,
            "topics": self.topics,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class SummarizationStrategy(Protocol):
    """Protocol for summarization strategies."""

    def summarize(
        self,
        content: str,
        max_length: int,
    ) -> tuple[str, list[str]]:
        """Summarize content.

        Args:
            content: Text to summarize
            max_length: Maximum summary length

        Returns:
            Tuple of (summary, key_points)
        """
        ...


class MemorySummarizer:
    """Service for summarizing memory content.

    Provides:
    - Multiple summary levels
    - Extractive summarization
    - Key point extraction
    - Conversation summarization
    - Batch processing
    """

    def __init__(
        self,
        config: SummaryConfig | None = None,
        strategy: SummarizationStrategy | None = None,
    ):
        """Initialize summarizer.

        Args:
            config: Summarization configuration
            strategy: Summarization strategy to use
        """
        self._config = config or SummaryConfig()
        self._strategy = strategy or DefaultSummarizer()

        # Statistics
        self._total_summarized = 0
        self._total_chars_processed = 0
        self._total_chars_output = 0

    def summarize_entry(
        self,
        entry: MemoryEntry,
        level: SummaryLevel | None = None,
    ) -> SummaryResult:
        """Summarize a memory entry.

        Args:
            entry: Memory entry to summarize
            level: Summary level (uses default if not specified)

        Returns:
            Summary result
        """
        start_time = datetime.now(timezone.utc)
        level = level or self._config.default_level
        max_length = self._config.max_lengths[level]

        content = entry.content
        original_length = len(content)

        # Skip if content is already short enough
        if original_length <= max_length:
            return SummaryResult(
                original_length=original_length,
                summary_length=original_length,
                summary=content,
                level=level,
                compression_ratio=1.0,
                topics=entry.tags[:5],
            )

        # Apply summarization strategy
        summary, key_points = self._strategy.summarize(content, max_length)

        # Ensure summary fits within max length
        if len(summary) > max_length:
            summary = self._truncate_at_sentence(summary, max_length)

        # Calculate metrics
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Update statistics
        self._total_summarized += 1
        self._total_chars_processed += original_length
        self._total_chars_output += len(summary)

        result = SummaryResult(
            original_length=original_length,
            summary_length=len(summary),
            summary=summary,
            level=level,
            compression_ratio=len(summary) / original_length if original_length else 1.0,
            key_points=key_points[:5],
            topics=entry.tags[:5],
            processing_time_ms=elapsed,
        )

        logger.debug(
            "Entry summarized",
            original=original_length,
            summary=len(summary),
            level=level.value,
        )

        return result

    def summarize_content(
        self,
        content: str,
        level: SummaryLevel | None = None,
    ) -> SummaryResult:
        """Summarize raw content.

        Args:
            content: Text content to summarize
            level: Summary level

        Returns:
            Summary result
        """
        start_time = datetime.now(timezone.utc)
        level = level or self._config.default_level
        max_length = self._config.max_lengths[level]

        original_length = len(content)

        if original_length <= max_length:
            return SummaryResult(
                original_length=original_length,
                summary_length=original_length,
                summary=content,
                level=level,
                compression_ratio=1.0,
            )

        summary, key_points = self._strategy.summarize(content, max_length)

        if len(summary) > max_length:
            summary = self._truncate_at_sentence(summary, max_length)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return SummaryResult(
            original_length=original_length,
            summary_length=len(summary),
            summary=summary,
            level=level,
            compression_ratio=len(summary) / original_length if original_length else 1.0,
            key_points=key_points[:5],
            processing_time_ms=elapsed,
        )

    def summarize_batch(
        self,
        entries: list[MemoryEntry],
        level: SummaryLevel | None = None,
    ) -> list[SummaryResult]:
        """Summarize multiple entries.

        Args:
            entries: Entries to summarize
            level: Summary level

        Returns:
            List of summary results
        """
        return [self.summarize_entry(entry, level) for entry in entries]

    def summarize_conversation(
        self,
        messages: list[dict],
        level: SummaryLevel | None = None,
    ) -> SummaryResult:
        """Summarize a conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            level: Summary level

        Returns:
            Summary result
        """
        # Combine messages into content
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")

        combined = "\n".join(parts)
        return self.summarize_content(combined, level)

    def summarize_group(
        self,
        entries: list[MemoryEntry],
        level: SummaryLevel | None = None,
    ) -> SummaryResult:
        """Summarize a group of entries into a single summary.

        Args:
            entries: Entries to combine and summarize
            level: Summary level

        Returns:
            Combined summary
        """
        # Combine all content
        contents = [entry.content for entry in entries]
        combined = "\n\n".join(contents)

        # Collect all topics
        all_topics = []
        for entry in entries:
            all_topics.extend(entry.tags)
        unique_topics = list(dict.fromkeys(all_topics))[:10]

        result = self.summarize_content(combined, level)
        result.topics = unique_topics

        return result

    def _truncate_at_sentence(self, text: str, max_length: int) -> str:
        """Truncate text at sentence boundary.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # Find last sentence ending before max_length
        truncated = text[:max_length]
        last_ending = -1

        for char in self._config.sentence_endings:
            pos = truncated.rfind(char)
            if pos > last_ending:
                last_ending = pos

        if last_ending > max_length // 2:
            return truncated[:last_ending + 1].strip()

        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            return truncated[:last_space].strip() + "..."

        return truncated.strip() + "..."

    def get_stats(self) -> dict:
        """Get summarizer statistics."""
        return {
            "total_summarized": self._total_summarized,
            "total_chars_processed": self._total_chars_processed,
            "total_chars_output": self._total_chars_output,
            "overall_compression": (
                self._total_chars_output / self._total_chars_processed
                if self._total_chars_processed else 1.0
            ),
            "config": {
                "default_level": self._config.default_level.value,
                "max_lengths": {
                    k.value: v for k, v in self._config.max_lengths.items()
                },
            },
        }


class DefaultSummarizer:
    """Default extractive summarization strategy.

    Uses sentence scoring based on:
    - Position in text
    - Keyword density
    - Sentence length
    """

    def __init__(self):
        self._stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "it", "its", "they", "them",
        }

    def summarize(
        self,
        content: str,
        max_length: int,
    ) -> tuple[str, list[str]]:
        """Summarize using extractive method.

        Args:
            content: Text to summarize
            max_length: Maximum summary length

        Returns:
            Tuple of (summary, key_points)
        """
        # Split into sentences
        sentences = self._split_sentences(content)

        if not sentences:
            return content[:max_length], []

        if len(sentences) == 1:
            return sentences[0][:max_length], []

        # Extract keywords
        keywords = self._extract_keywords(content)

        # Score sentences
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences), keywords)
            scored.append((sentence, score, i))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences until max_length
        selected = []
        current_length = 0

        for sentence, score, orig_idx in scored:
            if current_length + len(sentence) + 1 <= max_length:
                selected.append((sentence, orig_idx))
                current_length += len(sentence) + 1
            else:
                break

        # Sort by original position for coherence
        selected.sort(key=lambda x: x[1])

        # Build summary
        summary = " ".join(s for s, _ in selected)

        # Extract key points (top 3-5 scored sentences)
        key_points = [s for s, _, _ in scored[:5]]

        return summary, key_points

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_keywords(self, text: str, top_n: int = 10) -> set[str]:
        """Extract keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self._stop_words]

        word_counts = Counter(words)
        return {w for w, _ in word_counts.most_common(top_n)}

    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total: int,
        keywords: set[str],
    ) -> float:
        """Score a sentence for importance."""
        score = 0.0

        # Position score (first and last sentences more important)
        if position == 0:
            score += 0.3
        elif position == total - 1:
            score += 0.2
        elif position < total * 0.2:
            score += 0.15

        # Keyword score
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()))
        keyword_overlap = len(words & keywords)
        score += min(0.4, keyword_overlap * 0.1)

        # Length score (prefer medium-length sentences)
        length = len(sentence)
        if 30 <= length <= 150:
            score += 0.2
        elif 20 <= length <= 200:
            score += 0.1

        # Indicator phrases
        indicators = ["important", "key", "main", "significant", "essential", "crucial"]
        for indicator in indicators:
            if indicator in sentence.lower():
                score += 0.15
                break

        return score


def create_summarizer(
    level: SummaryLevel = SummaryLevel.SHORT,
    strategy: SummarizationStrategy | None = None,
) -> MemorySummarizer:
    """Create a summarizer with specified configuration.

    Args:
        level: Default summary level
        strategy: Summarization strategy

    Returns:
        Configured MemorySummarizer
    """
    config = SummaryConfig(default_level=level)
    return MemorySummarizer(config, strategy)
