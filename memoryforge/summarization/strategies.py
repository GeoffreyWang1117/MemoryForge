"""Summarization strategies for different content types."""

import re
from collections import Counter
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


class ExtractiveSummarizer:
    """Extractive summarization using sentence selection.

    Selects the most important sentences based on:
    - TF-IDF scoring
    - Position heuristics
    - Sentence similarity
    """

    def __init__(
        self,
        min_sentence_length: int = 10,
        max_sentence_length: int = 300,
    ):
        """Initialize extractive summarizer.

        Args:
            min_sentence_length: Minimum sentence length to consider
            max_sentence_length: Maximum sentence length to include
        """
        self._min_length = min_sentence_length
        self._max_length = max_sentence_length
        self._stop_words = self._get_stop_words()

    def summarize(
        self,
        content: str,
        max_length: int,
    ) -> tuple[str, list[str]]:
        """Create extractive summary.

        Args:
            content: Text to summarize
            max_length: Maximum summary length

        Returns:
            Tuple of (summary, key_points)
        """
        sentences = self._split_sentences(content)

        if not sentences:
            return content[:max_length], []

        # Filter by length
        valid_sentences = [
            s for s in sentences
            if self._min_length <= len(s) <= self._max_length
        ]

        if not valid_sentences:
            valid_sentences = sentences[:5]

        # Calculate word frequencies
        word_freq = self._calculate_word_frequencies(content)

        # Score sentences
        scored = []
        for i, sentence in enumerate(valid_sentences):
            score = self._score_sentence(sentence, i, len(valid_sentences), word_freq)
            scored.append((sentence, score, i))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select sentences
        selected = []
        current_length = 0
        used_indices = set()

        for sentence, score, idx in scored:
            if current_length + len(sentence) + 1 > max_length:
                continue
            if idx in used_indices:
                continue

            selected.append((sentence, idx))
            used_indices.add(idx)
            current_length += len(sentence) + 1

        # Sort by original position
        selected.sort(key=lambda x: x[1])

        summary = " ".join(s for s, _ in selected)
        key_points = [s for s, _, _ in scored[:5]]

        return summary, key_points

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.\s', r'\1__DOT__ ', text)
        text = re.sub(r'\b(i\.e|e\.g|etc)\.\s', r'\1__DOT__ ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Restore abbreviations
        sentences = [s.replace('__DOT__', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _calculate_word_frequencies(self, text: str) -> dict[str, float]:
        """Calculate normalized word frequencies."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self._stop_words]

        if not words:
            return {}

        counts = Counter(words)
        max_freq = max(counts.values())

        return {word: count / max_freq for word, count in counts.items()}

    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total: int,
        word_freq: dict[str, float],
    ) -> float:
        """Score sentence importance."""
        score = 0.0

        # Word frequency score
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        if words:
            freq_score = sum(word_freq.get(w, 0) for w in words) / len(words)
            score += freq_score * 0.5

        # Position score
        if position == 0:
            score += 0.25
        elif position < total * 0.2:
            score += 0.15
        elif position == total - 1:
            score += 0.1

        # Length preference (medium sentences)
        length = len(sentence)
        if 50 <= length <= 150:
            score += 0.15
        elif 30 <= length <= 200:
            score += 0.1

        # Named entity boost
        if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', sentence):
            score += 0.1

        return score

    def _get_stop_words(self) -> set[str]:
        """Get stop words set."""
        return {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "it", "its", "they", "them",
            "their", "what", "which", "who", "whom", "how", "when", "where",
            "why", "not", "no", "yes", "all", "any", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "than", "too",
            "very", "just", "also", "only", "own", "same", "so", "then",
        }


class KeyPointSummarizer:
    """Summarization focused on extracting key points.

    Identifies and extracts:
    - Main topics
    - Important facts
    - Action items
    - Conclusions
    """

    def __init__(self):
        """Initialize key point summarizer."""
        self._importance_markers = [
            r'\b(important|key|main|critical|essential|crucial)\b',
            r'\b(must|should|need to|have to)\b',
            r'\b(first|second|finally|in conclusion)\b',
            r'\b(result|conclusion|summary|therefore)\b',
        ]

    def summarize(
        self,
        content: str,
        max_length: int,
    ) -> tuple[str, list[str]]:
        """Extract key points as summary.

        Args:
            content: Text to summarize
            max_length: Maximum summary length

        Returns:
            Tuple of (summary, key_points)
        """
        sentences = self._split_sentences(content)

        if not sentences:
            return content[:max_length], []

        # Score sentences for key point potential
        scored = []
        for sentence in sentences:
            score = self._score_key_point(sentence)
            if score > 0:
                scored.append((sentence, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build key points list
        key_points = []
        summary_parts = []
        current_length = 0

        for sentence, score in scored:
            # Add to key points
            key_points.append(sentence)

            # Add to summary if fits
            if current_length + len(sentence) + 2 <= max_length:
                summary_parts.append(sentence)
                current_length += len(sentence) + 2

        # Build summary
        if summary_parts:
            summary = " ".join(summary_parts)
        else:
            # Fall back to first sentences
            summary = " ".join(sentences[:3])[:max_length]

        return summary, key_points[:10]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _score_key_point(self, sentence: str) -> float:
        """Score sentence as potential key point."""
        score = 0.0
        sentence_lower = sentence.lower()

        # Check importance markers
        for pattern in self._importance_markers:
            if re.search(pattern, sentence_lower):
                score += 0.3

        # Numbers often indicate specific facts
        if re.search(r'\b\d+(\.\d+)?(%|x|times)?\b', sentence):
            score += 0.2

        # Lists or enumerations
        if re.search(r'^\s*[\d•\-\*]\s*', sentence):
            score += 0.25

        # Definitions or explanations
        if re.search(r'\b(is defined as|means|refers to)\b', sentence_lower):
            score += 0.25

        # Action items
        if re.search(r'\b(todo|action|implement|create|update|fix)\b', sentence_lower):
            score += 0.3

        # Conclusions
        if re.search(r'\b(therefore|thus|hence|as a result|in conclusion)\b', sentence_lower):
            score += 0.35

        return score


class ConversationSummarizer:
    """Summarization specialized for conversations.

    Handles:
    - Multi-party dialogues
    - Topic tracking
    - Decision extraction
    - Action item identification
    """

    def __init__(self):
        """Initialize conversation summarizer."""
        self._role_patterns = [
            (r'^(user|human|customer):\s*', 'user'),
            (r'^(assistant|ai|bot|agent):\s*', 'assistant'),
            (r'^(system):\s*', 'system'),
            (r'^(\w+):\s*', 'participant'),
        ]

    def summarize(
        self,
        content: str,
        max_length: int,
    ) -> tuple[str, list[str]]:
        """Summarize conversation content.

        Args:
            content: Conversation text
            max_length: Maximum summary length

        Returns:
            Tuple of (summary, key_points)
        """
        # Parse conversation
        messages = self._parse_messages(content)

        if not messages:
            return content[:max_length], []

        # Extract topics discussed
        topics = self._extract_topics(messages)

        # Extract decisions and action items
        decisions = self._extract_decisions(messages)
        actions = self._extract_actions(messages)

        # Build summary
        summary_parts = []

        # Topic overview
        if topics:
            topic_str = ", ".join(topics[:5])
            summary_parts.append(f"Topics discussed: {topic_str}.")

        # Message count
        summary_parts.append(f"Conversation with {len(messages)} exchanges.")

        # Key decisions
        if decisions:
            summary_parts.append(f"Decisions: {decisions[0]}")

        # Action items
        if actions:
            action_str = "; ".join(actions[:3])
            summary_parts.append(f"Actions: {action_str}")

        summary = " ".join(summary_parts)

        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."

        # Build key points
        key_points = []
        key_points.extend([f"Topic: {t}" for t in topics[:3]])
        key_points.extend([f"Decision: {d}" for d in decisions[:2]])
        key_points.extend([f"Action: {a}" for a in actions[:3]])

        return summary, key_points

    def _parse_messages(self, content: str) -> list[dict]:
        """Parse conversation into messages."""
        messages = []
        lines = content.split('\n')

        current_role = None
        current_content = []

        for line in lines:
            # Check for role prefix
            matched = False
            for pattern, role in self._role_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous message
                    if current_role and current_content:
                        messages.append({
                            'role': current_role,
                            'content': ' '.join(current_content),
                        })

                    current_role = role
                    current_content = [line[match.end():].strip()]
                    matched = True
                    break

            if not matched and current_role:
                current_content.append(line.strip())

        # Save last message
        if current_role and current_content:
            messages.append({
                'role': current_role,
                'content': ' '.join(current_content),
            })

        return messages

    def _extract_topics(self, messages: list[dict]) -> list[str]:
        """Extract topics from conversation."""
        all_text = " ".join(m['content'] for m in messages)

        # Simple topic extraction via keyword frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        stop_words = {
            "that", "this", "with", "have", "will", "would", "could",
            "should", "there", "their", "about", "which", "where",
            "what", "when", "from", "they", "been", "were", "being",
        }
        words = [w for w in words if w not in stop_words]

        counts = Counter(words)
        topics = [word for word, _ in counts.most_common(10)]

        return topics

    def _extract_decisions(self, messages: list[dict]) -> list[str]:
        """Extract decisions from conversation."""
        decisions = []
        decision_patterns = [
            r"(?:we )?(?:decided|agreed|concluded) (?:to |that )(.+?)(?:\.|$)",
            r"(?:the )?decision is (?:to )?(.+?)(?:\.|$)",
            r"(?:we )?(?:will|should) (.+?)(?:\.|$)",
        ]

        for msg in messages:
            content = msg['content'].lower()
            for pattern in decision_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                decisions.extend(matches)

        return [d.strip() for d in decisions if len(d.strip()) > 10][:5]

    def _extract_actions(self, messages: list[dict]) -> list[str]:
        """Extract action items from conversation."""
        actions = []
        action_patterns = [
            r"(?:todo|action|task):\s*(.+?)(?:\.|$)",
            r"(?:need to|should|must|will) (.+?)(?:\.|$)",
            r"(?:please |can you )(.+?)(?:\.|$|\?)",
        ]

        for msg in messages:
            content = msg['content']
            for pattern in action_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                actions.extend(matches)

        return [a.strip() for a in actions if len(a.strip()) > 5][:10]


@dataclass
class HybridSummarizer:
    """Combines multiple summarization strategies.

    Uses:
    - Extractive for factual content
    - Key points for technical content
    - Conversation for dialogue
    """

    extractive: ExtractiveSummarizer = field(default_factory=ExtractiveSummarizer)
    key_point: KeyPointSummarizer = field(default_factory=KeyPointSummarizer)
    conversation: ConversationSummarizer = field(default_factory=ConversationSummarizer)

    def summarize(
        self,
        content: str,
        max_length: int,
    ) -> tuple[str, list[str]]:
        """Automatically select and apply appropriate strategy.

        Args:
            content: Text to summarize
            max_length: Maximum summary length

        Returns:
            Tuple of (summary, key_points)
        """
        content_type = self._detect_content_type(content)

        if content_type == "conversation":
            return self.conversation.summarize(content, max_length)
        elif content_type == "technical":
            return self.key_point.summarize(content, max_length)
        else:
            return self.extractive.summarize(content, max_length)

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content."""
        # Check for conversation markers
        if re.search(r'^(user|assistant|human|ai):\s', content, re.MULTILINE | re.IGNORECASE):
            return "conversation"

        # Check for technical content
        technical_markers = [
            r'```',  # Code blocks
            r'\bdef\s+\w+\(',  # Python functions
            r'\bfunction\s+\w+\(',  # JS functions
            r'\bclass\s+\w+',  # Classes
            r'\b(api|endpoint|database|query)\b',  # Technical terms
        ]
        tech_count = sum(1 for p in technical_markers if re.search(p, content, re.IGNORECASE))
        if tech_count >= 2:
            return "technical"

        return "general"
