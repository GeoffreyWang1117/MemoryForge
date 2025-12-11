"""Automatic tagging system for memory entries."""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import structlog

from memoryforge.core.types import MemoryEntry, MemoryLayer

logger = structlog.get_logger()


@dataclass
class TaggingConfig:
    """Configuration for auto-tagging."""

    # Maximum tags per memory
    max_tags: int = 10

    # Minimum confidence for tag assignment
    min_confidence: float = 0.3

    # Enable different tagging methods
    enable_keywords: bool = True
    enable_entities: bool = True
    enable_topics: bool = True
    enable_rules: bool = True

    # Keyword extraction settings
    min_keyword_length: int = 3
    max_keyword_length: int = 50
    keyword_top_n: int = 5

    # Common words to exclude
    stop_words: set[str] = field(default_factory=lambda: {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "what", "which", "who", "whom", "how", "when", "where", "why",
        "not", "no", "yes", "all", "any", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "than", "too", "very",
        "just", "also", "only", "own", "same", "so", "then", "there", "here",
        "now", "about", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further", "once",
    })

    # Predefined tag categories
    tag_categories: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class TagRule:
    """Rule for automatic tag assignment."""

    name: str
    pattern: str  # Regex pattern
    tags: list[str]
    priority: int = 0
    case_sensitive: bool = False

    def matches(self, content: str) -> bool:
        """Check if content matches the rule."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return bool(re.search(self.pattern, content, flags))


@dataclass
class TaggingResult:
    """Result of auto-tagging operation."""

    memory_id: str
    original_tags: list[str]
    added_tags: list[str]
    all_tags: list[str]
    confidence_scores: dict[str, float]
    methods_used: list[str]
    processing_time_ms: float

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "original_tags": self.original_tags,
            "added_tags": self.added_tags,
            "all_tags": self.all_tags,
            "confidence_scores": self.confidence_scores,
            "methods_used": self.methods_used,
            "processing_time_ms": self.processing_time_ms,
        }


class AutoTagger:
    """Automatic tagging system for memories.

    Provides:
    - Keyword extraction
    - Entity recognition
    - Topic classification
    - Rule-based tagging
    - Custom tag handlers
    """

    def __init__(self, config: TaggingConfig | None = None):
        """Initialize auto-tagger.

        Args:
            config: Tagging configuration
        """
        self._config = config or TaggingConfig()
        self._rules: list[TagRule] = []
        self._custom_handlers: list[Callable[[MemoryEntry], list[tuple[str, float]]]] = []

        # Statistics
        self._total_tagged = 0
        self._tags_added = 0

    def add_rule(self, rule: TagRule) -> None:
        """Add a tagging rule.

        Args:
            rule: Rule to add
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_rules(self, rules: list[TagRule]) -> None:
        """Add multiple tagging rules.

        Args:
            rules: Rules to add
        """
        for rule in rules:
            self.add_rule(rule)

    def add_handler(
        self,
        handler: Callable[[MemoryEntry], list[tuple[str, float]]],
    ) -> None:
        """Add a custom tag handler.

        Handler should return list of (tag, confidence) tuples.

        Args:
            handler: Custom handler function
        """
        self._custom_handlers.append(handler)

    def tag_entry(
        self,
        entry: MemoryEntry,
        merge_existing: bool = True,
    ) -> TaggingResult:
        """Auto-tag a memory entry.

        Args:
            entry: Memory entry to tag
            merge_existing: Whether to keep existing tags

        Returns:
            Tagging result
        """
        start_time = datetime.now(timezone.utc)
        original_tags = entry.tags.copy()
        tag_scores: dict[str, float] = {}
        methods_used = []

        # Apply keyword extraction
        if self._config.enable_keywords:
            keywords = self._extract_keywords(entry.content)
            for tag, score in keywords:
                tag_scores[tag] = max(tag_scores.get(tag, 0), score)
            if keywords:
                methods_used.append("keywords")

        # Apply entity extraction
        if self._config.enable_entities:
            entities = self._extract_entities(entry.content)
            for tag, score in entities:
                tag_scores[tag] = max(tag_scores.get(tag, 0), score)
            if entities:
                methods_used.append("entities")

        # Apply topic extraction
        if self._config.enable_topics:
            topics = self._extract_topics(entry)
            for tag, score in topics:
                tag_scores[tag] = max(tag_scores.get(tag, 0), score)
            if topics:
                methods_used.append("topics")

        # Apply rules
        if self._config.enable_rules:
            rule_tags = self._apply_rules(entry.content)
            for tag, score in rule_tags:
                tag_scores[tag] = max(tag_scores.get(tag, 0), score)
            if rule_tags:
                methods_used.append("rules")

        # Apply custom handlers
        for handler in self._custom_handlers:
            try:
                handler_tags = handler(entry)
                for tag, score in handler_tags:
                    tag_scores[tag] = max(tag_scores.get(tag, 0), score)
                if handler_tags:
                    methods_used.append("custom")
            except Exception as e:
                logger.warning(f"Custom handler failed: {e}")

        # Filter by confidence and select top tags
        qualified_tags = [
            (tag, score)
            for tag, score in tag_scores.items()
            if score >= self._config.min_confidence
        ]
        qualified_tags.sort(key=lambda x: x[1], reverse=True)
        qualified_tags = qualified_tags[:self._config.max_tags]

        # Determine new tags
        new_tags = [tag for tag, _ in qualified_tags]
        added_tags = [tag for tag in new_tags if tag not in original_tags]

        # Build final tag list
        if merge_existing:
            all_tags = list(set(original_tags + new_tags))
        else:
            all_tags = new_tags

        # Limit total tags
        all_tags = all_tags[:self._config.max_tags]

        # Update entry tags
        entry.tags = all_tags

        # Calculate processing time
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Update statistics
        self._total_tagged += 1
        self._tags_added += len(added_tags)

        result = TaggingResult(
            memory_id=str(entry.id),
            original_tags=original_tags,
            added_tags=added_tags,
            all_tags=all_tags,
            confidence_scores={tag: score for tag, score in qualified_tags},
            methods_used=methods_used,
            processing_time_ms=elapsed,
        )

        logger.debug(
            "Entry tagged",
            memory_id=str(entry.id)[:8],
            added=len(added_tags),
            total=len(all_tags),
        )

        return result

    def tag_batch(
        self,
        entries: list[MemoryEntry],
        merge_existing: bool = True,
    ) -> list[TaggingResult]:
        """Auto-tag multiple entries.

        Args:
            entries: Entries to tag
            merge_existing: Whether to keep existing tags

        Returns:
            List of tagging results
        """
        results = []
        for entry in entries:
            result = self.tag_entry(entry, merge_existing)
            results.append(result)
        return results

    def _extract_keywords(self, content: str) -> list[tuple[str, float]]:
        """Extract keywords from content.

        Args:
            content: Text content

        Returns:
            List of (keyword, confidence) tuples
        """
        # Tokenize
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', content.lower())

        # Filter by length and stop words
        words = [
            w for w in words
            if (self._config.min_keyword_length <= len(w) <= self._config.max_keyword_length
                and w not in self._config.stop_words)
        ]

        # Count frequency
        word_counts = Counter(words)
        total = sum(word_counts.values()) or 1

        # Calculate scores based on frequency
        keywords = []
        for word, count in word_counts.most_common(self._config.keyword_top_n):
            score = min(1.0, count / total * 10)  # Normalize score
            keywords.append((word, score))

        return keywords

    def _extract_entities(self, content: str) -> list[tuple[str, float]]:
        """Extract named entities from content.

        Uses pattern matching for common entity types.

        Args:
            content: Text content

        Returns:
            List of (entity_tag, confidence) tuples
        """
        entities = []

        # Email pattern
        if re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', content):
            entities.append(("email", 0.9))

        # URL pattern
        if re.search(r'https?://\S+', content):
            entities.append(("url", 0.9))

        # Code patterns
        if re.search(r'```[\s\S]*?```', content):
            entities.append(("code", 0.95))
        elif re.search(r'`[^`]+`', content):
            entities.append(("code", 0.7))

        # Function/method patterns
        if re.search(r'\b(def|function|func|class|interface)\s+\w+', content, re.IGNORECASE):
            entities.append(("code", 0.9))

        # Error/exception patterns
        if re.search(r'\b(error|exception|traceback|failed|failure)\b', content, re.IGNORECASE):
            entities.append(("error", 0.8))

        # Question patterns
        if re.search(r'\?$|\bhow\s+(do|can|to)\b|\bwhat\s+(is|are)\b|\bwhy\b', content, re.IGNORECASE):
            entities.append(("question", 0.7))

        # Task/todo patterns
        if re.search(r'\b(todo|task|implement|fix|create|add|update|remove)\b', content, re.IGNORECASE):
            entities.append(("task", 0.7))

        # File path patterns
        if re.search(r'[/\\][\w.-]+\.[a-zA-Z]{1,4}\b', content):
            entities.append(("file", 0.8))

        # JSON/dict patterns
        if re.search(r'\{[^{}]*:[^{}]*\}', content):
            entities.append(("data", 0.7))

        # Date patterns
        if re.search(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{2}[-/]\d{2}[-/]\d{4}\b', content):
            entities.append(("date", 0.8))

        # Number/metric patterns
        if re.search(r'\b\d+(\.\d+)?%\b|\b\d+\s*(ms|sec|min|hour|day|KB|MB|GB)\b', content, re.IGNORECASE):
            entities.append(("metric", 0.7))

        return entities

    def _extract_topics(self, entry: MemoryEntry) -> list[tuple[str, float]]:
        """Extract topic tags based on content and metadata.

        Args:
            entry: Memory entry

        Returns:
            List of (topic_tag, confidence) tuples
        """
        topics = []
        content_lower = entry.content.lower()

        # Layer-based topics
        layer_topics = {
            MemoryLayer.WORKING: [("recent", 0.5), ("active", 0.5)],
            MemoryLayer.EPISODIC: [("conversation", 0.5), ("context", 0.5)],
            MemoryLayer.SEMANTIC: [("knowledge", 0.5), ("learned", 0.5)],
        }
        topics.extend(layer_topics.get(entry.layer, []))

        # Technology topics
        tech_patterns = {
            "python": (r'\bpython\b|\.py\b', 0.85),
            "javascript": (r'\bjavascript\b|\.js\b|\bnode\b', 0.85),
            "typescript": (r'\btypescript\b|\.ts\b', 0.85),
            "sql": (r'\bsql\b|select\s+.+\s+from\b', 0.8),
            "api": (r'\bapi\b|\brest\b|\bendpoint\b', 0.75),
            "database": (r'\bdatabase\b|\bdb\b|\bsqlite\b|\bpostgres\b|\bmongo\b', 0.8),
            "testing": (r'\btest\b|\bpytest\b|\bunittest\b|\bspec\b', 0.75),
            "docker": (r'\bdocker\b|\bcontainer\b|\bkubernetes\b|\bk8s\b', 0.85),
            "git": (r'\bgit\b|\bcommit\b|\bbranch\b|\bmerge\b', 0.75),
            "ai": (r'\b(ai|ml|llm|gpt|claude|openai|embedding)\b', 0.85),
        }

        for topic, (pattern, confidence) in tech_patterns.items():
            if re.search(pattern, content_lower, re.IGNORECASE):
                topics.append((topic, confidence))

        # Category from metadata
        if "category" in entry.metadata:
            category = entry.metadata["category"]
            if isinstance(category, str):
                topics.append((category.lower(), 0.9))

        # Source from metadata
        if "source" in entry.metadata:
            source = entry.metadata["source"]
            if isinstance(source, str):
                topics.append((f"from:{source.lower()}", 0.7))

        return topics

    def _apply_rules(self, content: str) -> list[tuple[str, float]]:
        """Apply tagging rules to content.

        Args:
            content: Text content

        Returns:
            List of (tag, confidence) tuples
        """
        tags = []
        for rule in self._rules:
            if rule.matches(content):
                for tag in rule.tags:
                    tags.append((tag, 0.9))  # High confidence for rule matches
        return tags

    def suggest_tags(
        self,
        content: str,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Suggest tags for content without applying them.

        Args:
            content: Text content
            top_n: Number of suggestions

        Returns:
            List of (tag, confidence) tuples
        """
        tag_scores: dict[str, float] = {}

        # Extract from all methods
        for tag, score in self._extract_keywords(content):
            tag_scores[tag] = max(tag_scores.get(tag, 0), score)

        for tag, score in self._extract_entities(content):
            tag_scores[tag] = max(tag_scores.get(tag, 0), score)

        for tag, score in self._apply_rules(content):
            tag_scores[tag] = max(tag_scores.get(tag, 0), score)

        # Sort and return top N
        suggestions = sorted(
            tag_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return suggestions[:top_n]

    def get_stats(self) -> dict:
        """Get tagger statistics."""
        return {
            "total_tagged": self._total_tagged,
            "tags_added": self._tags_added,
            "avg_tags_per_entry": self._tags_added / self._total_tagged if self._total_tagged else 0,
            "rules_count": len(self._rules),
            "custom_handlers": len(self._custom_handlers),
            "config": {
                "max_tags": self._config.max_tags,
                "min_confidence": self._config.min_confidence,
                "enable_keywords": self._config.enable_keywords,
                "enable_entities": self._config.enable_entities,
                "enable_topics": self._config.enable_topics,
                "enable_rules": self._config.enable_rules,
            },
        }


# Predefined common rules
COMMON_RULES = [
    TagRule(
        name="bug_report",
        pattern=r"\bbug\b|\bissue\b|\bproblem\b|\berror\b",
        tags=["bug", "issue"],
        priority=5,
    ),
    TagRule(
        name="feature_request",
        pattern=r"\bfeature\b|\brequest\b|\benhancement\b|\bimprovement\b",
        tags=["feature", "enhancement"],
        priority=5,
    ),
    TagRule(
        name="documentation",
        pattern=r"\bdoc(s|umentation)?\b|\breadme\b|\bguide\b",
        tags=["documentation"],
        priority=3,
    ),
    TagRule(
        name="security",
        pattern=r"\bsecurity\b|\bvulnerability\b|\bauth(entication|orization)?\b|\btoken\b",
        tags=["security"],
        priority=7,
    ),
    TagRule(
        name="performance",
        pattern=r"\bperformance\b|\boptimiz(e|ation)\b|\bslow\b|\bfast\b|\bspeed\b",
        tags=["performance"],
        priority=4,
    ),
    TagRule(
        name="refactor",
        pattern=r"\brefactor\b|\bcleanup\b|\brestructure\b",
        tags=["refactor"],
        priority=3,
    ),
    TagRule(
        name="urgent",
        pattern=r"\burgent\b|\basap\b|\bcritical\b|\bimmediate\b",
        tags=["urgent", "priority:high"],
        priority=10,
    ),
]


def create_auto_tagger(
    include_common_rules: bool = True,
    **config_kwargs,
) -> AutoTagger:
    """Create an auto-tagger with optional common rules.

    Args:
        include_common_rules: Include predefined common rules
        **config_kwargs: TaggingConfig parameters

    Returns:
        Configured AutoTagger
    """
    config = TaggingConfig(**config_kwargs)
    tagger = AutoTagger(config)

    if include_common_rules:
        tagger.add_rules(COMMON_RULES)

    return tagger
