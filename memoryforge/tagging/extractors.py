"""Tag extractors for different content types."""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import structlog

logger = structlog.get_logger()


class TagExtractor(Protocol):
    """Protocol for tag extractors."""

    def extract(self, content: str) -> list[tuple[str, float]]:
        """Extract tags from content.

        Args:
            content: Text content

        Returns:
            List of (tag, confidence) tuples
        """
        ...


@dataclass
class KeywordExtractor:
    """Extract keywords as tags from text content.

    Uses TF-IDF-like scoring for keyword importance.
    """

    # Minimum word length
    min_length: int = 3

    # Maximum word length
    max_length: int = 30

    # Top N keywords to extract
    top_n: int = 5

    # Minimum frequency to consider
    min_frequency: int = 1

    # Stop words to exclude
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
    })

    # Technical terms to boost
    boost_terms: set[str] = field(default_factory=lambda: {
        "api", "database", "server", "client", "function", "class", "method",
        "error", "bug", "fix", "feature", "test", "config", "deploy", "build",
        "cache", "memory", "query", "response", "request", "async", "sync",
    })

    def extract(self, content: str) -> list[tuple[str, float]]:
        """Extract keywords from content.

        Args:
            content: Text content

        Returns:
            List of (keyword, confidence) tuples
        """
        # Tokenize
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', content.lower())

        # Filter
        filtered = [
            w for w in words
            if (self.min_length <= len(w) <= self.max_length
                and w not in self.stop_words)
        ]

        if not filtered:
            return []

        # Count frequencies
        word_counts = Counter(filtered)

        # Filter by minimum frequency
        word_counts = Counter({
            w: c for w, c in word_counts.items()
            if c >= self.min_frequency
        })

        if not word_counts:
            return []

        # Calculate scores
        max_count = max(word_counts.values())
        keywords = []

        for word, count in word_counts.most_common(self.top_n * 2):
            # Base score from frequency
            score = count / max_count

            # Boost technical terms
            if word in self.boost_terms:
                score = min(1.0, score * 1.5)

            # Slight penalty for very short words
            if len(word) < 5:
                score *= 0.9

            keywords.append((word, round(score, 3)))

        # Sort by score and return top N
        keywords.sort(key=lambda x: x[1], reverse=True)
        return keywords[:self.top_n]


@dataclass
class EntityExtractor:
    """Extract named entities as tags.

    Recognizes patterns for:
    - Emails, URLs
    - Code snippets
    - File paths
    - Dates, numbers
    - Technical patterns
    """

    # Entity patterns with (pattern, tag, confidence)
    patterns: list[tuple[str, str, float]] = field(default_factory=lambda: [
        # Communication
        (r'\b[\w.-]+@[\w.-]+\.\w+\b', 'email', 0.9),
        (r'https?://\S+', 'url', 0.9),

        # Code
        (r'```[\s\S]*?```', 'code-block', 0.95),
        (r'`[^`]+`', 'code-inline', 0.7),
        (r'\bdef\s+\w+\s*\(', 'python', 0.9),
        (r'\bfunction\s+\w+\s*\(', 'javascript', 0.85),
        (r'\bclass\s+[A-Z]\w+', 'class', 0.85),
        (r'\bimport\s+[\w.]+', 'import', 0.7),
        (r'\bfrom\s+[\w.]+\s+import\b', 'python', 0.8),

        # Files
        (r'\b[\w/\\-]+\.py\b', 'python-file', 0.85),
        (r'\b[\w/\\-]+\.js\b', 'js-file', 0.85),
        (r'\b[\w/\\-]+\.ts\b', 'ts-file', 0.85),
        (r'\b[\w/\\-]+\.json\b', 'json-file', 0.85),
        (r'\b[\w/\\-]+\.yaml\b', 'yaml-file', 0.85),
        (r'\b[\w/\\-]+\.md\b', 'markdown-file', 0.85),

        # Data formats
        (r'\{[^{}]*"[^"]*":\s*[^{}]*\}', 'json', 0.75),
        (r'\[\s*\{', 'json-array', 0.7),

        # Dates and times
        (r'\b\d{4}-\d{2}-\d{2}\b', 'date', 0.85),
        (r'\b\d{2}:\d{2}(:\d{2})?\b', 'time', 0.8),

        # Metrics
        (r'\b\d+(\.\d+)?%\b', 'percentage', 0.75),
        (r'\b\d+\s*(ms|sec|min|hour|day)\b', 'duration', 0.75),
        (r'\b\d+\s*(KB|MB|GB|TB)\b', 'size', 0.75),

        # Errors
        (r'\bTraceback\b', 'traceback', 0.95),
        (r'\b(Error|Exception):', 'error', 0.9),
        (r'\bfailed\b', 'failure', 0.7),

        # Commands
        (r'\$\s*[\w-]+', 'shell-command', 0.75),
        (r'\bgit\s+(commit|push|pull|merge|branch|checkout)\b', 'git', 0.85),
        (r'\b(npm|yarn|pip)\s+(install|run|build)\b', 'package-manager', 0.8),
        (r'\bdocker\s+(run|build|compose)\b', 'docker', 0.85),

        # API
        (r'\b(GET|POST|PUT|DELETE|PATCH)\s+/[\w/]+', 'http-method', 0.9),
        (r'\bapi/v\d+/', 'api-versioned', 0.85),

        # Environment
        (r'\b[A-Z][A-Z0-9_]+\s*=', 'env-var', 0.75),
    ])

    def extract(self, content: str) -> list[tuple[str, float]]:
        """Extract entities from content.

        Args:
            content: Text content

        Returns:
            List of (entity_tag, confidence) tuples
        """
        entities = []
        seen_tags = set()

        for pattern, tag, confidence in self.patterns:
            if tag in seen_tags:
                continue

            if re.search(pattern, content, re.IGNORECASE):
                entities.append((tag, confidence))
                seen_tags.add(tag)

        return entities


@dataclass
class TopicExtractor:
    """Extract topic tags based on content analysis.

    Identifies topics from:
    - Domain-specific vocabulary
    - Category indicators
    - Content structure
    """

    # Topic vocabularies: topic -> keywords
    topic_vocabularies: dict[str, set[str]] = field(default_factory=lambda: {
        "web-development": {
            "html", "css", "javascript", "react", "vue", "angular", "dom",
            "browser", "frontend", "backend", "spa", "ssr", "webpack",
        },
        "data-science": {
            "pandas", "numpy", "matplotlib", "sklearn", "tensorflow", "pytorch",
            "dataset", "model", "training", "prediction", "classification",
        },
        "devops": {
            "docker", "kubernetes", "ci", "cd", "pipeline", "deploy", "terraform",
            "ansible", "jenkins", "github-actions", "aws", "azure", "gcp",
        },
        "database": {
            "sql", "postgres", "mysql", "sqlite", "mongodb", "redis", "query",
            "index", "migration", "schema", "table", "collection",
        },
        "security": {
            "auth", "authentication", "authorization", "token", "jwt", "oauth",
            "encryption", "hash", "password", "vulnerability", "xss", "csrf",
        },
        "testing": {
            "test", "pytest", "unittest", "jest", "mocha", "coverage", "mock",
            "fixture", "assert", "spec", "tdd", "bdd",
        },
        "api": {
            "rest", "graphql", "endpoint", "request", "response", "json",
            "http", "webhook", "websocket", "grpc", "openapi", "swagger",
        },
        "machine-learning": {
            "llm", "gpt", "claude", "embedding", "vector", "transformer",
            "attention", "neural", "inference", "fine-tune", "prompt",
        },
    })

    # Minimum matches to assign topic
    min_matches: int = 2

    def extract(self, content: str) -> list[tuple[str, float]]:
        """Extract topics from content.

        Args:
            content: Text content

        Returns:
            List of (topic_tag, confidence) tuples
        """
        content_lower = content.lower()
        words = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', content_lower))

        topics = []

        for topic, vocabulary in self.topic_vocabularies.items():
            matches = words & vocabulary
            if len(matches) >= self.min_matches:
                # Score based on match ratio
                score = min(1.0, len(matches) / self.min_matches * 0.5)
                topics.append((topic, round(score, 3)))

        return topics


@dataclass
class CompositeExtractor:
    """Combines multiple extractors.

    Merges results from all extractors, keeping highest confidence
    for duplicate tags.
    """

    extractors: list[TagExtractor] = field(default_factory=list)

    def add_extractor(self, extractor: TagExtractor) -> None:
        """Add an extractor.

        Args:
            extractor: Extractor to add
        """
        self.extractors.append(extractor)

    def extract(self, content: str) -> list[tuple[str, float]]:
        """Extract tags using all extractors.

        Args:
            content: Text content

        Returns:
            List of (tag, confidence) tuples
        """
        tag_scores: dict[str, float] = {}

        for extractor in self.extractors:
            try:
                tags = extractor.extract(content)
                for tag, score in tags:
                    tag_scores[tag] = max(tag_scores.get(tag, 0), score)
            except Exception as e:
                logger.warning(f"Extractor failed: {e}")

        # Sort by score
        sorted_tags = sorted(
            tag_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_tags


def create_default_extractor() -> CompositeExtractor:
    """Create a composite extractor with default configuration.

    Returns:
        Configured CompositeExtractor
    """
    extractor = CompositeExtractor()
    extractor.add_extractor(KeywordExtractor())
    extractor.add_extractor(EntityExtractor())
    extractor.add_extractor(TopicExtractor())
    return extractor
