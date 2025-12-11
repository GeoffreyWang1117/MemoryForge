"""Memory analytics for insights into memory usage patterns."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

import structlog

from memoryforge.core.types import MemoryEntry, MemoryLayer

logger = structlog.get_logger()


@dataclass
class TopicCluster:
    """A cluster of related memories by topic."""

    topic: str
    keywords: list[str]
    memory_ids: list[str]
    memory_count: int
    avg_importance: float
    first_seen: datetime
    last_seen: datetime

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "keywords": self.keywords,
            "memory_count": self.memory_count,
            "avg_importance": round(self.avg_importance, 3),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "duration_hours": (self.last_seen - self.first_seen).total_seconds() / 3600,
        }


@dataclass
class UsagePattern:
    """A detected memory usage pattern."""

    pattern_type: str
    description: str
    frequency: int
    examples: list[str]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "examples": self.examples[:3],
            "metadata": self.metadata,
        }


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report for memory usage."""

    # Time range
    start_time: datetime
    end_time: datetime

    # Basic stats
    total_memories: int = 0
    memories_by_layer: dict[str, int] = field(default_factory=dict)
    avg_importance: float = 0.0
    importance_distribution: dict[str, int] = field(default_factory=dict)

    # Time-based metrics
    memories_per_hour: dict[str, int] = field(default_factory=dict)
    peak_hours: list[int] = field(default_factory=list)
    daily_trend: list[dict] = field(default_factory=list)

    # Content analysis
    top_tags: list[tuple[str, int]] = field(default_factory=list)
    topic_clusters: list[TopicCluster] = field(default_factory=list)
    avg_content_length: float = 0.0

    # Access patterns
    most_accessed: list[dict] = field(default_factory=list)
    recently_accessed: list[dict] = field(default_factory=list)
    usage_patterns: list[UsagePattern] = field(default_factory=list)

    # Retention metrics
    retention_rate: float = 0.0
    compression_candidates: int = 0

    def to_dict(self) -> dict:
        return {
            "time_range": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "duration_hours": (self.end_time - self.start_time).total_seconds() / 3600,
            },
            "summary": {
                "total_memories": self.total_memories,
                "memories_by_layer": self.memories_by_layer,
                "avg_importance": round(self.avg_importance, 3),
                "avg_content_length": round(self.avg_content_length, 1),
            },
            "importance_distribution": self.importance_distribution,
            "time_analysis": {
                "peak_hours": self.peak_hours,
                "memories_per_hour": self.memories_per_hour,
                "daily_trend": self.daily_trend,
            },
            "content_analysis": {
                "top_tags": [{"tag": t, "count": c} for t, c in self.top_tags],
                "topic_clusters": [c.to_dict() for c in self.topic_clusters],
            },
            "access_patterns": {
                "most_accessed": self.most_accessed,
                "recently_accessed": self.recently_accessed,
                "patterns": [p.to_dict() for p in self.usage_patterns],
            },
            "retention": {
                "retention_rate": round(self.retention_rate, 3),
                "compression_candidates": self.compression_candidates,
            },
        }


class MemoryAnalyzer:
    """Analyzes memory entries to provide insights and patterns.

    Provides:
    - Usage statistics and trends
    - Topic clustering
    - Access pattern detection
    - Retention recommendations
    - Time-based analysis
    """

    def __init__(
        self,
        stop_words: set[str] | None = None,
        min_cluster_size: int = 2,
    ):
        """Initialize the analyzer.

        Args:
            stop_words: Words to exclude from topic analysis
            min_cluster_size: Minimum memories for a topic cluster
        """
        self._stop_words = stop_words or {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "but", "and", "or", "if", "because", "until", "while",
            "this", "that", "these", "those", "it", "its", "i", "you", "he",
            "she", "we", "they", "what", "which", "who", "whom", "your",
        }
        self._min_cluster_size = min_cluster_size

    def analyze(
        self,
        entries: list[MemoryEntry],
        time_window_hours: int = 24,
    ) -> AnalyticsReport:
        """Generate a comprehensive analytics report.

        Args:
            entries: Memory entries to analyze
            time_window_hours: Time window for hourly analysis

        Returns:
            Analytics report with insights
        """
        if not entries:
            now = datetime.now(timezone.utc)
            return AnalyticsReport(start_time=now, end_time=now)

        # Determine time range
        times = [e.created_at for e in entries]
        start_time = min(times)
        end_time = max(times)

        report = AnalyticsReport(
            start_time=start_time,
            end_time=end_time,
            total_memories=len(entries),
        )

        # Basic statistics
        self._analyze_basic_stats(entries, report)

        # Time-based analysis
        self._analyze_time_patterns(entries, report, time_window_hours)

        # Content analysis
        self._analyze_content(entries, report)

        # Access patterns
        self._analyze_access_patterns(entries, report)

        # Retention metrics
        self._analyze_retention(entries, report)

        return report

    def _analyze_basic_stats(
        self,
        entries: list[MemoryEntry],
        report: AnalyticsReport,
    ) -> None:
        """Analyze basic statistics."""
        # Layer distribution
        layer_counts = Counter(e.layer.value for e in entries)
        report.memories_by_layer = dict(layer_counts)

        # Importance statistics
        importances = [e.importance.effective_score for e in entries]
        report.avg_importance = sum(importances) / len(importances)

        # Importance distribution
        bins = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for imp in importances:
            if imp < 0.3:
                bins["low"] += 1
            elif imp < 0.6:
                bins["medium"] += 1
            elif imp < 0.9:
                bins["high"] += 1
            else:
                bins["critical"] += 1
        report.importance_distribution = bins

        # Content length
        lengths = [len(e.content) for e in entries]
        report.avg_content_length = sum(lengths) / len(lengths)

    def _analyze_time_patterns(
        self,
        entries: list[MemoryEntry],
        report: AnalyticsReport,
        window_hours: int,
    ) -> None:
        """Analyze time-based patterns."""
        # Memories per hour
        hourly: dict[str, int] = defaultdict(int)
        hour_counts = Counter()

        for entry in entries:
            hour_key = entry.created_at.strftime("%Y-%m-%d %H:00")
            hourly[hour_key] += 1
            hour_counts[entry.created_at.hour] += 1

        report.memories_per_hour = dict(hourly)

        # Peak hours (top 3)
        report.peak_hours = [h for h, _ in hour_counts.most_common(3)]

        # Daily trend
        daily: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "avg_importance": 0, "importances": []}
        )
        for entry in entries:
            day_key = entry.created_at.strftime("%Y-%m-%d")
            daily[day_key]["count"] += 1
            daily[day_key]["importances"].append(entry.importance.effective_score)

        report.daily_trend = [
            {
                "date": day,
                "count": data["count"],
                "avg_importance": sum(data["importances"]) / len(data["importances"]),
            }
            for day, data in sorted(daily.items())
        ]

    def _analyze_content(
        self,
        entries: list[MemoryEntry],
        report: AnalyticsReport,
    ) -> None:
        """Analyze content and topics."""
        # Tag analysis
        tag_counts = Counter()
        for entry in entries:
            tag_counts.update(entry.tags)
        report.top_tags = tag_counts.most_common(10)

        # Topic clustering using keyword extraction
        word_to_entries: dict[str, list[MemoryEntry]] = defaultdict(list)

        for entry in entries:
            keywords = self._extract_keywords(entry.content)
            for word in keywords:
                word_to_entries[word].append(entry)

        # Create clusters from frequent keywords
        clusters = []
        used_entries: set[str] = set()

        for word, word_entries in sorted(
            word_to_entries.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        ):
            if len(word_entries) < self._min_cluster_size:
                continue

            # Skip if most entries already used
            unused = [e for e in word_entries if str(e.id) not in used_entries]
            if len(unused) < self._min_cluster_size:
                continue

            # Create cluster
            entry_ids = [str(e.id) for e in unused[:20]]  # Limit cluster size
            used_entries.update(entry_ids)

            importances = [e.importance.effective_score for e in unused[:20]]
            times = [e.created_at for e in unused[:20]]

            # Get related keywords
            related = self._get_related_keywords(word, unused[:20])

            cluster = TopicCluster(
                topic=word,
                keywords=related[:5],
                memory_ids=entry_ids,
                memory_count=len(entry_ids),
                avg_importance=sum(importances) / len(importances),
                first_seen=min(times),
                last_seen=max(times),
            )
            clusters.append(cluster)

            if len(clusters) >= 10:  # Limit to top 10 clusters
                break

        report.topic_clusters = clusters

    def _analyze_access_patterns(
        self,
        entries: list[MemoryEntry],
        report: AnalyticsReport,
    ) -> None:
        """Analyze memory access patterns."""
        # Most accessed
        sorted_by_access = sorted(
            entries,
            key=lambda e: e.importance.access_count,
            reverse=True,
        )[:5]

        report.most_accessed = [
            {
                "id": str(e.id),
                "content_preview": e.content[:100],
                "access_count": e.importance.access_count,
            }
            for e in sorted_by_access
        ]

        # Recently accessed
        sorted_by_recent = sorted(
            entries,
            key=lambda e: e.importance.last_accessed,
            reverse=True,
        )[:5]

        report.recently_accessed = [
            {
                "id": str(e.id),
                "content_preview": e.content[:100],
                "last_accessed": e.importance.last_accessed.isoformat(),
            }
            for e in sorted_by_recent
        ]

        # Detect usage patterns
        patterns = []

        # Pattern: Burst creation
        time_diffs = []
        sorted_entries = sorted(entries, key=lambda e: e.created_at)
        for i in range(1, len(sorted_entries)):
            diff = (
                sorted_entries[i].created_at - sorted_entries[i - 1].created_at
            ).total_seconds()
            time_diffs.append(diff)

        if time_diffs:
            avg_diff = sum(time_diffs) / len(time_diffs)
            burst_count = sum(1 for d in time_diffs if d < avg_diff / 3)

            if burst_count > len(time_diffs) * 0.3:
                patterns.append(
                    UsagePattern(
                        pattern_type="burst_creation",
                        description="Memories are created in bursts",
                        frequency=burst_count,
                        examples=[sorted_entries[i].content[:50] for i in range(min(3, len(sorted_entries)))],
                        metadata={"avg_interval_seconds": avg_diff},
                    )
                )

        # Pattern: High importance focus
        high_imp = [e for e in entries if e.importance.effective_score > 0.7]
        if len(high_imp) > len(entries) * 0.3:
            patterns.append(
                UsagePattern(
                    pattern_type="high_importance_focus",
                    description="Many memories marked as high importance",
                    frequency=len(high_imp),
                    examples=[e.content[:50] for e in high_imp[:3]],
                    metadata={"percentage": len(high_imp) / len(entries) * 100},
                )
            )

        # Pattern: Tag heavy
        tagged = [e for e in entries if e.tags]
        if len(tagged) > len(entries) * 0.5:
            patterns.append(
                UsagePattern(
                    pattern_type="tag_heavy",
                    description="Most memories are tagged",
                    frequency=len(tagged),
                    examples=[f"{e.content[:30]}... tags: {e.tags}" for e in tagged[:3]],
                    metadata={"percentage": len(tagged) / len(entries) * 100},
                )
            )

        report.usage_patterns = patterns

    def _analyze_retention(
        self,
        entries: list[MemoryEntry],
        report: AnalyticsReport,
    ) -> None:
        """Analyze retention metrics."""
        now = datetime.now(timezone.utc)

        # Retention rate (accessed recently / total)
        recent_threshold = now - timedelta(hours=24)
        recently_accessed = sum(
            1 for e in entries
            if e.importance.last_accessed > recent_threshold
        )
        report.retention_rate = recently_accessed / len(entries) if entries else 0

        # Compression candidates (old, low importance, not accessed)
        old_threshold = now - timedelta(hours=48)
        compression_candidates = sum(
            1 for e in entries
            if (
                e.created_at < old_threshold
                and e.importance.effective_score < 0.5
                and e.importance.access_count < 3
            )
        )
        report.compression_candidates = compression_candidates

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = text.lower().split()

        # Clean and filter
        keywords = []
        for word in words:
            # Remove punctuation
            word = "".join(c for c in word if c.isalnum())
            if (
                word
                and len(word) > 2
                and word not in self._stop_words
                and not word.isdigit()
            ):
                keywords.append(word)

        return list(set(keywords))

    def _get_related_keywords(
        self,
        topic: str,
        entries: list[MemoryEntry],
    ) -> list[str]:
        """Get keywords related to a topic from entries."""
        word_counts: Counter = Counter()

        for entry in entries:
            keywords = self._extract_keywords(entry.content)
            for word in keywords:
                if word != topic:
                    word_counts[word] += 1

        return [w for w, _ in word_counts.most_common(10)]

    def get_recommendations(
        self,
        report: AnalyticsReport,
    ) -> list[dict]:
        """Generate recommendations based on analytics.

        Args:
            report: Analytics report to analyze

        Returns:
            List of recommendations
        """
        recommendations = []

        # Recommendation: Compress old memories
        if report.compression_candidates > 10:
            recommendations.append({
                "type": "compression",
                "priority": "medium",
                "title": "Consider memory compression",
                "description": f"Found {report.compression_candidates} memories that could be compressed",
                "action": "Run compression pipeline on old, low-importance memories",
            })

        # Recommendation: Low retention
        if report.retention_rate < 0.3:
            recommendations.append({
                "type": "retention",
                "priority": "high",
                "title": "Low memory retention rate",
                "description": f"Only {report.retention_rate:.1%} of memories accessed recently",
                "action": "Review memory importance scoring or increase context usage",
            })

        # Recommendation: Untagged memories
        if report.total_memories > 0:
            tagged = sum(c for _, c in report.top_tags) if report.top_tags else 0
            if tagged < report.total_memories * 0.3:
                recommendations.append({
                    "type": "organization",
                    "priority": "low",
                    "title": "Most memories are untagged",
                    "description": "Adding tags improves retrieval accuracy",
                    "action": "Enable auto-tagging or add tags manually",
                })

        # Recommendation: Importance skew
        if report.importance_distribution:
            critical = report.importance_distribution.get("critical", 0)
            low = report.importance_distribution.get("low", 0)

            if critical > report.total_memories * 0.5:
                recommendations.append({
                    "type": "scoring",
                    "priority": "medium",
                    "title": "Too many critical-importance memories",
                    "description": f"{critical} memories marked as critical",
                    "action": "Calibrate importance scoring thresholds",
                })

            if low > report.total_memories * 0.7:
                recommendations.append({
                    "type": "scoring",
                    "priority": "medium",
                    "title": "Most memories have low importance",
                    "description": f"{low} memories with low importance",
                    "action": "Review importance scoring configuration",
                })

        return recommendations
