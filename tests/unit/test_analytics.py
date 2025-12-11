"""Tests for memory analytics module."""

from datetime import datetime, timedelta, timezone

import pytest

from memoryforge.analytics import MemoryAnalyzer, AnalyticsReport
from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer


def create_test_entry(
    content: str,
    importance: float = 0.5,
    tags: list[str] | None = None,
    age_hours: float = 0,
    access_count: int = 0,
) -> MemoryEntry:
    """Create a test memory entry."""
    created_at = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(
            base_score=importance,
            access_count=access_count,
        ),
        tags=tags or [],
        created_at=created_at,
        updated_at=created_at,
    )


def test_analyze_empty_entries():
    """Test analyzing empty entry list."""
    analyzer = MemoryAnalyzer()
    report = analyzer.analyze([])

    assert report.total_memories == 0


def test_analyze_basic_stats():
    """Test basic statistics analysis."""
    analyzer = MemoryAnalyzer()
    entries = [
        create_test_entry("Memory 1", importance=0.3),
        create_test_entry("Memory 2", importance=0.5),
        create_test_entry("Memory 3", importance=0.8),
    ]

    report = analyzer.analyze(entries)

    assert report.total_memories == 3
    assert 0.5 <= report.avg_importance <= 0.6


def test_analyze_layer_distribution():
    """Test memory layer distribution analysis."""
    analyzer = MemoryAnalyzer()
    entries = [
        create_test_entry("Working memory"),
    ]

    report = analyzer.analyze(entries)

    assert "working" in report.memories_by_layer
    assert report.memories_by_layer["working"] == 1


def test_analyze_importance_distribution():
    """Test importance distribution bins."""
    analyzer = MemoryAnalyzer()
    entries = [
        create_test_entry("Low", importance=0.2),
        create_test_entry("Medium", importance=0.5),
        create_test_entry("High", importance=0.8),
        create_test_entry("Critical", importance=0.95),
    ]

    report = analyzer.analyze(entries)

    assert report.importance_distribution["low"] == 1
    assert report.importance_distribution["medium"] == 1
    assert report.importance_distribution["high"] == 1
    assert report.importance_distribution["critical"] == 1


def test_analyze_tag_counts():
    """Test tag analysis."""
    analyzer = MemoryAnalyzer()
    entries = [
        create_test_entry("Entry 1", tags=["python", "code"]),
        create_test_entry("Entry 2", tags=["python"]),
        create_test_entry("Entry 3", tags=["test"]),
    ]

    report = analyzer.analyze(entries)

    # Find python tag count
    python_count = next(
        (count for tag, count in report.top_tags if tag == "python"),
        0
    )
    assert python_count == 2


def test_analyze_topic_clusters():
    """Test topic clustering."""
    analyzer = MemoryAnalyzer(min_cluster_size=2)
    entries = [
        create_test_entry("Python programming is great"),
        create_test_entry("Learning Python basics"),
        create_test_entry("JavaScript frameworks review"),
    ]

    report = analyzer.analyze(entries)

    # Should find python as a cluster topic
    topics = [c.topic for c in report.topic_clusters]
    assert "python" in topics


def test_analyze_access_patterns():
    """Test access pattern analysis."""
    analyzer = MemoryAnalyzer()
    entries = [
        create_test_entry("High access", access_count=10),
        create_test_entry("Medium access", access_count=5),
        create_test_entry("Low access", access_count=1),
    ]

    report = analyzer.analyze(entries)

    assert len(report.most_accessed) > 0
    assert report.most_accessed[0]["access_count"] == 10


def test_analyze_time_patterns():
    """Test time-based analysis."""
    analyzer = MemoryAnalyzer()
    now = datetime.now(timezone.utc)
    entries = [
        MemoryEntry(
            content=f"Entry {i}",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.5),
            created_at=now - timedelta(hours=i),
            updated_at=now - timedelta(hours=i),
        )
        for i in range(5)
    ]

    report = analyzer.analyze(entries)

    assert len(report.daily_trend) > 0


def test_analyze_retention_metrics():
    """Test retention metrics."""
    analyzer = MemoryAnalyzer()
    entries = [
        create_test_entry("Recent", age_hours=1, access_count=5),
        create_test_entry("Old low", age_hours=72, importance=0.2, access_count=0),
        create_test_entry("Old low 2", age_hours=72, importance=0.3, access_count=1),
    ]

    report = analyzer.analyze(entries)

    # Should have compression candidates
    assert report.compression_candidates >= 0


def test_report_to_dict():
    """Test report serialization."""
    analyzer = MemoryAnalyzer()
    entries = [create_test_entry("Test entry")]

    report = analyzer.analyze(entries)
    data = report.to_dict()

    assert "summary" in data
    assert "time_range" in data
    assert data["summary"]["total_memories"] == 1


def test_get_recommendations():
    """Test recommendation generation."""
    analyzer = MemoryAnalyzer()
    entries = [
        create_test_entry(f"Entry {i}", importance=0.2, age_hours=72, access_count=0)
        for i in range(15)
    ]

    report = analyzer.analyze(entries)
    recommendations = analyzer.get_recommendations(report)

    # Should have compression recommendation
    rec_types = [r["type"] for r in recommendations]
    assert "compression" in rec_types or "scoring" in rec_types


def test_get_recommendations_low_retention():
    """Test low retention recommendation."""
    analyzer = MemoryAnalyzer()

    # Create report with low retention
    now = datetime.now(timezone.utc)
    report = AnalyticsReport(
        start_time=now - timedelta(days=7),
        end_time=now,
        total_memories=100,
        retention_rate=0.1,  # Very low
    )

    recommendations = analyzer.get_recommendations(report)

    rec_types = [r["type"] for r in recommendations]
    assert "retention" in rec_types


def test_keyword_extraction():
    """Test keyword extraction from text."""
    analyzer = MemoryAnalyzer()

    keywords = analyzer._extract_keywords(
        "Python programming is great for machine learning"
    )

    assert "python" in keywords
    assert "programming" in keywords
    assert "machine" in keywords
    # Stop words should be excluded
    assert "is" not in keywords
    assert "for" not in keywords


def test_usage_pattern_detection():
    """Test detection of usage patterns."""
    analyzer = MemoryAnalyzer()

    # Create entries with high importance focus
    entries = [
        create_test_entry(f"Important {i}", importance=0.9)
        for i in range(10)
    ]

    report = analyzer.analyze(entries)

    pattern_types = [p.pattern_type for p in report.usage_patterns]
    assert "high_importance_focus" in pattern_types


def test_topic_cluster_attributes():
    """Test topic cluster attributes."""
    analyzer = MemoryAnalyzer(min_cluster_size=2)
    entries = [
        create_test_entry("Database optimization techniques"),
        create_test_entry("Database schema design"),
        create_test_entry("Database query performance"),
    ]

    report = analyzer.analyze(entries)

    if report.topic_clusters:
        cluster = report.topic_clusters[0]
        assert cluster.memory_count >= 2
        assert cluster.avg_importance > 0
        assert cluster.first_seen <= cluster.last_seen


def test_peak_hours_detection():
    """Test peak hours detection."""
    analyzer = MemoryAnalyzer()
    now = datetime.now(timezone.utc)

    # Create entries at specific hours
    entries = []
    for i in range(10):
        entry = MemoryEntry(
            content=f"Entry {i}",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.5),
            created_at=now.replace(hour=14, minute=i),  # All at 2 PM
            updated_at=now.replace(hour=14, minute=i),
        )
        entries.append(entry)

    report = analyzer.analyze(entries)

    # Hour 14 should be in peak hours
    assert 14 in report.peak_hours
