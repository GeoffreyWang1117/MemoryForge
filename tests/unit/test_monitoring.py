"""Tests for metrics and monitoring functionality."""

import pytest
import asyncio
from datetime import datetime, timezone

from memoryforge.monitoring import (
    MetricsCollector,
    MetricsConfig,
    Counter,
    Gauge,
    Histogram,
    Timer,
    HealthChecker,
    HealthStatus,
    ComponentHealth,
)
from memoryforge.monitoring.metrics import get_metrics, inc_counter, set_gauge, observe
from memoryforge.monitoring.health import (
    create_database_check,
    create_dependency_check,
    create_storage_check,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_basic_increment(self):
        """Test basic counter increment."""
        counter = Counter("test_counter")

        counter.inc()

        assert counter.get() == 1.0

    def test_increment_by_value(self):
        """Test increment by specific value."""
        counter = Counter("test_counter")

        counter.inc(5.0)

        assert counter.get() == 5.0

    def test_multiple_increments(self):
        """Test multiple increments."""
        counter = Counter("test_counter")

        counter.inc(1.0)
        counter.inc(2.0)
        counter.inc(3.0)

        assert counter.get() == 6.0

    def test_labels(self):
        """Test counter with labels."""
        counter = Counter("test_counter")

        counter.inc(1.0, method="GET")
        counter.inc(2.0, method="POST")
        counter.inc(1.0, method="GET")

        assert counter.get(method="GET") == 2.0
        assert counter.get(method="POST") == 2.0

    def test_reset(self):
        """Test counter reset."""
        counter = Counter("test_counter")
        counter.inc(10.0)

        counter.reset()

        assert counter.get() == 0.0

    def test_collect(self):
        """Test counter collection."""
        counter = Counter("test_counter", "Test description")
        counter.inc(5.0, label="a")
        counter.inc(3.0, label="b")

        samples = counter.collect()

        assert len(samples) == 2
        assert all(s["type"] == "counter" for s in samples)


class TestGauge:
    """Tests for Gauge metric."""

    def test_set_value(self):
        """Test setting gauge value."""
        gauge = Gauge("test_gauge")

        gauge.set(42.0)

        assert gauge.get() == 42.0

    def test_increment(self):
        """Test gauge increment."""
        gauge = Gauge("test_gauge")
        gauge.set(10.0)

        gauge.inc(5.0)

        assert gauge.get() == 15.0

    def test_decrement(self):
        """Test gauge decrement."""
        gauge = Gauge("test_gauge")
        gauge.set(10.0)

        gauge.dec(3.0)

        assert gauge.get() == 7.0

    def test_labels(self):
        """Test gauge with labels."""
        gauge = Gauge("test_gauge")

        gauge.set(100.0, instance="a")
        gauge.set(200.0, instance="b")

        assert gauge.get(instance="a") == 100.0
        assert gauge.get(instance="b") == 200.0

    def test_reset(self):
        """Test gauge reset."""
        gauge = Gauge("test_gauge")
        gauge.set(100.0)

        gauge.reset()

        assert gauge.get() == 0.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_basic_observation(self):
        """Test basic histogram observation."""
        histogram = Histogram("test_histogram", buckets=[0.1, 0.5, 1.0])

        histogram.observe(0.3)

        stats = histogram.get_stats()
        assert stats["count"] == 1
        assert stats["sum"] == 0.3

    def test_multiple_observations(self):
        """Test multiple observations."""
        histogram = Histogram("test_histogram", buckets=[0.1, 0.5, 1.0])

        histogram.observe(0.1)
        histogram.observe(0.2)
        histogram.observe(0.3)

        stats = histogram.get_stats()
        assert stats["count"] == 3
        assert abs(stats["sum"] - 0.6) < 0.001  # Float tolerance

    def test_labels(self):
        """Test histogram with labels."""
        histogram = Histogram("test_histogram")

        histogram.observe(0.1, endpoint="/api")
        histogram.observe(0.2, endpoint="/api")
        histogram.observe(0.5, endpoint="/health")

        api_stats = histogram.get_stats(endpoint="/api")
        health_stats = histogram.get_stats(endpoint="/health")

        assert api_stats["count"] == 2
        assert health_stats["count"] == 1

    def test_collect(self):
        """Test histogram collection."""
        histogram = Histogram("test_histogram", buckets=[0.1, 0.5, 1.0])
        histogram.observe(0.3)
        histogram.observe(0.7)

        samples = histogram.collect()

        assert len(samples) == 1
        assert samples[0]["count"] == 2
        assert "buckets" in samples[0]


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_records_duration(self):
        """Test that timer records duration."""
        histogram = Histogram("test_duration")

        with Timer(histogram):
            import time
            time.sleep(0.01)

        stats = histogram.get_stats()
        assert stats["count"] == 1
        assert stats["sum"] > 0.01

    def test_timer_with_labels(self):
        """Test timer with labels."""
        histogram = Histogram("test_duration")

        with Timer(histogram, operation="test"):
            pass

        stats = histogram.get_stats(operation="test")
        assert stats["count"] == 1


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def setup_method(self):
        """Reset singleton before each test."""
        MetricsCollector.reset_instance()

    def test_singleton(self):
        """Test singleton pattern."""
        collector1 = MetricsCollector.get_instance()
        collector2 = MetricsCollector.get_instance()

        assert collector1 is collector2

    def test_register_counter(self):
        """Test counter registration."""
        collector = MetricsCollector()

        counter = collector.register_counter("test_counter", "Test")

        assert counter is not None
        assert collector.counter("test_counter") is counter

    def test_register_gauge(self):
        """Test gauge registration."""
        collector = MetricsCollector()

        gauge = collector.register_gauge("test_gauge", "Test")

        assert gauge is not None
        assert collector.gauge("test_gauge") is gauge

    def test_register_histogram(self):
        """Test histogram registration."""
        collector = MetricsCollector()

        histogram = collector.register_histogram("test_histogram", "Test")

        assert histogram is not None
        assert collector.histogram("test_histogram") is histogram

    def test_inc_counter(self):
        """Test counter increment via collector."""
        collector = MetricsCollector()
        collector.register_counter("test_counter")

        collector.inc_counter("test_counter", 5.0, label="test")

        counter = collector.counter("test_counter")
        assert counter.get(label="test") == 5.0

    def test_set_gauge(self):
        """Test gauge set via collector."""
        collector = MetricsCollector()
        collector.register_gauge("test_gauge")

        collector.set_gauge("test_gauge", 42.0)

        gauge = collector.gauge("test_gauge")
        assert gauge.get() == 42.0

    def test_observe(self):
        """Test histogram observation via collector."""
        collector = MetricsCollector()
        collector.register_histogram("test_histogram")

        collector.observe("test_histogram", 0.5)

        histogram = collector.histogram("test_histogram")
        assert histogram.get_stats()["count"] == 1

    def test_collect_all(self):
        """Test collecting all metrics."""
        collector = MetricsCollector()
        collector.register_counter("test_counter")
        collector.register_gauge("test_gauge")
        collector.inc_counter("test_counter", 1.0)
        collector.set_gauge("test_gauge", 10.0)

        data = collector.collect_all()

        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data
        assert "timestamp" in data

    def test_export_prometheus(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()
        collector.register_counter("test_requests_total", "Total requests")
        collector.inc_counter("test_requests_total", 100.0, method="GET")

        output = collector.export_prometheus()

        assert "test_requests_total" in output
        assert "# HELP" in output
        assert "# TYPE" in output

    def test_reset_all(self):
        """Test resetting all metrics."""
        collector = MetricsCollector()
        collector.register_counter("test_counter")
        collector.inc_counter("test_counter", 100.0)

        collector.reset_all()

        counter = collector.counter("test_counter")
        assert counter.get() == 0.0

    def test_default_metrics_registered(self):
        """Test default metrics are registered."""
        config = MetricsConfig(prefix="test")
        collector = MetricsCollector(config)

        assert collector.counter("test_requests_total") is not None
        assert collector.gauge("test_memories_count") is not None

    def test_get_stats(self):
        """Test getting collector statistics."""
        collector = MetricsCollector()

        stats = collector.get_stats()

        assert "counters" in stats
        assert "gauges" in stats
        assert "histograms" in stats


class TestHealthChecker:
    """Tests for HealthChecker."""

    @pytest.mark.asyncio
    async def test_check_component(self):
        """Test single component health check."""
        checker = HealthChecker()

        async def test_check():
            return HealthStatus.HEALTHY, "All good", {"key": "value"}

        checker.register_check("test", test_check)

        result = await checker.check_component("test")

        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_check_all(self):
        """Test checking all components."""
        checker = HealthChecker()

        async def healthy_check():
            return HealthStatus.HEALTHY, "OK", {}

        async def degraded_check():
            return HealthStatus.DEGRADED, "Slow", {}

        checker.register_check("healthy", healthy_check)
        checker.register_check("degraded", degraded_check)

        result = await checker.check_all()

        assert result.status == HealthStatus.DEGRADED
        assert len(result.components) >= 2

    @pytest.mark.asyncio
    async def test_unhealthy_component(self):
        """Test unhealthy component affects overall status."""
        checker = HealthChecker()

        async def healthy_check():
            return HealthStatus.HEALTHY, "OK", {}

        async def unhealthy_check():
            return HealthStatus.UNHEALTHY, "Failed", {}

        checker.register_check("healthy", healthy_check)
        checker.register_check("unhealthy", unhealthy_check)

        result = await checker.check_all()

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for slow checks."""
        checker = HealthChecker(timeout_seconds=0.1)

        async def slow_check():
            await asyncio.sleep(1.0)
            return HealthStatus.HEALTHY, "OK", {}

        checker.register_check("slow", slow_check)

        result = await checker.check_component("slow")

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test exception handling in checks."""
        checker = HealthChecker()

        async def failing_check():
            raise ValueError("Test error")

        checker.register_check("failing", failing_check)

        result = await checker.check_component("failing")

        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_is_healthy(self):
        """Test is_healthy convenience method."""
        checker = HealthChecker()

        # Default checks should be healthy
        is_healthy = await checker.is_healthy()

        assert isinstance(is_healthy, bool)

    @pytest.mark.asyncio
    async def test_is_ready(self):
        """Test is_ready convenience method."""
        checker = HealthChecker()

        async def degraded_check():
            return HealthStatus.DEGRADED, "Slow", {}

        checker.register_check("service", degraded_check)

        # Degraded should still be ready
        is_ready = await checker.is_ready()

        assert is_ready is True

    def test_register_unregister(self):
        """Test registering and unregistering checks."""
        checker = HealthChecker()

        async def test_check():
            return HealthStatus.HEALTHY, "OK", {}

        checker.register_check("test", test_check)
        assert "test" in checker.get_registered_checks()

        removed = checker.unregister_check("test")
        assert removed is True
        assert "test" not in checker.get_registered_checks()

    @pytest.mark.asyncio
    async def test_to_dict(self):
        """Test health result serialization."""
        checker = HealthChecker()

        result = await checker.check_all()
        data = result.to_dict()

        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "summary" in data


class TestHealthCheckHelpers:
    """Tests for health check helper functions."""

    @pytest.mark.asyncio
    async def test_database_check_healthy(self):
        """Test database health check when healthy."""
        async def check_connection():
            return True

        check = create_database_check(check_connection)
        status, message, metadata = await check()

        assert status == HealthStatus.HEALTHY
        assert "connected" in message.lower()

    @pytest.mark.asyncio
    async def test_database_check_unhealthy(self):
        """Test database health check when unhealthy."""
        async def check_connection():
            return False

        check = create_database_check(check_connection)
        status, message, metadata = await check()

        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_dependency_check(self):
        """Test dependency health check."""
        async def check_dependency():
            return True

        check = create_dependency_check("Redis", check_dependency)
        status, message, metadata = await check()

        assert status == HealthStatus.HEALTHY
        assert "Redis" in message

    @pytest.mark.asyncio
    async def test_storage_check(self):
        """Test storage health check."""
        import tempfile

        check = create_storage_check(tempfile.gettempdir(), min_free_mb=1)
        status, message, metadata = await check()

        # Should be healthy on most systems
        assert status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        assert "free_mb" in metadata


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_to_dict(self):
        """Test ComponentHealth serialization."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            response_time_ms=10.5,
            metadata={"key": "value"},
        )

        data = health.to_dict()

        assert data["name"] == "test"
        assert data["status"] == "healthy"
        assert data["message"] == "All good"
        assert data["response_time_ms"] == 10.5


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MetricsConfig()

        assert config.enabled is True
        assert config.prefix == "memoryforge"
        assert len(config.latency_buckets) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = MetricsConfig(
            prefix="custom",
            enabled=False,
        )

        assert config.prefix == "custom"
        assert config.enabled is False


class TestGlobalMetricsFunctions:
    """Tests for global metrics convenience functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        MetricsCollector.reset_instance()

    def test_get_metrics(self):
        """Test getting global metrics."""
        metrics = get_metrics()

        assert metrics is not None
        assert isinstance(metrics, MetricsCollector)

    def test_inc_counter_global(self):
        """Test global counter increment."""
        metrics = get_metrics()
        metrics.register_counter("test_global_counter")

        inc_counter("test_global_counter", 5.0)

        counter = metrics.counter("test_global_counter")
        assert counter.get() == 5.0

    def test_set_gauge_global(self):
        """Test global gauge set."""
        metrics = get_metrics()
        metrics.register_gauge("test_global_gauge")

        set_gauge("test_global_gauge", 42.0)

        gauge = metrics.gauge("test_global_gauge")
        assert gauge.get() == 42.0

    def test_observe_global(self):
        """Test global histogram observation."""
        metrics = get_metrics()
        metrics.register_histogram("test_global_histogram")

        observe("test_global_histogram", 0.5)

        histogram = metrics.histogram("test_global_histogram")
        assert histogram.get_stats()["count"] == 1
