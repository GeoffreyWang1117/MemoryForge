"""Metrics collection and reporting for MemoryForge."""

import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Generator

import structlog

logger = structlog.get_logger()


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    # Enable metrics collection
    enabled: bool = True

    # Metric prefix
    prefix: str = "memoryforge"

    # Default labels
    default_labels: dict[str, str] = field(default_factory=dict)

    # Histogram buckets for latency
    latency_buckets: list[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])

    # Histogram buckets for sizes
    size_buckets: list[float] = field(default_factory=lambda: [
        100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000
    ])


class Counter:
    """A monotonically increasing counter metric."""

    def __init__(self, name: str, description: str = ""):
        """Initialize counter.

        Args:
            name: Metric name
            description: Metric description
        """
        self._name = name
        self._description = description
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment counter.

        Args:
            value: Amount to increment
            **labels: Label key-value pairs
        """
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] += value

    def get(self, **labels) -> float:
        """Get counter value.

        Args:
            **labels: Label key-value pairs

        Returns:
            Counter value
        """
        label_key = tuple(sorted(labels.items()))
        return self._values.get(label_key, 0.0)

    def reset(self) -> None:
        """Reset all counter values."""
        with self._lock:
            self._values.clear()

    def collect(self) -> list[dict]:
        """Collect all metric values.

        Returns:
            List of metric samples
        """
        samples = []
        with self._lock:
            for label_key, value in self._values.items():
                samples.append({
                    "name": self._name,
                    "type": "counter",
                    "labels": dict(label_key),
                    "value": value,
                })
        return samples


class Gauge:
    """A metric that can go up and down."""

    def __init__(self, name: str, description: str = ""):
        """Initialize gauge.

        Args:
            name: Metric name
            description: Metric description
        """
        self._name = name
        self._description = description
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def set(self, value: float, **labels) -> None:
        """Set gauge value.

        Args:
            value: New value
            **labels: Label key-value pairs
        """
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment gauge.

        Args:
            value: Amount to increment
            **labels: Label key-value pairs
        """
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] += value

    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement gauge.

        Args:
            value: Amount to decrement
            **labels: Label key-value pairs
        """
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] -= value

    def get(self, **labels) -> float:
        """Get gauge value.

        Args:
            **labels: Label key-value pairs

        Returns:
            Gauge value
        """
        label_key = tuple(sorted(labels.items()))
        return self._values.get(label_key, 0.0)

    def reset(self) -> None:
        """Reset all gauge values."""
        with self._lock:
            self._values.clear()

    def collect(self) -> list[dict]:
        """Collect all metric values.

        Returns:
            List of metric samples
        """
        samples = []
        with self._lock:
            for label_key, value in self._values.items():
                samples.append({
                    "name": self._name,
                    "type": "gauge",
                    "labels": dict(label_key),
                    "value": value,
                })
        return samples


class Histogram:
    """A histogram for measuring distributions."""

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: list[float] | None = None,
    ):
        """Initialize histogram.

        Args:
            name: Metric name
            description: Metric description
            buckets: Bucket boundaries
        """
        self._name = name
        self._description = description
        self._buckets = sorted(buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
        self._data: dict[tuple, dict] = defaultdict(
            lambda: {"sum": 0.0, "count": 0, "buckets": defaultdict(int)}
        )
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def observe(self, value: float, **labels) -> None:
        """Record an observation.

        Args:
            value: Observed value
            **labels: Label key-value pairs
        """
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            data = self._data[label_key]
            data["sum"] += value
            data["count"] += 1

            # Update bucket counts
            for bucket in self._buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

    def get_stats(self, **labels) -> dict:
        """Get histogram statistics.

        Args:
            **labels: Label key-value pairs

        Returns:
            Statistics dict with sum, count, mean
        """
        label_key = tuple(sorted(labels.items()))
        data = self._data.get(label_key, {"sum": 0.0, "count": 0})

        return {
            "sum": data["sum"],
            "count": data["count"],
            "mean": data["sum"] / data["count"] if data["count"] > 0 else 0.0,
        }

    def reset(self) -> None:
        """Reset histogram data."""
        with self._lock:
            self._data.clear()

    def collect(self) -> list[dict]:
        """Collect all metric values.

        Returns:
            List of metric samples
        """
        samples = []
        with self._lock:
            for label_key, data in self._data.items():
                samples.append({
                    "name": self._name,
                    "type": "histogram",
                    "labels": dict(label_key),
                    "sum": data["sum"],
                    "count": data["count"],
                    "buckets": dict(data["buckets"]),
                })
        return samples


class Timer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram, **labels):
        """Initialize timer.

        Args:
            histogram: Histogram to record to
            **labels: Label key-value pairs
        """
        self._histogram = histogram
        self._labels = labels
        self._start_time: float = 0.0

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.perf_counter() - self._start_time
        self._histogram.observe(elapsed, **self._labels)


class MetricsCollector:
    """Central metrics collection and management.

    Provides:
    - Metric registration
    - Collection and export
    - Prometheus-compatible output
    """

    _instance: "MetricsCollector | None" = None

    def __init__(self, config: MetricsConfig | None = None):
        """Initialize metrics collector.

        Args:
            config: Metrics configuration
        """
        self._config = config or MetricsConfig()
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

        # Pre-register common metrics
        self._register_default_metrics()

    @classmethod
    def get_instance(cls, config: MetricsConfig | None = None) -> "MetricsCollector":
        """Get singleton instance.

        Args:
            config: Configuration (only used on first call)

        Returns:
            MetricsCollector instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None

    def _register_default_metrics(self) -> None:
        """Register default metrics."""
        prefix = self._config.prefix

        # Request metrics
        self.register_counter(
            f"{prefix}_requests_total",
            "Total number of requests"
        )
        self.register_histogram(
            f"{prefix}_request_duration_seconds",
            "Request duration in seconds",
            buckets=self._config.latency_buckets,
        )

        # Memory metrics
        self.register_counter(
            f"{prefix}_memories_stored_total",
            "Total memories stored"
        )
        self.register_counter(
            f"{prefix}_memories_retrieved_total",
            "Total memories retrieved"
        )
        self.register_gauge(
            f"{prefix}_memories_count",
            "Current memory count"
        )

        # Cache metrics
        self.register_counter(
            f"{prefix}_cache_hits_total",
            "Total cache hits"
        )
        self.register_counter(
            f"{prefix}_cache_misses_total",
            "Total cache misses"
        )

        # Error metrics
        self.register_counter(
            f"{prefix}_errors_total",
            "Total errors"
        )

    def register_counter(self, name: str, description: str = "") -> Counter:
        """Register a counter metric.

        Args:
            name: Metric name
            description: Metric description

        Returns:
            Counter instance
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def register_gauge(self, name: str, description: str = "") -> Gauge:
        """Register a gauge metric.

        Args:
            name: Metric name
            description: Metric description

        Returns:
            Gauge instance
        """
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def register_histogram(
        self,
        name: str,
        description: str = "",
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Register a histogram metric.

        Args:
            name: Metric name
            description: Metric description
            buckets: Histogram buckets

        Returns:
            Histogram instance
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets)
            return self._histograms[name]

    def counter(self, name: str) -> Counter | None:
        """Get a counter by name.

        Args:
            name: Metric name

        Returns:
            Counter or None
        """
        return self._counters.get(name)

    def gauge(self, name: str) -> Gauge | None:
        """Get a gauge by name.

        Args:
            name: Metric name

        Returns:
            Gauge or None
        """
        return self._gauges.get(name)

    def histogram(self, name: str) -> Histogram | None:
        """Get a histogram by name.

        Args:
            name: Metric name

        Returns:
            Histogram or None
        """
        return self._histograms.get(name)

    @contextmanager
    def timer(self, name: str, **labels) -> Generator[Timer, None, None]:
        """Create a timer context manager.

        Args:
            name: Histogram name
            **labels: Labels for the observation

        Yields:
            Timer instance
        """
        histogram = self._histograms.get(name)
        if histogram:
            timer = Timer(histogram, **labels)
            yield timer
        else:
            yield None  # type: ignore

    def inc_counter(self, name: str, value: float = 1.0, **labels) -> None:
        """Increment a counter.

        Args:
            name: Counter name
            value: Increment value
            **labels: Labels
        """
        counter = self._counters.get(name)
        if counter:
            counter.inc(value, **labels)

    def set_gauge(self, name: str, value: float, **labels) -> None:
        """Set a gauge value.

        Args:
            name: Gauge name
            value: New value
            **labels: Labels
        """
        gauge = self._gauges.get(name)
        if gauge:
            gauge.set(value, **labels)

    def observe(self, name: str, value: float, **labels) -> None:
        """Record a histogram observation.

        Args:
            name: Histogram name
            value: Observed value
            **labels: Labels
        """
        histogram = self._histograms.get(name)
        if histogram:
            histogram.observe(value, **labels)

    def collect_all(self) -> dict:
        """Collect all metrics.

        Returns:
            Dict with all metric data
        """
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prefix": self._config.prefix,
            "counters": [],
            "gauges": [],
            "histograms": [],
        }

        for counter in self._counters.values():
            result["counters"].extend(counter.collect())

        for gauge in self._gauges.values():
            result["gauges"].extend(gauge.collect())

        for histogram in self._histograms.values():
            result["histograms"].extend(histogram.collect())

        return result

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Counters
        for counter in self._counters.values():
            lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            for sample in counter.collect():
                labels = self._format_labels(sample["labels"])
                lines.append(f"{sample['name']}{labels} {sample['value']}")

        # Gauges
        for gauge in self._gauges.values():
            lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            for sample in gauge.collect():
                labels = self._format_labels(sample["labels"])
                lines.append(f"{sample['name']}{labels} {sample['value']}")

        # Histograms
        for histogram in self._histograms.values():
            lines.append(f"# HELP {histogram.name} {histogram.description}")
            lines.append(f"# TYPE {histogram.name} histogram")
            for sample in histogram.collect():
                labels = self._format_labels(sample["labels"])
                name = sample["name"]
                lines.append(f"{name}_sum{labels} {sample['sum']}")
                lines.append(f"{name}_count{labels} {sample['count']}")
                for bucket, count in sorted(sample["buckets"].items()):
                    bucket_labels = {**sample["labels"], "le": str(bucket)}
                    bucket_label_str = self._format_labels(bucket_labels)
                    lines.append(f"{name}_bucket{bucket_label_str} {count}")

        return "\n".join(lines)

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus output.

        Args:
            labels: Label dict

        Returns:
            Formatted label string
        """
        if not labels:
            return ""

        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def reset_all(self) -> None:
        """Reset all metrics."""
        for counter in self._counters.values():
            counter.reset()
        for gauge in self._gauges.values():
            gauge.reset()
        for histogram in self._histograms.values():
            histogram.reset()

    def get_stats(self) -> dict:
        """Get metrics collector statistics.

        Returns:
            Statistics dict
        """
        return {
            "enabled": self._config.enabled,
            "prefix": self._config.prefix,
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "histograms": len(self._histograms),
        }


# Convenience functions for global metrics
def get_metrics() -> MetricsCollector:
    """Get the global metrics collector.

    Returns:
        MetricsCollector instance
    """
    return MetricsCollector.get_instance()


def inc_counter(name: str, value: float = 1.0, **labels) -> None:
    """Increment a global counter.

    Args:
        name: Counter name
        value: Increment value
        **labels: Labels
    """
    get_metrics().inc_counter(name, value, **labels)


def set_gauge(name: str, value: float, **labels) -> None:
    """Set a global gauge value.

    Args:
        name: Gauge name
        value: New value
        **labels: Labels
    """
    get_metrics().set_gauge(name, value, **labels)


def observe(name: str, value: float, **labels) -> None:
    """Record a global histogram observation.

    Args:
        name: Histogram name
        value: Observed value
        **labels: Labels
    """
    get_metrics().observe(name, value, **labels)
