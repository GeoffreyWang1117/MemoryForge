"""Health check system for MemoryForge components."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable

import structlog

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time_ms": round(self.response_time_ms, 2),
            "last_check": self.last_check.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: list[ComponentHealth]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "components": [c.to_dict() for c in self.components],
            "summary": {
                "healthy": sum(1 for c in self.components if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.components if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.components if c.status == HealthStatus.UNHEALTHY),
            },
        }


# Type for health check functions
HealthCheckFunc = Callable[[], Awaitable[tuple[HealthStatus, str, dict]]]


class HealthChecker:
    """System health checker.

    Provides:
    - Component health registration
    - Async health checks
    - Aggregated system status
    - Health history tracking
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        version: str = "1.0.0",
    ):
        """Initialize health checker.

        Args:
            timeout_seconds: Timeout for health checks
            version: Application version
        """
        self._timeout = timeout_seconds
        self._version = version
        self._checks: dict[str, HealthCheckFunc] = {}
        self._last_results: dict[str, ComponentHealth] = {}

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # Memory check
        async def memory_check() -> tuple[HealthStatus, str, dict]:
            import sys
            try:
                # Simple memory check
                import gc
                gc.collect()
                return HealthStatus.HEALTHY, "Memory OK", {"gc_collected": True}
            except Exception as e:
                return HealthStatus.DEGRADED, str(e), {}

        self.register_check("memory", memory_check)

        # Python runtime check
        async def runtime_check() -> tuple[HealthStatus, str, dict]:
            import sys
            import platform
            return HealthStatus.HEALTHY, "Runtime OK", {
                "python_version": sys.version_info[:3],
                "platform": platform.system(),
            }

        self.register_check("runtime", runtime_check)

    def register_check(
        self,
        name: str,
        check_func: HealthCheckFunc,
    ) -> None:
        """Register a health check.

        Args:
            name: Check name
            check_func: Async function returning (status, message, metadata)
        """
        self._checks[name] = check_func
        logger.debug(f"Registered health check: {name}")

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check.

        Args:
            name: Check name

        Returns:
            True if removed
        """
        if name in self._checks:
            del self._checks[name]
            return True
        return False

    async def check_component(self, name: str) -> ComponentHealth:
        """Run a single component health check.

        Args:
            name: Component name

        Returns:
            Component health status
        """
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="No health check registered",
            )

        check_func = self._checks[name]
        start_time = time.perf_counter()

        try:
            status, message, metadata = await asyncio.wait_for(
                check_func(),
                timeout=self._timeout,
            )
            elapsed = (time.perf_counter() - start_time) * 1000

            result = ComponentHealth(
                name=name,
                status=status,
                message=message,
                response_time_ms=elapsed,
                metadata=metadata,
            )

        except asyncio.TimeoutError:
            elapsed = self._timeout * 1000
            result = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self._timeout}s",
                response_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            result = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=elapsed,
            )
            logger.error(f"Health check failed for {name}", error=str(e))

        self._last_results[name] = result
        return result

    async def check_all(self) -> SystemHealth:
        """Run all health checks.

        Returns:
            System health status
        """
        # Run all checks concurrently
        tasks = [
            self.check_component(name)
            for name in self._checks
        ]

        components = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed = []
        for i, result in enumerate(components):
            if isinstance(result, Exception):
                name = list(self._checks.keys())[i]
                processed.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                ))
            else:
                processed.append(result)

        # Determine overall status
        overall = self._calculate_overall_status(processed)

        return SystemHealth(
            status=overall,
            components=processed,
            version=self._version,
        )

    def _calculate_overall_status(
        self,
        components: list[ComponentHealth],
    ) -> HealthStatus:
        """Calculate overall system status from components.

        Args:
            components: Component health results

        Returns:
            Overall status
        """
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY

        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED

        return HealthStatus.UNKNOWN

    def get_last_result(self, name: str) -> ComponentHealth | None:
        """Get last health check result for a component.

        Args:
            name: Component name

        Returns:
            Last health result or None
        """
        return self._last_results.get(name)

    def get_all_last_results(self) -> dict[str, ComponentHealth]:
        """Get all last health check results.

        Returns:
            Dict of component name to health status
        """
        return self._last_results.copy()

    async def is_healthy(self) -> bool:
        """Quick check if system is healthy.

        Returns:
            True if all components healthy
        """
        health = await self.check_all()
        return health.status == HealthStatus.HEALTHY

    async def is_ready(self) -> bool:
        """Check if system is ready to handle requests.

        Returns:
            True if system is ready (healthy or degraded)
        """
        health = await self.check_all()
        return health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def get_registered_checks(self) -> list[str]:
        """Get list of registered check names.

        Returns:
            List of check names
        """
        return list(self._checks.keys())

    def get_stats(self) -> dict:
        """Get health checker statistics.

        Returns:
            Statistics dict
        """
        return {
            "registered_checks": len(self._checks),
            "check_names": list(self._checks.keys()),
            "timeout_seconds": self._timeout,
            "version": self._version,
            "last_results_count": len(self._last_results),
        }


# Helper functions for creating common health checks
def create_database_check(
    check_connection: Callable[[], Awaitable[bool]],
) -> HealthCheckFunc:
    """Create a database health check.

    Args:
        check_connection: Async function to test connection

    Returns:
        Health check function
    """
    async def check() -> tuple[HealthStatus, str, dict]:
        try:
            is_connected = await check_connection()
            if is_connected:
                return HealthStatus.HEALTHY, "Database connected", {}
            else:
                return HealthStatus.UNHEALTHY, "Database not connected", {}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Database error: {e}", {}

    return check


def create_dependency_check(
    name: str,
    check_func: Callable[[], Awaitable[bool]],
) -> HealthCheckFunc:
    """Create a dependency health check.

    Args:
        name: Dependency name
        check_func: Async function to check dependency

    Returns:
        Health check function
    """
    async def check() -> tuple[HealthStatus, str, dict]:
        try:
            is_available = await check_func()
            if is_available:
                return HealthStatus.HEALTHY, f"{name} available", {}
            else:
                return HealthStatus.DEGRADED, f"{name} unavailable", {}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"{name} error: {e}", {}

    return check


def create_storage_check(
    path: str,
    min_free_mb: int = 100,
) -> HealthCheckFunc:
    """Create a storage health check.

    Args:
        path: Path to check
        min_free_mb: Minimum free space in MB

    Returns:
        Health check function
    """
    async def check() -> tuple[HealthStatus, str, dict]:
        import os
        try:
            stat = os.statvfs(path)
            free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)

            metadata = {
                "path": path,
                "free_mb": round(free_mb, 2),
                "min_required_mb": min_free_mb,
            }

            if free_mb >= min_free_mb:
                return HealthStatus.HEALTHY, f"Storage OK ({free_mb:.0f}MB free)", metadata
            elif free_mb >= min_free_mb * 0.5:
                return HealthStatus.DEGRADED, f"Storage low ({free_mb:.0f}MB free)", metadata
            else:
                return HealthStatus.UNHEALTHY, f"Storage critical ({free_mb:.0f}MB free)", metadata

        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Storage check failed: {e}", {"path": path}

    return check
