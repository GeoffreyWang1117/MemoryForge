"""Automatic memory consolidation scheduler."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Awaitable, Callable
from uuid import UUID, uuid4

import structlog

from memoryforge.compression import CompressionPipeline, CompressionConfig
from memoryforge.core.types import MemoryEntry

logger = structlog.get_logger()


class TaskStatus(Enum):
    """Status of a scheduled task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """A scheduled consolidation task."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "scheduled_at": self.scheduled_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class SchedulerConfig:
    """Configuration for the consolidation scheduler."""

    # Enable automatic scheduling
    enabled: bool = True

    # Consolidation interval in minutes
    interval_minutes: int = 60

    # Minimum memories before consolidation triggers
    min_memories_threshold: int = 50

    # Target memory count after consolidation
    target_memory_count: int = 30

    # Age threshold for consolidation eligibility (hours)
    min_age_hours: float = 24.0

    # Maximum concurrent tasks
    max_concurrent_tasks: int = 1

    # Retain task history count
    history_limit: int = 100


# Type for memory provider function
MemoryProvider = Callable[[], Awaitable[list[MemoryEntry]]]
MemoryUpdater = Callable[[list[MemoryEntry], list[UUID]], Awaitable[None]]


class ConsolidationScheduler:
    """Automatic memory consolidation scheduler.

    Provides:
    - Periodic consolidation of old memories
    - Threshold-based triggering
    - Task history and monitoring
    - Configurable compression settings
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        compression_config: CompressionConfig | None = None,
    ):
        """Initialize the scheduler.

        Args:
            config: Scheduler configuration
            compression_config: Compression configuration
        """
        self._config = config or SchedulerConfig()
        self._compression_config = compression_config or CompressionConfig()
        self._pipeline = CompressionPipeline(self._compression_config)

        self._running = False
        self._task: asyncio.Task | None = None
        self._history: list[ScheduledTask] = []
        self._current_task: ScheduledTask | None = None

        # Callbacks
        self._memory_provider: MemoryProvider | None = None
        self._memory_updater: MemoryUpdater | None = None

        # Statistics
        self._total_runs = 0
        self._total_consolidated = 0
        self._total_removed = 0

    def set_memory_provider(self, provider: MemoryProvider) -> None:
        """Set the function to get current memories.

        Args:
            provider: Async function returning list of MemoryEntry
        """
        self._memory_provider = provider

    def set_memory_updater(self, updater: MemoryUpdater) -> None:
        """Set the function to update memories after consolidation.

        Args:
            updater: Async function taking (new_entries, removed_ids)
        """
        self._memory_updater = updater

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        if not self._memory_provider:
            raise ValueError("Memory provider not set")

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Consolidation scheduler started",
            interval_minutes=self._config.interval_minutes,
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Consolidation scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_consolidate()
                await asyncio.sleep(self._config.interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry

    async def _check_and_consolidate(self) -> None:
        """Check conditions and run consolidation if needed."""
        if not self._config.enabled:
            return

        # Get current memories
        memories = await self._memory_provider()

        # Check threshold
        if len(memories) < self._config.min_memories_threshold:
            logger.debug(
                "Below threshold, skipping consolidation",
                current=len(memories),
                threshold=self._config.min_memories_threshold,
            )
            return

        # Run consolidation
        await self.run_consolidation(memories)

    async def run_consolidation(
        self,
        memories: list[MemoryEntry] | None = None,
    ) -> ScheduledTask:
        """Run a consolidation task manually.

        Args:
            memories: Optional list of memories (fetched if not provided)

        Returns:
            The completed task
        """
        task = ScheduledTask(
            name="consolidation",
            scheduled_at=datetime.now(timezone.utc),
        )
        self._current_task = task

        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)

            # Get memories if not provided
            if memories is None:
                if not self._memory_provider:
                    raise ValueError("Memory provider not set")
                memories = await self._memory_provider()

            original_count = len(memories)

            # Run compression
            from memoryforge.compression import SummaryCompressor, TimeWindowCompressor

            self._pipeline.add_strategy(SummaryCompressor())
            self._pipeline.add_strategy(TimeWindowCompressor())

            result = await self._pipeline.auto_compress(
                memories,
                max_entries=self._config.target_memory_count,
            )

            # Update memories if updater is set
            if self._memory_updater and result.compressed_entries:
                await self._memory_updater(
                    result.compressed_entries,
                    result.removed_ids,
                )

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.result = {
                "original_count": original_count,
                "compressed_count": result.compressed_count,
                "removed_count": len(result.removed_ids),
                "compression_ratio": result.compression_ratio,
                "token_savings": result.token_savings,
            }

            # Update statistics
            self._total_runs += 1
            self._total_consolidated += result.compressed_count
            self._total_removed += len(result.removed_ids)

            logger.info(
                "Consolidation completed",
                original=original_count,
                compressed=result.compressed_count,
                removed=len(result.removed_ids),
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            task.error = str(e)
            logger.error("Consolidation failed", error=str(e))

        # Store in history
        self._history.append(task)
        if len(self._history) > self._config.history_limit:
            self._history = self._history[-self._config.history_limit :]

        self._current_task = None
        return task

    async def trigger_consolidation(self) -> ScheduledTask:
        """Manually trigger a consolidation run.

        Returns:
            The consolidation task
        """
        return await self.run_consolidation()

    def get_history(self, limit: int = 20) -> list[dict]:
        """Get recent task history.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of task dictionaries
        """
        tasks = self._history[-limit:]
        return [t.to_dict() for t in reversed(tasks)]

    def get_current_task(self) -> dict | None:
        """Get the currently running task if any."""
        if self._current_task:
            return self._current_task.to_dict()
        return None

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "interval_minutes": self._config.interval_minutes,
            "threshold": self._config.min_memories_threshold,
            "target_count": self._config.target_memory_count,
            "total_runs": self._total_runs,
            "total_consolidated": self._total_consolidated,
            "total_removed": self._total_removed,
            "history_count": len(self._history),
        }

    def update_config(self, **kwargs) -> None:
        """Update scheduler configuration.

        Args:
            **kwargs: Configuration fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Scheduler config updated: {key}={value}")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    @property
    def config(self) -> SchedulerConfig:
        """Get current configuration."""
        return self._config


class MemoryMaintenanceScheduler:
    """Extended scheduler with multiple maintenance tasks.

    Includes:
    - Memory consolidation
    - Cache cleanup
    - Statistics aggregation
    - Health checks
    """

    def __init__(
        self,
        consolidation_scheduler: ConsolidationScheduler | None = None,
    ):
        """Initialize maintenance scheduler.

        Args:
            consolidation_scheduler: Optional consolidation scheduler instance
        """
        self._consolidation = consolidation_scheduler or ConsolidationScheduler()
        self._running = False
        self._tasks: dict[str, asyncio.Task] = {}
        self._maintenance_callbacks: dict[str, Callable[[], Awaitable[None]]] = {}

    def register_maintenance_task(
        self,
        name: str,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """Register a custom maintenance task.

        Args:
            name: Task name
            callback: Async callback function
        """
        self._maintenance_callbacks[name] = callback
        logger.info(f"Registered maintenance task: {name}")

    async def start(self) -> None:
        """Start all maintenance tasks."""
        self._running = True
        await self._consolidation.start()
        logger.info("Maintenance scheduler started")

    async def stop(self) -> None:
        """Stop all maintenance tasks."""
        self._running = False
        await self._consolidation.stop()

        for task in self._tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("Maintenance scheduler stopped")

    async def run_maintenance_task(self, name: str) -> bool:
        """Run a specific maintenance task.

        Args:
            name: Task name

        Returns:
            True if task ran successfully
        """
        callback = self._maintenance_callbacks.get(name)
        if not callback:
            logger.warning(f"Unknown maintenance task: {name}")
            return False

        try:
            await callback()
            logger.info(f"Maintenance task completed: {name}")
            return True
        except Exception as e:
            logger.error(f"Maintenance task failed: {name}", error=str(e))
            return False

    def get_status(self) -> dict:
        """Get status of all maintenance tasks."""
        return {
            "running": self._running,
            "consolidation": self._consolidation.get_stats(),
            "registered_tasks": list(self._maintenance_callbacks.keys()),
        }
