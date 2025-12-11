"""Memory event hooks and callbacks system."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

import structlog

from memoryforge.core.types import MemoryEntry, MemoryQuery, MemoryResult

logger = structlog.get_logger()


class EventType(Enum):
    """Types of memory events."""

    # Memory lifecycle events
    MEMORY_CREATED = "memory.created"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_DELETED = "memory.deleted"
    MEMORY_ACCESSED = "memory.accessed"

    # Query events
    QUERY_STARTED = "query.started"
    QUERY_COMPLETED = "query.completed"

    # Consolidation events
    CONSOLIDATION_STARTED = "consolidation.started"
    CONSOLIDATION_COMPLETED = "consolidation.completed"

    # Session events
    SESSION_CREATED = "session.created"
    SESSION_ENDED = "session.ended"

    # System events
    CACHE_CLEARED = "cache.cleared"
    THRESHOLD_REACHED = "threshold.reached"
    ERROR_OCCURRED = "error.occurred"


class HookPriority(Enum):
    """Priority levels for hooks."""

    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class MemoryEvent:
    """A memory system event."""

    id: UUID = field(default_factory=uuid4)
    type: EventType = EventType.MEMORY_CREATED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict = field(default_factory=dict)
    source: str = ""

    # For modification hooks
    cancelled: bool = False
    modified_data: dict | None = None

    def cancel(self) -> None:
        """Cancel the event (for pre-hooks)."""
        self.cancelled = True

    def modify(self, **kwargs) -> None:
        """Modify event data (for pre-hooks)."""
        if self.modified_data is None:
            self.modified_data = {}
        self.modified_data.update(kwargs)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "cancelled": self.cancelled,
        }


# Hook callback types
SyncHookCallback = Callable[[MemoryEvent], None]
AsyncHookCallback = Callable[[MemoryEvent], Awaitable[None]]
HookCallback = SyncHookCallback | AsyncHookCallback


@dataclass
class Hook:
    """A registered hook."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    event_type: EventType = EventType.MEMORY_CREATED
    callback: HookCallback = field(default=lambda e: None)
    priority: HookPriority = HookPriority.NORMAL
    enabled: bool = True
    is_async: bool = False

    # Statistics
    call_count: int = 0
    error_count: int = 0
    last_called: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "event_type": self.event_type.value,
            "priority": self.priority.value,
            "enabled": self.enabled,
            "is_async": self.is_async,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "last_called": self.last_called.isoformat() if self.last_called else None,
        }


class HookRegistry:
    """Registry for memory event hooks.

    Provides:
    - Event subscription and unsubscription
    - Priority-based execution order
    - Sync and async hook support
    - Pre and post event hooks
    - Event cancellation and modification
    """

    def __init__(self):
        """Initialize the hook registry."""
        self._hooks: dict[EventType, list[Hook]] = defaultdict(list)
        self._hook_index: dict[UUID, Hook] = {}
        self._event_history: list[MemoryEvent] = []
        self._history_limit = 1000

    def register(
        self,
        event_type: EventType,
        callback: HookCallback,
        name: str = "",
        priority: HookPriority = HookPriority.NORMAL,
    ) -> UUID:
        """Register a hook for an event type.

        Args:
            event_type: Type of event to hook
            callback: Function to call when event occurs
            name: Optional name for the hook
            priority: Execution priority

        Returns:
            Hook ID for later reference
        """
        is_async = asyncio.iscoroutinefunction(callback)

        hook = Hook(
            name=name or f"hook_{event_type.value}",
            event_type=event_type,
            callback=callback,
            priority=priority,
            is_async=is_async,
        )

        self._hooks[event_type].append(hook)
        self._hooks[event_type].sort(key=lambda h: h.priority.value)
        self._hook_index[hook.id] = hook

        logger.debug(
            "Hook registered",
            name=hook.name,
            event_type=event_type.value,
            priority=priority.value,
        )

        return hook.id

    def unregister(self, hook_id: UUID) -> bool:
        """Unregister a hook by ID.

        Args:
            hook_id: ID of the hook to remove

        Returns:
            True if hook was found and removed
        """
        hook = self._hook_index.pop(hook_id, None)
        if hook is None:
            return False

        self._hooks[hook.event_type] = [
            h for h in self._hooks[hook.event_type] if h.id != hook_id
        ]

        logger.debug("Hook unregistered", hook_id=str(hook_id))
        return True

    def enable(self, hook_id: UUID) -> bool:
        """Enable a hook."""
        hook = self._hook_index.get(hook_id)
        if hook:
            hook.enabled = True
            return True
        return False

    def disable(self, hook_id: UUID) -> bool:
        """Disable a hook."""
        hook = self._hook_index.get(hook_id)
        if hook:
            hook.enabled = False
            return True
        return False

    async def emit(
        self,
        event_type: EventType,
        data: dict | None = None,
        source: str = "",
    ) -> MemoryEvent:
        """Emit an event and call all registered hooks.

        Args:
            event_type: Type of event
            data: Event data
            source: Source of the event

        Returns:
            The event object (may be modified by hooks)
        """
        event = MemoryEvent(
            type=event_type,
            data=data or {},
            source=source,
        )

        hooks = self._hooks.get(event_type, [])

        for hook in hooks:
            if not hook.enabled:
                continue

            try:
                if hook.is_async:
                    await hook.callback(event)
                else:
                    hook.callback(event)

                hook.call_count += 1
                hook.last_called = datetime.now(timezone.utc)

                # Check if event was cancelled
                if event.cancelled:
                    logger.debug(
                        "Event cancelled by hook",
                        event_type=event_type.value,
                        hook=hook.name,
                    )
                    break

            except Exception as e:
                hook.error_count += 1
                logger.error(
                    "Hook error",
                    hook=hook.name,
                    event_type=event_type.value,
                    error=str(e),
                )

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._history_limit:
            self._event_history = self._event_history[-self._history_limit :]

        return event

    def emit_sync(
        self,
        event_type: EventType,
        data: dict | None = None,
        source: str = "",
    ) -> MemoryEvent:
        """Emit an event synchronously (only calls sync hooks).

        Args:
            event_type: Type of event
            data: Event data
            source: Source of the event

        Returns:
            The event object
        """
        event = MemoryEvent(
            type=event_type,
            data=data or {},
            source=source,
        )

        hooks = self._hooks.get(event_type, [])

        for hook in hooks:
            if not hook.enabled or hook.is_async:
                continue

            try:
                hook.callback(event)
                hook.call_count += 1
                hook.last_called = datetime.now(timezone.utc)

                if event.cancelled:
                    break

            except Exception as e:
                hook.error_count += 1
                logger.error("Hook error", hook=hook.name, error=str(e))

        self._event_history.append(event)
        if len(self._event_history) > self._history_limit:
            self._event_history = self._event_history[-self._history_limit :]

        return event

    def get_hooks(self, event_type: EventType | None = None) -> list[dict]:
        """Get registered hooks.

        Args:
            event_type: Filter by event type (None for all)

        Returns:
            List of hook dictionaries
        """
        if event_type:
            return [h.to_dict() for h in self._hooks.get(event_type, [])]

        all_hooks = []
        for hooks in self._hooks.values():
            all_hooks.extend(h.to_dict() for h in hooks)
        return all_hooks

    def get_hook(self, hook_id: UUID) -> dict | None:
        """Get a specific hook by ID."""
        hook = self._hook_index.get(hook_id)
        if hook:
            return hook.to_dict()
        return None

    def get_event_history(
        self,
        event_type: EventType | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent event history.

        Args:
            event_type: Filter by event type
            limit: Maximum events to return

        Returns:
            List of event dictionaries
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.type == event_type]
        return [e.to_dict() for e in events[-limit:]]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def get_stats(self) -> dict:
        """Get hook registry statistics."""
        total_hooks = sum(len(hooks) for hooks in self._hooks.values())
        enabled = sum(
            1 for hooks in self._hooks.values()
            for h in hooks if h.enabled
        )
        total_calls = sum(
            h.call_count for hooks in self._hooks.values()
            for h in hooks
        )
        total_errors = sum(
            h.error_count for hooks in self._hooks.values()
            for h in hooks
        )

        return {
            "total_hooks": total_hooks,
            "enabled_hooks": enabled,
            "disabled_hooks": total_hooks - enabled,
            "event_types_with_hooks": len(self._hooks),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "history_size": len(self._event_history),
        }


# Decorator for registering hooks
def on_event(
    registry: HookRegistry,
    event_type: EventType,
    priority: HookPriority = HookPriority.NORMAL,
    name: str = "",
):
    """Decorator for registering a function as an event hook.

    Usage:
        @on_event(registry, EventType.MEMORY_CREATED)
        async def handle_memory_created(event: MemoryEvent):
            print(f"Memory created: {event.data}")
    """

    def decorator(func: HookCallback):
        registry.register(
            event_type=event_type,
            callback=func,
            name=name or func.__name__,
            priority=priority,
        )
        return func

    return decorator


# Global default registry
_default_registry: HookRegistry | None = None


def get_default_registry() -> HookRegistry:
    """Get or create the default hook registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = HookRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry."""
    global _default_registry
    _default_registry = None
