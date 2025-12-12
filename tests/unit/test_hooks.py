"""Unit tests for hooks and events system."""

import asyncio
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from memoryforge.hooks.events import (
    EventType,
    HookPriority,
    MemoryEvent,
    Hook,
    HookRegistry,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_memory_events_defined(self):
        """Test memory lifecycle events are defined."""
        assert EventType.MEMORY_CREATED.value == "memory.created"
        assert EventType.MEMORY_UPDATED.value == "memory.updated"
        assert EventType.MEMORY_DELETED.value == "memory.deleted"
        assert EventType.MEMORY_ACCESSED.value == "memory.accessed"

    def test_query_events_defined(self):
        """Test query events are defined."""
        assert EventType.QUERY_STARTED.value == "query.started"
        assert EventType.QUERY_COMPLETED.value == "query.completed"

    def test_session_events_defined(self):
        """Test session events are defined."""
        assert EventType.SESSION_CREATED.value == "session.created"
        assert EventType.SESSION_ENDED.value == "session.ended"

    def test_system_events_defined(self):
        """Test system events are defined."""
        assert EventType.ERROR_OCCURRED.value == "error.occurred"
        assert EventType.THRESHOLD_REACHED.value == "threshold.reached"


class TestHookPriority:
    """Tests for HookPriority enum."""

    def test_priority_order(self):
        """Test priority values are ordered correctly."""
        assert HookPriority.HIGHEST.value < HookPriority.HIGH.value
        assert HookPriority.HIGH.value < HookPriority.NORMAL.value
        assert HookPriority.NORMAL.value < HookPriority.LOW.value
        assert HookPriority.LOW.value < HookPriority.LOWEST.value


class TestMemoryEvent:
    """Tests for MemoryEvent dataclass."""

    def test_creation(self):
        """Test creating an event."""
        event = MemoryEvent(
            type=EventType.MEMORY_CREATED,
            data={"content": "test"},
            source="test_module",
        )

        assert event.type == EventType.MEMORY_CREATED
        assert event.data["content"] == "test"
        assert event.source == "test_module"
        assert not event.cancelled

    def test_default_values(self):
        """Test default values."""
        event = MemoryEvent()

        assert event.type == EventType.MEMORY_CREATED
        assert event.data == {}
        assert event.cancelled is False
        assert event.modified_data is None
        assert event.id is not None
        assert event.timestamp is not None

    def test_cancel(self):
        """Test cancelling an event."""
        event = MemoryEvent()
        assert not event.cancelled

        event.cancel()
        assert event.cancelled

    def test_modify(self):
        """Test modifying event data."""
        event = MemoryEvent(data={"original": True})

        event.modify(new_key="new_value")

        assert event.modified_data is not None
        assert event.modified_data["new_key"] == "new_value"

    def test_modify_multiple(self):
        """Test multiple modifications."""
        event = MemoryEvent()

        event.modify(key1="value1")
        event.modify(key2="value2")

        assert event.modified_data["key1"] == "value1"
        assert event.modified_data["key2"] == "value2"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = MemoryEvent(
            type=EventType.QUERY_COMPLETED,
            data={"results": 5},
            source="search",
        )

        result = event.to_dict()

        assert result["type"] == "query.completed"
        assert result["data"]["results"] == 5
        assert result["source"] == "search"
        assert "id" in result
        assert "timestamp" in result


class TestHook:
    """Tests for Hook dataclass."""

    def test_creation(self):
        """Test creating a hook."""
        callback = lambda e: None

        hook = Hook(
            name="test_hook",
            event_type=EventType.MEMORY_CREATED,
            callback=callback,
            priority=HookPriority.HIGH,
        )

        assert hook.name == "test_hook"
        assert hook.event_type == EventType.MEMORY_CREATED
        assert hook.priority == HookPriority.HIGH
        assert hook.enabled is True

    def test_default_values(self):
        """Test default values."""
        hook = Hook()

        assert hook.name == ""
        assert hook.priority == HookPriority.NORMAL
        assert hook.enabled is True
        assert hook.is_async is False
        assert hook.call_count == 0
        assert hook.error_count == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hook = Hook(
            name="my_hook",
            event_type=EventType.SESSION_CREATED,
            priority=HookPriority.LOW,
        )

        result = hook.to_dict()

        assert result["name"] == "my_hook"
        assert result["event_type"] == "session.created"
        assert result["priority"] == HookPriority.LOW.value
        assert "id" in result


class TestHookRegistry:
    """Tests for HookRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a hook registry."""
        return HookRegistry()

    def test_init(self, registry):
        """Test initialization."""
        assert len(registry._hooks) == 0
        assert len(registry._hook_index) == 0

    def test_register_sync_hook(self, registry):
        """Test registering a synchronous hook."""
        called = []

        def callback(event):
            called.append(event)

        hook_id = registry.register(
            EventType.MEMORY_CREATED,
            callback,
            name="sync_hook",
        )

        assert hook_id is not None
        assert hook_id in registry._hook_index

    def test_register_async_hook(self, registry):
        """Test registering an async hook."""
        called = []

        async def callback(event):
            called.append(event)

        hook_id = registry.register(
            EventType.MEMORY_CREATED,
            callback,
            name="async_hook",
        )

        assert hook_id is not None
        # Hook should be marked as async
        hook = registry._hook_index[hook_id]
        assert hook.is_async is True

    def test_register_with_priority(self, registry):
        """Test registering with priority."""
        hook_id = registry.register(
            EventType.MEMORY_CREATED,
            lambda e: None,
            priority=HookPriority.HIGHEST,
        )

        hook = registry._hook_index[hook_id]
        assert hook.priority == HookPriority.HIGHEST

    def test_unregister(self, registry):
        """Test unregistering a hook."""
        hook_id = registry.register(
            EventType.MEMORY_CREATED,
            lambda e: None,
        )

        result = registry.unregister(hook_id)

        assert result is True
        assert hook_id not in registry._hook_index

    def test_unregister_nonexistent(self, registry):
        """Test unregistering non-existent hook."""
        result = registry.unregister(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_emit_sync_hook(self, registry):
        """Test emitting event to sync hook."""
        received = []

        def callback(event):
            received.append(event.type)

        registry.register(EventType.MEMORY_CREATED, callback)

        # emit() takes event_type, not MemoryEvent
        event = await registry.emit(EventType.MEMORY_CREATED)

        assert EventType.MEMORY_CREATED in received

    @pytest.mark.asyncio
    async def test_emit_async_hook(self, registry):
        """Test emitting event to async hook."""
        received = []

        async def callback(event):
            received.append(event.type)

        registry.register(EventType.QUERY_COMPLETED, callback)

        event = await registry.emit(EventType.QUERY_COMPLETED)

        assert EventType.QUERY_COMPLETED in received

    @pytest.mark.asyncio
    async def test_emit_respects_priority(self, registry):
        """Test that hooks are called in priority order."""
        order = []

        registry.register(
            EventType.MEMORY_CREATED,
            lambda e: order.append("low"),
            priority=HookPriority.LOW,
        )
        registry.register(
            EventType.MEMORY_CREATED,
            lambda e: order.append("high"),
            priority=HookPriority.HIGH,
        )
        registry.register(
            EventType.MEMORY_CREATED,
            lambda e: order.append("normal"),
            priority=HookPriority.NORMAL,
        )

        await registry.emit(EventType.MEMORY_CREATED)

        assert order == ["high", "normal", "low"]

    @pytest.mark.asyncio
    async def test_emit_only_matching_type(self, registry):
        """Test that only hooks for matching event type are called."""
        called = []

        registry.register(EventType.MEMORY_CREATED, lambda e: called.append("created"))
        registry.register(EventType.MEMORY_DELETED, lambda e: called.append("deleted"))

        await registry.emit(EventType.MEMORY_CREATED)

        assert called == ["created"]

    @pytest.mark.asyncio
    async def test_emit_disabled_hook(self, registry):
        """Test that disabled hooks are not called."""
        called = []

        hook_id = registry.register(
            EventType.MEMORY_CREATED,
            lambda e: called.append("called"),
        )

        # Disable the hook
        registry.disable(hook_id)

        await registry.emit(EventType.MEMORY_CREATED)

        assert called == []

    @pytest.mark.asyncio
    async def test_emit_cancelled_event(self, registry):
        """Test handling of cancelled events."""
        order = []

        def cancel_hook(event):
            order.append("cancel")
            event.cancel()

        def regular_hook(event):
            order.append("regular")

        registry.register(
            EventType.MEMORY_CREATED,
            cancel_hook,
            priority=HookPriority.HIGHEST,
        )
        registry.register(
            EventType.MEMORY_CREATED,
            regular_hook,
            priority=HookPriority.LOW,
        )

        event = await registry.emit(EventType.MEMORY_CREATED)

        # Cancel hook should be called
        assert "cancel" in order

    def test_enable_disable(self, registry):
        """Test enabling and disabling hooks."""
        hook_id = registry.register(EventType.MEMORY_CREATED, lambda e: None)

        registry.disable(hook_id)
        hook = registry._hook_index[hook_id]
        assert hook.enabled is False

        registry.enable(hook_id)
        assert hook.enabled is True

    def test_get_hooks(self, registry):
        """Test getting hooks by event type."""
        registry.register(EventType.MEMORY_CREATED, lambda e: None, name="hook1")
        registry.register(EventType.MEMORY_CREATED, lambda e: None, name="hook2")
        registry.register(EventType.MEMORY_DELETED, lambda e: None, name="hook3")

        # get_hooks returns list[dict]
        created_hooks = registry.get_hooks(EventType.MEMORY_CREATED)
        assert len(created_hooks) == 2

        deleted_hooks = registry.get_hooks(EventType.MEMORY_DELETED)
        assert len(deleted_hooks) == 1

    def test_get_all_hooks(self, registry):
        """Test getting all registered hooks."""
        registry.register(EventType.MEMORY_CREATED, lambda e: None)
        registry.register(EventType.QUERY_STARTED, lambda e: None)

        # Pass None to get all hooks
        all_hooks = registry.get_hooks(None)
        assert len(all_hooks) == 2

    def test_clear_history(self, registry):
        """Test clearing event history."""
        registry.clear_history()
        # Should not raise, history is cleared
        history = registry.get_event_history()
        assert len(history) == 0

    def test_get_stats(self, registry):
        """Test getting statistics."""
        registry.register(EventType.MEMORY_CREATED, lambda e: None)

        stats = registry.get_stats()

        assert "total_hooks" in stats
        assert "enabled_hooks" in stats
        assert stats["total_hooks"] == 1

    @pytest.mark.asyncio
    async def test_hook_call_count(self, registry):
        """Test that hook call count is updated."""
        hook_id = registry.register(EventType.MEMORY_CREATED, lambda e: None)

        await registry.emit(EventType.MEMORY_CREATED)
        await registry.emit(EventType.MEMORY_CREATED)

        hook = registry._hook_index[hook_id]
        assert hook.call_count == 2

    @pytest.mark.asyncio
    async def test_hook_error_handling(self, registry):
        """Test error handling in hooks."""
        def error_hook(event):
            raise ValueError("Test error")

        hook_id = registry.register(EventType.MEMORY_CREATED, error_hook)

        # Should not raise, error should be caught
        await registry.emit(EventType.MEMORY_CREATED)

        hook = registry._hook_index[hook_id]
        assert hook.error_count == 1


class TestHookRegistryEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def registry(self):
        return HookRegistry()

    @pytest.mark.asyncio
    async def test_emit_no_hooks(self, registry):
        """Test emitting event with no registered hooks."""
        # Should not raise
        event = await registry.emit(EventType.MEMORY_CREATED)
        assert event is not None

    @pytest.mark.asyncio
    async def test_concurrent_emit(self, registry):
        """Test concurrent event emission."""
        results = []

        async def slow_hook(event):
            await asyncio.sleep(0.01)
            results.append(event.data.get("id"))

        registry.register(EventType.MEMORY_CREATED, slow_hook)

        # Emit multiple events concurrently
        await asyncio.gather(*[
            registry.emit(EventType.MEMORY_CREATED, data={"id": i})
            for i in range(5)
        ])

        assert len(results) == 5

    def test_register_same_callback_multiple_times(self, registry):
        """Test registering same callback multiple times."""
        callback = lambda e: None

        id1 = registry.register(EventType.MEMORY_CREATED, callback)
        id2 = registry.register(EventType.MEMORY_CREATED, callback)

        # Should create separate hooks
        assert id1 != id2
        assert len(registry.get_hooks(EventType.MEMORY_CREATED)) == 2

    @pytest.mark.asyncio
    async def test_modify_event_in_hook(self, registry):
        """Test modifying event data in hook."""
        def modifier(event):
            event.modify(processed=True)

        registry.register(EventType.MEMORY_CREATED, modifier)

        event = await registry.emit(EventType.MEMORY_CREATED)

        assert event.modified_data is not None
        assert event.modified_data.get("processed") is True

    @pytest.mark.asyncio
    async def test_emit_with_data(self, registry):
        """Test emitting event with data."""
        received_data = []

        def callback(event):
            received_data.append(event.data)

        registry.register(EventType.MEMORY_CREATED, callback)

        await registry.emit(EventType.MEMORY_CREATED, data={"key": "value"})

        assert len(received_data) == 1
        assert received_data[0]["key"] == "value"

    @pytest.mark.asyncio
    async def test_emit_with_source(self, registry):
        """Test emitting event with source."""
        received_sources = []

        def callback(event):
            received_sources.append(event.source)

        registry.register(EventType.MEMORY_CREATED, callback)

        await registry.emit(EventType.MEMORY_CREATED, source="test_module")

        assert received_sources == ["test_module"]

    def test_get_event_history(self, registry):
        """Test getting event history."""
        history = registry.get_event_history()
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_event_history_populated(self, registry):
        """Test that event history is populated after emit."""
        await registry.emit(EventType.MEMORY_CREATED, data={"test": True})

        history = registry.get_event_history()
        assert len(history) == 1
        assert history[0]["type"] == "memory.created"
