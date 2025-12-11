"""Multi-session manager for handling concurrent memory sessions."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import AsyncIterator
from uuid import UUID, uuid4

import structlog

from memoryforge.core.types import MemoryEntry, MemoryQuery, MemoryResult
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.storage.sqlite import SQLiteMemoryStore

logger = structlog.get_logger()


class SessionState(Enum):
    """Session states."""

    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    EXPIRED = "expired"


@dataclass
class SessionConfig:
    """Configuration for a session."""

    # Session name for identification
    name: str = ""

    # Maximum entries in working memory
    max_entries: int = 100

    # Maximum tokens in working memory
    max_tokens: int = 8000

    # Session timeout in minutes (0 = no timeout)
    timeout_minutes: int = 60

    # Auto-save interval in minutes (0 = disabled)
    auto_save_minutes: int = 5

    # Tags to associate with this session
    tags: list[str] = field(default_factory=list)

    # Custom metadata
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    """A memory session with its own context."""

    id: str = field(default_factory=lambda: str(uuid4()))
    config: SessionConfig = field(default_factory=SessionConfig)
    state: SessionState = SessionState.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Working memory for this session
    working_memory: WorkingMemory | None = None

    # Statistics
    message_count: int = 0
    memory_operations: int = 0

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.config.timeout_minutes <= 0:
            return False
        timeout = timedelta(minutes=self.config.timeout_minutes)
        return datetime.now(timezone.utc) - self.last_activity > timeout

    def to_dict(self) -> dict:
        """Convert session to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.config.name,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "memory_operations": self.memory_operations,
            "memory_count": len(self.working_memory.entries) if self.working_memory else 0,
            "config": {
                "max_entries": self.config.max_entries,
                "max_tokens": self.config.max_tokens,
                "timeout_minutes": self.config.timeout_minutes,
                "tags": self.config.tags,
            },
        }


class SessionManager:
    """Manages multiple concurrent memory sessions.

    Provides:
    - Session creation and lifecycle management
    - Session isolation (each session has its own memory space)
    - Session persistence and recovery
    - Automatic session expiration
    - Session switching and context management
    """

    def __init__(
        self,
        store: SQLiteMemoryStore | None = None,
        default_config: SessionConfig | None = None,
        max_sessions: int = 100,
    ):
        """Initialize the session manager.

        Args:
            store: SQLite store for persistence
            default_config: Default configuration for new sessions
            max_sessions: Maximum number of active sessions
        """
        self._sessions: dict[str, Session] = {}
        self._store = store
        self._default_config = default_config or SessionConfig()
        self._max_sessions = max_sessions
        self._current_session_id: str | None = None

    async def create_session(
        self,
        config: SessionConfig | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create a new session.

        Args:
            config: Session configuration (uses default if not provided)
            session_id: Optional custom session ID

        Returns:
            The created session

        Raises:
            ValueError: If max sessions reached
        """
        # Check session limit
        active_count = sum(
            1 for s in self._sessions.values()
            if s.state == SessionState.ACTIVE
        )
        if active_count >= self._max_sessions:
            # Try to clean up expired sessions first
            await self._cleanup_expired()
            active_count = sum(
                1 for s in self._sessions.values()
                if s.state == SessionState.ACTIVE
            )
            if active_count >= self._max_sessions:
                raise ValueError(f"Maximum sessions ({self._max_sessions}) reached")

        # Create session
        session = Session(
            id=session_id or str(uuid4()),
            config=config or SessionConfig(**self._default_config.__dict__),
        )

        # Initialize working memory
        session.working_memory = WorkingMemory(
            max_entries=session.config.max_entries,
            max_tokens=session.config.max_tokens,
        )

        self._sessions[session.id] = session

        # Create in store if available
        if self._store:
            self._store.create_session(session.id, session.config.name)

        logger.info(
            "Session created",
            session_id=session.id,
            name=session.config.name,
        )

        return session

    async def get_session(
        self,
        session_id: str,
        create_if_missing: bool = False,
    ) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session ID to retrieve
            create_if_missing: Create a new session if not found

        Returns:
            The session or None if not found
        """
        session = self._sessions.get(session_id)

        if session:
            # Check if expired
            if session.is_expired():
                session.state = SessionState.EXPIRED
                logger.info("Session expired", session_id=session_id)
                if create_if_missing:
                    return await self.create_session(session_id=session_id)
                return None
            session.touch()
            return session

        # Try to load from store
        if self._store and create_if_missing:
            return await self.create_session(session_id=session_id)

        return None

    async def switch_session(self, session_id: str) -> Session | None:
        """Switch to a different session.

        Args:
            session_id: Session ID to switch to

        Returns:
            The session or None if not found
        """
        session = await self.get_session(session_id)
        if session:
            self._current_session_id = session_id
            logger.debug("Switched to session", session_id=session_id)
        return session

    @property
    def current_session(self) -> Session | None:
        """Get the current active session."""
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None

    async def store_memory(
        self,
        entry: MemoryEntry,
        session_id: str | None = None,
    ) -> None:
        """Store a memory in a session.

        Args:
            entry: Memory entry to store
            session_id: Session ID (uses current session if not provided)
        """
        session = await self._get_target_session(session_id)

        await session.working_memory.store(entry)
        session.memory_operations += 1
        session.touch()

        # Persist if store available
        if self._store:
            self._store.store(entry, session_id=session.id)

    async def retrieve_memory(
        self,
        query: MemoryQuery,
        session_id: str | None = None,
    ) -> MemoryResult:
        """Retrieve memories from a session.

        Args:
            query: Query to execute
            session_id: Session ID (uses current session if not provided)

        Returns:
            Query results
        """
        session = await self._get_target_session(session_id)

        result = await session.working_memory.retrieve(query)
        session.memory_operations += 1
        session.touch()

        return result

    async def clear_session(self, session_id: str | None = None) -> None:
        """Clear all memories in a session.

        Args:
            session_id: Session ID (uses current session if not provided)
        """
        session = await self._get_target_session(session_id)

        await session.working_memory.clear()
        session.touch()

        if self._store:
            self._store.clear(session_id=session.id)

        logger.info("Session cleared", session_id=session.id)

    async def pause_session(self, session_id: str) -> None:
        """Pause a session (saves state but marks inactive)."""
        session = self._sessions.get(session_id)
        if session:
            session.state = SessionState.PAUSED
            session.touch()
            await self._save_session(session)
            logger.info("Session paused", session_id=session_id)

    async def resume_session(self, session_id: str) -> Session | None:
        """Resume a paused session."""
        session = self._sessions.get(session_id)
        if session and session.state == SessionState.PAUSED:
            session.state = SessionState.ACTIVE
            session.touch()
            logger.info("Session resumed", session_id=session_id)
            return session
        return None

    async def archive_session(self, session_id: str) -> None:
        """Archive a session (persists and removes from active)."""
        session = self._sessions.get(session_id)
        if session:
            session.state = SessionState.ARCHIVED
            await self._save_session(session)

            # Remove from active sessions
            del self._sessions[session_id]

            if self._current_session_id == session_id:
                self._current_session_id = None

            logger.info("Session archived", session_id=session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

        if self._current_session_id == session_id:
            self._current_session_id = None

        if self._store:
            return self._store.delete_session(session_id)

        return True

    def list_sessions(
        self,
        state: SessionState | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List all sessions.

        Args:
            state: Filter by session state
            limit: Maximum number of sessions to return

        Returns:
            List of session info dictionaries
        """
        sessions = self._sessions.values()

        if state:
            sessions = [s for s in sessions if s.state == state]

        # Sort by last activity (most recent first)
        sessions = sorted(
            sessions,
            key=lambda s: s.last_activity,
            reverse=True,
        )[:limit]

        return [s.to_dict() for s in sessions]

    async def _get_target_session(self, session_id: str | None) -> Session:
        """Get the target session for an operation."""
        if session_id:
            session = await self.get_session(session_id)
        else:
            session = self.current_session

        if not session:
            raise ValueError("No active session")

        return session

    async def _save_session(self, session: Session) -> None:
        """Save session state to storage."""
        if not self._store or not session.working_memory:
            return

        for entry in session.working_memory.entries:
            self._store.store(entry, session_id=session.id)

    async def _cleanup_expired(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        expired = [
            s for s in self._sessions.values()
            if s.is_expired()
        ]

        for session in expired:
            session.state = SessionState.EXPIRED
            await self._save_session(session)
            del self._sessions[session.id]

        if expired:
            logger.info("Cleaned up expired sessions", count=len(expired))

        return len(expired)

    def get_stats(self) -> dict:
        """Get session manager statistics."""
        states = {}
        for state in SessionState:
            states[state.value] = sum(
                1 for s in self._sessions.values()
                if s.state == state
            )

        total_memories = sum(
            len(s.working_memory.entries)
            for s in self._sessions.values()
            if s.working_memory
        )

        total_operations = sum(
            s.memory_operations
            for s in self._sessions.values()
        )

        return {
            "total_sessions": len(self._sessions),
            "by_state": states,
            "current_session_id": self._current_session_id,
            "max_sessions": self._max_sessions,
            "total_memories": total_memories,
            "total_operations": total_operations,
        }

    async def export_session(self, session_id: str) -> dict:
        """Export a session to JSON format.

        Args:
            session_id: Session to export

        Returns:
            Session data as dictionary
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        data = session.to_dict()

        # Include memories
        if session.working_memory:
            data["memories"] = [
                {
                    "id": str(e.id),
                    "content": e.content,
                    "importance": e.importance.effective_score,
                    "tags": e.tags,
                    "created_at": e.created_at.isoformat(),
                }
                for e in session.working_memory.entries
            ]

        return data

    async def import_session(
        self,
        data: dict,
        merge: bool = False,
    ) -> Session:
        """Import a session from JSON data.

        Args:
            data: Session data dictionary
            merge: If True, merge with existing session

        Returns:
            The imported session
        """
        session_id = data.get("id", str(uuid4()))

        if merge and session_id in self._sessions:
            session = self._sessions[session_id]
        else:
            session = await self.create_session(
                session_id=session_id,
                config=SessionConfig(
                    name=data.get("name", ""),
                    max_entries=data.get("config", {}).get("max_entries", 100),
                    max_tokens=data.get("config", {}).get("max_tokens", 8000),
                ),
            )

        # Import memories
        for mem_data in data.get("memories", []):
            from memoryforge.core.types import ImportanceScore, MemoryLayer

            entry = MemoryEntry(
                content=mem_data["content"],
                layer=MemoryLayer.WORKING,
                importance=ImportanceScore(
                    base_score=mem_data.get("importance", 0.5)
                ),
                tags=mem_data.get("tags", []),
            )
            await session.working_memory.store(entry)

        return session
