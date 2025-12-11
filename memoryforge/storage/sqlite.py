"""SQLite storage backend for local development and persistence."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID

import structlog

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer

logger = structlog.get_logger()


class SQLiteMemoryStore:
    """SQLite-based persistent storage for memory entries.

    Provides local persistence without requiring external databases.
    Suitable for development, testing, and single-user deployments.
    """

    def __init__(self, db_path: str | Path = "memoryforge.db"):
        self._db_path = Path(db_path)
        self._init_db()
        logger.info("SQLite store initialized", path=str(self._db_path))

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with context management."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    importance_base REAL DEFAULT 0.5,
                    importance_recency REAL DEFAULT 1.0,
                    importance_access_count INTEGER DEFAULT 0,
                    importance_last_accessed TEXT,
                    metadata TEXT DEFAULT '{}',
                    embedding TEXT,
                    tags TEXT DEFAULT '[]',
                    source_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    session_id TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_memories_layer
                    ON memories(layer);
                CREATE INDEX IF NOT EXISTS idx_memories_created
                    ON memories(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                    ON memories(importance_base DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_session
                    ON memories(session_id);

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    summary TEXT NOT NULL,
                    key_facts TEXT DEFAULT '[]',
                    time_range_start TEXT,
                    time_range_end TEXT,
                    compression_ratio REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
            """)

    def store(self, entry: MemoryEntry, session_id: str | None = None) -> None:
        """Store a memory entry."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories (
                    id, content, layer,
                    importance_base, importance_recency,
                    importance_access_count, importance_last_accessed,
                    metadata, embedding, tags, source_id,
                    created_at, updated_at, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(entry.id),
                entry.content,
                entry.layer.value,
                entry.importance.base_score,
                entry.importance.recency_weight,
                entry.importance.access_count,
                entry.importance.last_accessed.isoformat(),
                json.dumps(entry.metadata),
                json.dumps(entry.embedding) if entry.embedding else None,
                json.dumps(entry.tags),
                str(entry.source_id) if entry.source_id else None,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                session_id,
            ))

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (entry_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_entry(row)

    def get_all(
        self,
        layer: MemoryLayer | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """Get all memory entries with optional filtering."""
        query = "SELECT * FROM memories WHERE 1=1"
        params = []

        if layer:
            query += " AND layer = ?"
            params.append(layer.value)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def search(
        self,
        query_text: str,
        layer: MemoryLayer | None = None,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search memories by content (simple text matching)."""
        sql_query = """
            SELECT *,
                   (CASE
                       WHEN content LIKE ? THEN 1.0
                       WHEN content LIKE ? THEN 0.7
                       ELSE 0.3
                   END) * importance_base as relevance
            FROM memories
            WHERE content LIKE ?
        """
        params = [
            f"%{query_text}%",  # exact match bonus
            f"% {query_text} %",  # word match
            f"%{query_text}%",  # filter
        ]

        if layer:
            sql_query += " AND layer = ?"
            params.append(layer.value)

        sql_query += " ORDER BY relevance DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(sql_query, params).fetchall()

        return [
            (self._row_to_entry(row), row["relevance"])
            for row in rows
        ]

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (entry_id,)
            )
        return cursor.rowcount > 0

    def clear(self, session_id: str | None = None) -> int:
        """Clear memories, optionally for a specific session."""
        with self._get_connection() as conn:
            if session_id:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE session_id = ?",
                    (session_id,)
                )
            else:
                cursor = conn.execute("DELETE FROM memories")
        return cursor.rowcount

    def count(
        self,
        layer: MemoryLayer | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memory entries."""
        query = "SELECT COUNT(*) FROM memories WHERE 1=1"
        params = []

        if layer:
            query += " AND layer = ?"
            params.append(layer.value)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        with self._get_connection() as conn:
            result = conn.execute(query, params).fetchone()

        return result[0]

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        return MemoryEntry(
            id=UUID(row["id"]),
            content=row["content"],
            layer=MemoryLayer(row["layer"]),
            importance=ImportanceScore(
                base_score=row["importance_base"],
                recency_weight=row["importance_recency"],
                access_count=row["importance_access_count"],
                last_accessed=datetime.fromisoformat(row["importance_last_accessed"]),
            ),
            metadata=json.loads(row["metadata"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            tags=json.loads(row["tags"]),
            source_id=UUID(row["source_id"]) if row["source_id"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # Session management

    def create_session(self, session_id: str, name: str = "") -> None:
        """Create a new session."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO sessions (id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, name, now, now))

    def get_sessions(self) -> list[dict]:
        """Get all sessions."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT s.*, COUNT(m.id) as memory_count
                FROM sessions s
                LEFT JOIN memories m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            """).fetchall()

        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its memories."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM memories WHERE session_id = ?",
                (session_id,)
            )
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,)
            )
        return cursor.rowcount > 0

    # Export/Import

    def export_to_json(self, session_id: str | None = None) -> dict:
        """Export memories to JSON format."""
        entries = self.get_all(session_id=session_id, limit=10000)

        return {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "count": len(entries),
            "memories": [
                {
                    "id": str(e.id),
                    "content": e.content,
                    "layer": e.layer.value,
                    "importance": e.importance.base_score,
                    "tags": e.tags,
                    "created_at": e.created_at.isoformat(),
                }
                for e in entries
            ],
        }

    def import_from_json(
        self,
        data: dict,
        session_id: str | None = None,
    ) -> int:
        """Import memories from JSON format."""
        imported = 0

        for item in data.get("memories", []):
            entry = MemoryEntry(
                content=item["content"],
                layer=MemoryLayer(item.get("layer", "working")),
                importance=ImportanceScore(
                    base_score=item.get("importance", 0.5)
                ),
                tags=item.get("tags", []),
            )
            self.store(entry, session_id=session_id)
            imported += 1

        return imported

    def get_stats(self) -> dict:
        """Get storage statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

            by_layer = {}
            for layer in MemoryLayer:
                count = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE layer = ?",
                    (layer.value,)
                ).fetchone()[0]
                by_layer[layer.value] = count

            avg_importance = conn.execute(
                "SELECT AVG(importance_base) FROM memories"
            ).fetchone()[0] or 0

            session_count = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]

        return {
            "total_memories": total,
            "by_layer": by_layer,
            "avg_importance": round(avg_importance, 3),
            "session_count": session_count,
            "db_path": str(self._db_path),
            "db_size_bytes": self._db_path.stat().st_size if self._db_path.exists() else 0,
        }
