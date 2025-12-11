"""WebSocket support for real-time chat with memory."""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Set
from uuid import uuid4

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from memoryforge.conversation import ConversationManager

logger = structlog.get_logger()


@dataclass
class ChatSession:
    """A WebSocket chat session."""

    session_id: str
    websocket: WebSocket
    conversation: ConversationManager
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0


class ConnectionManager:
    """Manages WebSocket connections and chat sessions."""

    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._api_key = os.getenv("LLM_OPENAI_API_KEY")

    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection and create a session."""
        await websocket.accept()

        session_id = str(uuid4())

        if not self._api_key:
            await websocket.send_json({
                "type": "error",
                "message": "API key not configured",
            })
            await websocket.close()
            raise ValueError("API key not configured")

        conversation = ConversationManager(
            api_key=self._api_key,
            model="gpt-4o-mini",
            auto_score_importance=True,
        )

        session = ChatSession(
            session_id=session_id,
            websocket=websocket,
            conversation=conversation,
        )

        self._sessions[session_id] = session

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to MemoryForge chat",
        })

        logger.info("WebSocket connected", session_id=session_id)

        return session_id

    async def disconnect(self, session_id: str) -> None:
        """Handle WebSocket disconnection."""
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            logger.info(
                "WebSocket disconnected",
                session_id=session_id,
                messages=session.message_count,
            )

    async def handle_message(self, session_id: str, data: dict) -> None:
        """Handle an incoming WebSocket message."""
        session = self._sessions.get(session_id)
        if not session:
            return

        message_type = data.get("type", "chat")

        if message_type == "chat":
            await self._handle_chat(session, data)
        elif message_type == "memory_query":
            await self._handle_memory_query(session, data)
        elif message_type == "memory_add":
            await self._handle_memory_add(session, data)
        elif message_type == "stats":
            await self._handle_stats(session)
        elif message_type == "ping":
            await session.websocket.send_json({"type": "pong"})
        else:
            await session.websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {message_type}",
            })

    async def _handle_chat(self, session: ChatSession, data: dict) -> None:
        """Handle a chat message with streaming response."""
        user_message = data.get("message", "").strip()

        if not user_message:
            await session.websocket.send_json({
                "type": "error",
                "message": "Empty message",
            })
            return

        session.message_count += 1

        # Send acknowledgment
        await session.websocket.send_json({
            "type": "chat_start",
            "turn": session.conversation.turn_count + 1,
        })

        # Stream the response
        try:
            full_response = ""
            async for chunk in session.conversation.chat_stream(user_message):
                full_response += chunk
                await session.websocket.send_json({
                    "type": "chat_chunk",
                    "content": chunk,
                })

            # Send completion
            stats = session.conversation.get_stats()
            await session.websocket.send_json({
                "type": "chat_complete",
                "turn": stats["turn_count"],
                "memory_count": stats["total_memories"],
                "pinned_count": stats["pinned_memories"],
            })

        except Exception as e:
            logger.error("Chat error", error=str(e), session_id=session.session_id)
            await session.websocket.send_json({
                "type": "error",
                "message": f"Chat error: {str(e)}",
            })

    async def _handle_memory_query(self, session: ChatSession, data: dict) -> None:
        """Handle a memory query request."""
        query = data.get("query", "")
        top_k = data.get("top_k", 5)

        try:
            results = await session.conversation.query_memory(query, top_k)
            await session.websocket.send_json({
                "type": "memory_results",
                "query": query,
                "results": results,
            })
        except Exception as e:
            await session.websocket.send_json({
                "type": "error",
                "message": f"Query error: {str(e)}",
            })

    async def _handle_memory_add(self, session: ChatSession, data: dict) -> None:
        """Handle adding a memory manually."""
        content = data.get("content", "")
        importance = data.get("importance", 0.8)

        if not content:
            await session.websocket.send_json({
                "type": "error",
                "message": "Empty content",
            })
            return

        try:
            entry_id = await session.conversation.add_memory(content, importance)
            await session.websocket.send_json({
                "type": "memory_added",
                "id": entry_id,
            })
        except Exception as e:
            await session.websocket.send_json({
                "type": "error",
                "message": f"Add memory error: {str(e)}",
            })

    async def _handle_stats(self, session: ChatSession) -> None:
        """Handle stats request."""
        stats = session.conversation.get_stats()
        stats["connected_at"] = session.created_at.isoformat()
        stats["message_count"] = session.message_count

        await session.websocket.send_json({
            "type": "stats",
            "data": stats,
        })

    def get_active_sessions(self) -> list[dict]:
        """Get information about active sessions."""
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat(),
                "message_count": s.message_count,
                "turn_count": s.conversation.turn_count,
            }
            for s in self._sessions.values()
        ]

    @property
    def connection_count(self) -> int:
        return len(self._sessions)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint handler."""
    session_id = None
    try:
        session_id = await manager.connect(websocket)

        while True:
            data = await websocket.receive_json()
            await manager.handle_message(session_id, data)

    except WebSocketDisconnect:
        if session_id:
            await manager.disconnect(session_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        if session_id:
            await manager.disconnect(session_id)
