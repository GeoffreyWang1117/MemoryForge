"""FastAPI application for MemoryForge."""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from memoryforge.config import get_settings
from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer, MemoryQuery
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.conversation import ConversationManager
from memoryforge.storage.sqlite import SQLiteMemoryStore


# Global state
_working_memory: WorkingMemory | None = None
_conversation_manager: ConversationManager | None = None
_sqlite_store: SQLiteMemoryStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan management."""
    global _working_memory, _conversation_manager, _sqlite_store

    # Initialize memory systems
    _working_memory = WorkingMemory(max_entries=100, max_tokens=8000)
    _sqlite_store = SQLiteMemoryStore("memoryforge.db")

    api_key = os.getenv("LLM_OPENAI_API_KEY")
    if api_key:
        _conversation_manager = ConversationManager(
            api_key=api_key,
            model="gpt-4o-mini",
            auto_score_importance=True,
        )

    yield

    # Cleanup
    if _working_memory:
        await _working_memory.clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="MemoryForge API",
        description="Hierarchical Context Memory System for LLM Agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(memory_router, prefix="/api/v1/memory", tags=["Memory"])
    app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
    app.include_router(health_router, tags=["Health"])
    app.include_router(persistence_router, prefix="/api/v1/persistence", tags=["Persistence"])

    # WebSocket endpoint
    from memoryforge.api.websocket import websocket_endpoint
    app.add_api_websocket_route("/ws/chat", websocket_endpoint)

    return app


# --- Request/Response Models ---

class StoreMemoryRequest(BaseModel):
    content: str = Field(..., description="Memory content")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class StoreMemoryResponse(BaseModel):
    id: str
    message: str


class QueryMemoryRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    tags_filter: list[str] | None = None


class MemoryEntryResponse(BaseModel):
    id: str
    content: str
    importance: float
    tags: list[str]
    created_at: str


class QueryMemoryResponse(BaseModel):
    entries: list[MemoryEntryResponse]
    total_candidates: int
    query_time_ms: float


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str | None = Field(default=None)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    turn_count: int
    memory_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    memory_entries: int


# --- Dependency ---

def get_working_memory() -> WorkingMemory:
    if _working_memory is None:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    return _working_memory


def get_conversation() -> ConversationManager:
    if _conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation system not available (API key not configured)"
        )
    return _conversation_manager


def get_sqlite_store() -> SQLiteMemoryStore:
    if _sqlite_store is None:
        raise HTTPException(status_code=503, detail="SQLite store not initialized")
    return _sqlite_store


# --- Routers ---

from fastapi import APIRouter

memory_router = APIRouter()
chat_router = APIRouter()
health_router = APIRouter()
persistence_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse)
async def health_check(memory: WorkingMemory = Depends(get_working_memory)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        memory_entries=len(memory.entries),
    )


@memory_router.post("/store", response_model=StoreMemoryResponse)
async def store_memory(
    request: StoreMemoryRequest,
    memory: WorkingMemory = Depends(get_working_memory),
):
    """Store a new memory entry."""
    entry = MemoryEntry(
        content=request.content,
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=request.importance),
        tags=request.tags,
        metadata=request.metadata,
    )

    await memory.store(entry)

    return StoreMemoryResponse(
        id=str(entry.id),
        message="Memory stored successfully",
    )


@memory_router.post("/query", response_model=QueryMemoryResponse)
async def query_memory(
    request: QueryMemoryRequest,
    memory: WorkingMemory = Depends(get_working_memory),
):
    """Query memories by content similarity."""
    query = MemoryQuery(
        query_text=request.query,
        target_layers=[MemoryLayer.WORKING],
        top_k=request.top_k,
        min_importance=request.min_importance,
        tags_filter=request.tags_filter,
    )

    result = await memory.retrieve(query)

    entries = [
        MemoryEntryResponse(
            id=str(e.id),
            content=e.content,
            importance=e.importance.effective_score,
            tags=e.tags,
            created_at=e.created_at.isoformat(),
        )
        for e in result.entries
    ]

    return QueryMemoryResponse(
        entries=entries,
        total_candidates=result.total_candidates,
        query_time_ms=result.query_time_ms,
    )


@memory_router.get("/list", response_model=QueryMemoryResponse)
async def list_memories(
    limit: int = 50,
    memory: WorkingMemory = Depends(get_working_memory),
):
    """List all stored memories."""
    entries = memory.entries[:limit]

    return QueryMemoryResponse(
        entries=[
            MemoryEntryResponse(
                id=str(e.id),
                content=e.content,
                importance=e.importance.effective_score,
                tags=e.tags,
                created_at=e.created_at.isoformat(),
            )
            for e in entries
        ],
        total_candidates=len(memory.entries),
        query_time_ms=0,
    )


@memory_router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    memory: WorkingMemory = Depends(get_working_memory),
):
    """Delete a memory entry by ID."""
    success = await memory.delete(memory_id)

    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"message": "Memory deleted successfully"}


@memory_router.delete("/")
async def clear_memories(memory: WorkingMemory = Depends(get_working_memory)):
    """Clear all memories."""
    await memory.clear()
    return {"message": "All memories cleared"}


@memory_router.get("/stats")
async def memory_stats(memory: WorkingMemory = Depends(get_working_memory)):
    """Get memory system statistics."""
    return {
        "total_entries": len(memory.entries),
        "pinned_entries": len(memory._pinned),
        "token_count": memory.token_count,
        "max_entries": memory._max_entries,
        "max_tokens": memory._max_tokens,
    }


@chat_router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    conversation: ConversationManager = Depends(get_conversation),
):
    """Send a message and get a response with memory integration."""
    response = await conversation.chat(request.message)
    stats = conversation.get_stats()

    return ChatResponse(
        response=response,
        session_id=stats["session_id"],
        turn_count=stats["turn_count"],
        memory_count=stats["total_memories"],
    )


@chat_router.get("/stats")
async def chat_stats(conversation: ConversationManager = Depends(get_conversation)):
    """Get conversation statistics."""
    return conversation.get_stats()


@chat_router.post("/memory")
async def add_chat_memory(
    content: str,
    importance: float = 0.8,
    conversation: ConversationManager = Depends(get_conversation),
):
    """Manually add a memory to the conversation context."""
    entry_id = await conversation.add_memory(content, importance)
    return {"id": entry_id, "message": "Memory added to conversation"}


# --- Persistence Endpoints ---

class SaveMemoriesRequest(BaseModel):
    session_id: str | None = None


class ImportMemoriesRequest(BaseModel):
    data: dict
    session_id: str | None = None


@persistence_router.post("/save")
async def save_memories(
    request: SaveMemoriesRequest,
    memory: WorkingMemory = Depends(get_working_memory),
    store: SQLiteMemoryStore = Depends(get_sqlite_store),
):
    """Save working memory entries to SQLite."""
    count = 0
    for entry in memory.entries:
        store.store(entry, session_id=request.session_id)
        count += 1
    return {"message": f"Saved {count} memories", "count": count}


@persistence_router.get("/load")
async def load_memories(
    session_id: str | None = None,
    limit: int = 100,
    store: SQLiteMemoryStore = Depends(get_sqlite_store),
    memory: WorkingMemory = Depends(get_working_memory),
):
    """Load memories from SQLite into working memory."""
    entries = store.get_all(session_id=session_id, limit=limit)
    loaded = 0
    for entry in entries:
        await memory.store(entry)
        loaded += 1
    return {"message": f"Loaded {loaded} memories", "count": loaded}


@persistence_router.get("/export")
async def export_memories(
    session_id: str | None = None,
    store: SQLiteMemoryStore = Depends(get_sqlite_store),
):
    """Export memories to JSON format."""
    return store.export_to_json(session_id=session_id)


@persistence_router.post("/import")
async def import_memories(
    request: ImportMemoriesRequest,
    store: SQLiteMemoryStore = Depends(get_sqlite_store),
):
    """Import memories from JSON format."""
    count = store.import_from_json(request.data, session_id=request.session_id)
    return {"message": f"Imported {count} memories", "count": count}


@persistence_router.get("/stats")
async def persistence_stats(store: SQLiteMemoryStore = Depends(get_sqlite_store)):
    """Get SQLite storage statistics."""
    return store.get_stats()


@persistence_router.get("/sessions")
async def list_sessions(store: SQLiteMemoryStore = Depends(get_sqlite_store)):
    """List all sessions."""
    return store.get_sessions()


@persistence_router.post("/sessions")
async def create_session(
    session_id: str,
    name: str = "",
    store: SQLiteMemoryStore = Depends(get_sqlite_store),
):
    """Create a new session."""
    store.create_session(session_id, name)
    return {"message": "Session created", "session_id": session_id}


@persistence_router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    store: SQLiteMemoryStore = Depends(get_sqlite_store),
):
    """Delete a session and all its memories."""
    success = store.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}


# Create default app instance
app = create_app()
