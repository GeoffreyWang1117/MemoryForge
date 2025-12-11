#!/usr/bin/env python3
"""Example usage of MemoryForge memory system."""

import asyncio
from datetime import datetime

from memoryforge.core.types import (
    ConversationTurn,
    ImportanceScore,
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
    SemanticEntity,
    SemanticRelation,
)
from memoryforge.memory.working.memory import WorkingMemory


async def demo_working_memory():
    """Demonstrate working memory functionality."""
    print("=" * 60)
    print("Working Memory Demo")
    print("=" * 60)

    # Create working memory
    wm = WorkingMemory(max_entries=50, max_tokens=4000, importance_threshold=0.7)

    # Store some entries
    entries = [
        MemoryEntry(
            content="User wants to build a REST API with FastAPI",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.8),
            tags=["requirement", "api"],
        ),
        MemoryEntry(
            content="Database choice: PostgreSQL with SQLAlchemy ORM",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.9),
            tags=["decision", "database"],
        ),
        MemoryEntry(
            content="Authentication will use JWT tokens",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.85),
            tags=["decision", "auth"],
        ),
        MemoryEntry(
            content="Discussed pagination approach for list endpoints",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.5),
            tags=["discussion"],
        ),
    ]

    for entry in entries:
        await wm.store(entry)
        print(f"Stored: {entry.content[:50]}...")

    print(f"\nTotal entries: {len(wm.entries)}")
    print(f"Pinned entries (high importance): {len(wm._pinned)}")

    # Query working memory
    query = MemoryQuery(
        query_text="database",
        target_layers=[MemoryLayer.WORKING],
        top_k=5,
    )
    result = await wm.retrieve(query)

    print(f"\nQuery results for 'database':")
    for entry, score in zip(result.entries, result.scores):
        print(f"  [{score:.2f}] {entry.content[:60]}...")

    # Get context window
    context = wm.get_context_window(max_tokens=500)
    print(f"\nContext window (500 tokens): {len(context)} entries")
    for entry in context:
        print(f"  - {entry.content[:50]}...")


async def demo_memory_types():
    """Demonstrate different memory type structures."""
    print("\n" + "=" * 60)
    print("Memory Types Demo")
    print("=" * 60)

    # Conversation turn
    turn = ConversationTurn(
        role="user",
        content="Can you help me implement user authentication?",
        timestamp=datetime.utcnow(),
        metadata={"session_id": "abc123"},
    )
    print(f"\nConversation Turn:")
    print(f"  Role: {turn.role}")
    print(f"  Content: {turn.content}")

    # Semantic entity
    entity = SemanticEntity(
        name="authenticate_user",
        entity_type="function",
        properties={
            "params": ["username", "password"],
            "returns": "User | None",
        },
        source_file="src/auth/handlers.py",
        line_number=42,
    )
    print(f"\nSemantic Entity:")
    print(f"  Name: {entity.name}")
    print(f"  Type: {entity.entity_type}")
    print(f"  Location: {entity.source_file}:{entity.line_number}")

    # Semantic relation
    relation = SemanticRelation(
        source_id=entity.id,
        target_id=entity.id,  # Would be different in real usage
        relation_type="calls",
        properties={"call_count": 15},
        weight=0.8,
    )
    print(f"\nSemantic Relation:")
    print(f"  Type: {relation.relation_type}")
    print(f"  Weight: {relation.weight}")


async def main():
    """Run all demos."""
    await demo_working_memory()
    await demo_memory_types()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start databases: docker-compose up -d")
    print("2. Configure .env with your API keys")
    print("3. Run tests: pytest tests/")


if __name__ == "__main__":
    asyncio.run(main())
