#!/usr/bin/env python3
"""Test LLM integration with the memory system."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from memoryforge.core.types import (
    ConversationTurn,
    ImportanceScore,
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
)
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.llm.summarizer import LLMSummarizer
from memoryforge.llm.embedder import OpenAIEmbedder


async def test_summarizer():
    """Test LLM summarization."""
    print("=" * 60)
    print("Testing LLM Summarizer")
    print("=" * 60)

    summarizer = LLMSummarizer(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("LLM_OPENAI_API_KEY"),
    )

    conversation = """
    User: I want to build a REST API for a todo application.
    Assistant: Great! I'll help you build a REST API. What tech stack would you prefer?
    User: Let's use FastAPI with Python and PostgreSQL for the database.
    Assistant: Excellent choice! FastAPI is fast and easy to use. For PostgreSQL, we can use SQLAlchemy as the ORM.
    User: Should we add authentication?
    Assistant: Yes, I recommend using JWT tokens for authentication. We can use python-jose for JWT handling.
    User: Let's also add rate limiting to prevent abuse.
    Assistant: Good idea. We can use slowapi for rate limiting in FastAPI.
    """

    print("\nOriginal conversation:")
    print(conversation[:200] + "...")
    print(f"\nOriginal length: {len(conversation)} chars")

    summary = await summarizer.summarize(conversation)
    print(f"\nSummary ({len(summary)} chars):")
    print(summary)

    key_facts = await summarizer.extract_key_facts(conversation)
    print(f"\nExtracted key facts ({len(key_facts)}):")
    for fact in key_facts:
        print(f"  - {fact}")

    compression = len(summary) / len(conversation)
    print(f"\nCompression ratio: {compression:.2%}")


async def test_openai_embeddings():
    """Test OpenAI embeddings."""
    print("\n" + "=" * 60)
    print("Testing OpenAI Embeddings")
    print("=" * 60)

    embedder = OpenAIEmbedder(
        api_key=os.getenv("LLM_OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )

    texts = [
        "Building a REST API with FastAPI",
        "Creating RESTful endpoints in Python",
        "PostgreSQL database schema design",
        "Machine learning model training",
    ]

    print(f"\nEmbedding {len(texts)} texts...")
    embeddings = await embedder.embed_batch(texts)

    print(f"Embedding dimension: {len(embeddings[0])}")

    # Calculate similarity
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nSimilarity matrix:")
    print(f"{'':30}", end="")
    for i in range(len(texts)):
        print(f"[{i}]    ", end="")
    print()

    for i, text in enumerate(texts):
        print(f"[{i}] {text[:25]:25}", end="")
        for j in range(len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"{sim:.3f}  ", end="")
        print()


async def test_memory_with_llm():
    """Test memory system with LLM integration."""
    print("\n" + "=" * 60)
    print("Testing Memory System with LLM")
    print("=" * 60)

    # Create working memory
    wm = WorkingMemory(max_entries=50, max_tokens=4000)

    # Simulate a development session
    session_entries = [
        ("User wants to build an e-commerce platform", 0.9, ["requirement"]),
        ("Tech stack: Next.js frontend, FastAPI backend", 0.95, ["decision", "tech"]),
        ("Database: PostgreSQL with Redis caching", 0.9, ["decision", "database"]),
        ("Payment integration: Stripe API", 0.85, ["decision", "payment"]),
        ("User mentioned they have 6 months timeline", 0.7, ["context"]),
        ("Need to support 10k concurrent users", 0.8, ["requirement", "scale"]),
        ("Mobile app planned for phase 2", 0.6, ["future"]),
    ]

    print("\nStoring session entries...")
    for content, importance, tags in session_entries:
        entry = MemoryEntry(
            content=content,
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=importance),
            tags=tags,
        )
        await wm.store(entry)
        print(f"  [{importance:.1f}] {content[:50]}...")

    # Query the memory
    print("\nQuerying for 'database decisions'...")
    query = MemoryQuery(
        query_text="database",
        target_layers=[MemoryLayer.WORKING],
        top_k=3,
    )
    result = await wm.retrieve(query)

    print("Results:")
    for entry, score in zip(result.entries, result.scores):
        print(f"  [{score:.2f}] {entry.content}")

    # Get context window for LLM
    print("\nContext window for LLM (max 500 tokens):")
    context_entries = wm.get_context_window(max_tokens=500)
    for entry in context_entries:
        print(f"  - {entry.content}")


async def main():
    """Run all tests."""
    await test_summarizer()
    await test_openai_embeddings()
    await test_memory_with_llm()

    print("\n" + "=" * 60)
    print("All LLM integration tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
