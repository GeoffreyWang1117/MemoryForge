"""Integration tests for multi-layer memory system."""

import pytest
from datetime import datetime, timezone

from memoryforge.core.types import (
    ImportanceScore,
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
)
from memoryforge.memory.working.memory import WorkingMemory


@pytest.fixture
def working_memory():
    """Create a working memory instance."""
    return WorkingMemory(max_entries=50, max_tokens=4000)


def create_entry(
    content: str,
    importance: float = 0.5,
    tags: list[str] | None = None,
    layer: MemoryLayer = MemoryLayer.WORKING,
) -> MemoryEntry:
    """Helper to create memory entries."""
    return MemoryEntry(
        content=content,
        layer=layer,
        importance=ImportanceScore(base_score=importance),
        tags=tags or [],
    )


class TestWorkingMemoryIntegration:
    """Test working memory operations."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_flow(self, working_memory):
        """Test complete store and retrieve flow."""
        # Store multiple entries
        entries = [
            create_entry("First important fact", importance=0.9, tags=["fact"]),
            create_entry("Second fact about Python", importance=0.7, tags=["python"]),
            create_entry("Third fact about testing", importance=0.8, tags=["testing"]),
        ]

        for entry in entries:
            await working_memory.store(entry)

        # Verify all stored
        assert len(working_memory.entries) == 3

        # Query by text
        query = MemoryQuery(
            query_text="Python programming",
            target_layers=[MemoryLayer.WORKING],
            top_k=5,
        )
        result = await working_memory.retrieve(query)
        assert len(result.entries) > 0

    @pytest.mark.asyncio
    async def test_importance_based_eviction(self, working_memory):
        """Test that high importance entries are pinned and survive."""
        # Add high importance entry first (will be auto-pinned due to threshold)
        high_importance = create_entry("Critical information", importance=0.95)
        await working_memory.store(high_importance)

        # Fill with low importance entries to trigger eviction
        for i in range(60):  # More than max_entries (50)
            entry = create_entry(f"Low importance entry {i}", importance=0.2)
            await working_memory.store(entry)

        # High importance should still be there (it's pinned)
        found = False
        for e in working_memory.entries:
            if e.content == "Critical information":
                found = True
                break
        assert found, "High importance pinned entry should survive eviction"

    @pytest.mark.asyncio
    async def test_tag_filtering(self, working_memory):
        """Test filtering by tags."""
        entries = [
            create_entry("API endpoint design", tags=["api", "design"]),
            create_entry("Database schema", tags=["database", "design"]),
            create_entry("Unit test coverage", tags=["testing"]),
            create_entry("API authentication", tags=["api", "security"]),
        ]

        for entry in entries:
            await working_memory.store(entry)

        # Query with tag filter
        query = MemoryQuery(
            query_text="design",
            target_layers=[MemoryLayer.WORKING],
            tags_filter=["api"],
            top_k=10,
        )
        result = await working_memory.retrieve(query)

        # Should find entries with "api" tag
        for entry in result.entries:
            assert "api" in entry.tags

    @pytest.mark.asyncio
    async def test_context_window_generation(self, working_memory):
        """Test generating context for LLM."""
        entries = [
            create_entry("The project uses Python 3.11", importance=0.8),
            create_entry("FastAPI is the web framework", importance=0.9),
            create_entry("Tests use pytest", importance=0.7),
        ]

        for entry in entries:
            await working_memory.store(entry)

        context_entries = working_memory.get_context_window(max_tokens=1000)

        # Should contain relevant entries
        assert len(context_entries) > 0
        contents = " ".join(e.content for e in context_entries)
        assert "Python" in contents or "FastAPI" in contents or "pytest" in contents


class TestMemoryWorkflow:
    """Test real-world memory workflows."""

    @pytest.mark.asyncio
    async def test_coding_session_workflow(self, working_memory):
        """Simulate a coding session with memory."""
        # User asks about the codebase
        await working_memory.store(
            create_entry(
                "User asked about authentication implementation",
                importance=0.7,
                tags=["question", "auth"],
            )
        )

        # System provides answer
        await working_memory.store(
            create_entry(
                "Authentication uses JWT tokens with RS256 signing",
                importance=0.9,
                tags=["answer", "auth", "jwt"],
            )
        )

        # User makes a decision
        await working_memory.store(
            create_entry(
                "Decision: Add refresh token support",
                importance=0.95,
                tags=["decision", "auth"],
            )
        )

        # Later query about auth
        query = MemoryQuery(
            query_text="authentication decisions",
            target_layers=[MemoryLayer.WORKING],
            top_k=5,
        )
        result = await working_memory.retrieve(query)

        # Should find relevant auth memories
        assert len(result.entries) > 0
        auth_entries = [e for e in result.entries if "auth" in e.tags]
        assert len(auth_entries) > 0

    @pytest.mark.asyncio
    async def test_debugging_session_workflow(self, working_memory):
        """Simulate a debugging session."""
        # Error encountered
        await working_memory.store(
            create_entry(
                "Error: ConnectionRefusedError on port 5432",
                importance=0.85,
                tags=["error", "database"],
            )
        )

        # Investigation findings
        await working_memory.store(
            create_entry(
                "Finding: PostgreSQL container not running",
                importance=0.8,
                tags=["finding", "database", "docker"],
            )
        )

        # Solution
        await working_memory.store(
            create_entry(
                "Solution: docker-compose up -d postgres",
                importance=0.9,
                tags=["solution", "database", "docker"],
            )
        )

        # Query for database issues
        query = MemoryQuery(
            query_text="database connection problems",
            target_layers=[MemoryLayer.WORKING],
            tags_filter=["database"],
            top_k=10,
        )
        result = await working_memory.retrieve(query)

        assert len(result.entries) >= 2
        # Should have error and solution
        contents = " ".join(e.content for e in result.entries)
        assert "Error" in contents or "Solution" in contents

    @pytest.mark.asyncio
    async def test_knowledge_accumulation(self, working_memory):
        """Test accumulating knowledge over time."""
        knowledge_items = [
            ("Python uses indentation for blocks", ["python", "syntax"]),
            ("List comprehensions are Pythonic", ["python", "best-practice"]),
            ("Type hints improve code quality", ["python", "typing"]),
            ("pytest is the recommended test framework", ["python", "testing"]),
            ("Black formats code consistently", ["python", "formatting"]),
        ]

        for content, tags in knowledge_items:
            await working_memory.store(
                create_entry(content, importance=0.75, tags=tags)
            )

        # Query for Python knowledge
        query = MemoryQuery(
            query_text="Python best practices",
            target_layers=[MemoryLayer.WORKING],
            top_k=10,
        )
        result = await working_memory.retrieve(query)

        assert len(result.entries) >= 3
        # All should be Python-related
        for entry in result.entries:
            assert "python" in entry.tags


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query(self, working_memory):
        """Test querying empty memory."""
        query = MemoryQuery(
            query_text="anything",
            target_layers=[MemoryLayer.WORKING],
            top_k=5,
        )
        result = await working_memory.retrieve(query)

        assert len(result.entries) == 0
        assert result.total_candidates == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, working_memory):
        """Test handling special characters."""
        special_content = "Code: print('Hello, World!') # Comment with émojis 🚀"
        entry = create_entry(special_content, importance=0.7)

        await working_memory.store(entry)

        query = MemoryQuery(
            query_text="code print",
            target_layers=[MemoryLayer.WORKING],
            top_k=5,
        )
        result = await working_memory.retrieve(query)

        assert len(result.entries) > 0
        assert result.entries[0].content == special_content

    @pytest.mark.asyncio
    async def test_very_long_content(self, working_memory):
        """Test handling very long content."""
        long_content = "Important fact. " * 500  # ~8000 chars
        entry = create_entry(long_content, importance=0.8)

        await working_memory.store(entry)

        assert len(working_memory.entries) == 1
        assert working_memory.entries[0].content == long_content

    @pytest.mark.asyncio
    async def test_duplicate_content_handling(self, working_memory):
        """Test storing duplicate content."""
        content = "This is duplicate content"

        entry1 = create_entry(content, importance=0.5)
        entry2 = create_entry(content, importance=0.7)

        await working_memory.store(entry1)
        await working_memory.store(entry2)

        # Both should be stored (different IDs)
        assert len(working_memory.entries) == 2

    @pytest.mark.asyncio
    async def test_rapid_store_retrieve_cycle(self, working_memory):
        """Test rapid store/retrieve operations."""
        for i in range(100):
            # Keep importance below threshold to avoid pinning
            entry = create_entry(f"Entry number {i}", importance=0.3 + (i % 3) * 0.05)
            await working_memory.store(entry)

            if i % 10 == 0:
                query = MemoryQuery(
                    query_text=f"entry {i}",
                    target_layers=[MemoryLayer.WORKING],
                    top_k=5,
                )
                await working_memory.retrieve(query)

        # Should handle all operations within limits
        # Note: entries includes both pinned and regular entries
        total_entries = len(working_memory.entries)
        regular_entries = len(working_memory._entries)
        assert regular_entries <= working_memory._max_entries
