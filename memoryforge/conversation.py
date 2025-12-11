"""End-to-end conversation manager with memory integration."""

import os
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import uuid4

import structlog
from openai import AsyncOpenAI

from memoryforge.core.types import (
    ConversationTurn,
    ImportanceScore,
    MemoryEntry,
    MemoryLayer,
    MemoryQuery,
)
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.llm.embedder import SentenceTransformerEmbedder
from memoryforge.llm.summarizer import LLMSummarizer

logger = structlog.get_logger()


class ImportanceScorer:
    """Scores the importance of conversation content."""

    def __init__(self, api_key: str | None = None):
        self._client = AsyncOpenAI(api_key=api_key)

    async def score(self, content: str, context: str = "") -> float:
        """Score the importance of content (0.0 to 1.0)."""
        prompt = f"""Rate the importance of the following content for future reference in a software development context.

Context (if any): {context[:500] if context else "None"}

Content to rate: {content}

Consider:
- Is this a key decision or requirement?
- Does it contain technical specifications?
- Is it a user preference or constraint?
- Would forgetting this cause problems later?

Respond with ONLY a number between 0.0 and 1.0, where:
- 0.0-0.3: Low importance (casual chat, acknowledgments)
- 0.4-0.6: Medium importance (context, clarifications)
- 0.7-0.9: High importance (decisions, requirements, specs)
- 1.0: Critical (must never forget)

Score:"""

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except (ValueError, Exception) as e:
            logger.warning("Failed to score importance", error=str(e))
            return 0.5


class ConversationManager:
    """Manages conversations with integrated memory."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_context_tokens: int = 4000,
        auto_score_importance: bool = True,
    ):
        self._api_key = api_key or os.getenv("LLM_OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=self._api_key)
        self._model = model
        self._max_context_tokens = max_context_tokens
        self._auto_score = auto_score_importance

        # Memory components
        self._working_memory = WorkingMemory(
            max_entries=100,
            max_tokens=max_context_tokens,
            importance_threshold=0.7,
        )
        self._importance_scorer = ImportanceScorer(api_key=self._api_key)

        # Conversation state
        self._turns: list[ConversationTurn] = []
        self._session_id = str(uuid4())
        self._turn_count = 0

        # System prompt
        self._system_prompt = """You are a helpful AI assistant with access to a hierarchical memory system.
You can remember important information from our conversation and retrieve it when relevant.
Be concise and helpful. When making decisions or noting requirements, be explicit so they can be remembered."""

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    async def chat(self, user_message: str) -> str:
        """Process a user message and generate a response."""
        self._turn_count += 1

        # Store user message in memory
        await self._store_turn("user", user_message)

        # Retrieve relevant context from memory
        context = await self._get_relevant_context(user_message)

        # Build messages for LLM
        messages = self._build_messages(user_message, context)

        # Generate response
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
        )

        assistant_message = response.choices[0].message.content

        # Store assistant response in memory
        await self._store_turn("assistant", assistant_message)

        logger.info(
            "Completed conversation turn",
            turn=self._turn_count,
            user_tokens=len(user_message) // 4,
            response_tokens=len(assistant_message) // 4,
            context_entries=len(self._working_memory.entries),
        )

        return assistant_message

    async def chat_stream(self, user_message: str) -> AsyncIterator[str]:
        """Process a user message and stream the response."""
        self._turn_count += 1

        # Store user message
        await self._store_turn("user", user_message)

        # Get context
        context = await self._get_relevant_context(user_message)

        # Build messages
        messages = self._build_messages(user_message, context)

        # Stream response
        full_response = ""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # Store complete response
        await self._store_turn("assistant", full_response)

    async def _store_turn(self, role: str, content: str) -> None:
        """Store a conversation turn in memory."""
        # Calculate importance
        if self._auto_score:
            context = "\n".join(
                f"{t.role}: {t.content[:100]}" for t in self._turns[-5:]
            )
            importance = await self._importance_scorer.score(content, context)
        else:
            importance = 0.5

        # Create memory entry
        entry = MemoryEntry(
            content=f"[{role.upper()}] {content}",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=importance),
            tags=[role, f"turn_{self._turn_count}"],
            metadata={
                "role": role,
                "turn": self._turn_count,
                "session_id": self._session_id,
            },
        )

        await self._working_memory.store(entry)

        # Also store as conversation turn
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
            metadata={"turn": self._turn_count},
            token_count=len(content) // 4,
        )
        self._turns.append(turn)

        logger.debug(
            "Stored turn",
            role=role,
            importance=importance,
            content_preview=content[:50],
        )

    async def _get_relevant_context(self, query: str) -> list[MemoryEntry]:
        """Retrieve relevant context from memory."""
        memory_query = MemoryQuery(
            query_text=query,
            target_layers=[MemoryLayer.WORKING],
            top_k=10,
            min_importance=0.3,
        )

        result = await self._working_memory.retrieve(memory_query)
        return result.entries

    def _build_messages(
        self, user_message: str, context: list[MemoryEntry]
    ) -> list[dict]:
        """Build the message list for the LLM."""
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add context from memory
        if context:
            context_text = "\n".join(
                f"- {e.content}" for e in context[:5]
            )
            messages.append({
                "role": "system",
                "content": f"Relevant context from memory:\n{context_text}",
            })

        # Add recent conversation history
        for turn in self._turns[-6:]:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    async def add_memory(
        self, content: str, importance: float = 0.8, tags: list[str] | None = None
    ) -> str:
        """Manually add a memory entry."""
        entry = MemoryEntry(
            content=content,
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=importance),
            tags=tags or ["manual"],
            metadata={"session_id": self._session_id, "manual": True},
        )

        await self._working_memory.store(entry)
        return str(entry.id)

    async def query_memory(self, query: str, top_k: int = 5) -> list[dict]:
        """Query the memory system."""
        memory_query = MemoryQuery(
            query_text=query,
            target_layers=[MemoryLayer.WORKING],
            top_k=top_k,
        )

        result = await self._working_memory.retrieve(memory_query)

        return [
            {
                "content": entry.content,
                "importance": entry.importance.effective_score,
                "tags": entry.tags,
            }
            for entry in result.entries
        ]

    def get_stats(self) -> dict:
        """Get conversation and memory statistics."""
        return {
            "session_id": self._session_id,
            "turn_count": self._turn_count,
            "total_memories": len(self._working_memory.entries),
            "pinned_memories": len(self._working_memory._pinned),
            "token_count": self._working_memory.token_count,
        }

    async def clear_memory(self) -> None:
        """Clear all memory."""
        await self._working_memory.clear()
        self._turns.clear()
        logger.info("Cleared conversation memory")
