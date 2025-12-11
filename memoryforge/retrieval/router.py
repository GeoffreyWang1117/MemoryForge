"""Query router for intelligent retrieval across memory layers."""

from enum import Enum

import structlog

from memoryforge.core.types import MemoryLayer, MemoryQuery, MemoryResult
from memoryforge.memory.manager import MemoryManager

logger = structlog.get_logger()


class QueryType(str, Enum):
    """Types of queries for routing."""

    FACTUAL = "factual"
    CONTEXTUAL = "contextual"
    STRUCTURAL = "structural"
    HISTORICAL = "historical"
    HYBRID = "hybrid"


class RetrievalRouter:
    """Routes queries to appropriate memory layers based on query analysis.

    Implements intelligent routing to minimize token usage while
    maximizing retrieval accuracy.
    """

    def __init__(self, memory_manager: MemoryManager):
        self._manager = memory_manager

        self._routing_rules: dict[QueryType, list[MemoryLayer]] = {
            QueryType.FACTUAL: [MemoryLayer.WORKING, MemoryLayer.SEMANTIC],
            QueryType.CONTEXTUAL: [MemoryLayer.WORKING, MemoryLayer.EPISODIC],
            QueryType.STRUCTURAL: [MemoryLayer.SEMANTIC],
            QueryType.HISTORICAL: [MemoryLayer.EPISODIC],
            QueryType.HYBRID: [MemoryLayer.WORKING, MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC],
        }

        self._structural_keywords = {
            "function", "class", "method", "api", "endpoint",
            "import", "module", "dependency", "type", "interface",
            "calls", "extends", "implements", "returns", "structure",
        }

        self._historical_keywords = {
            "before", "previously", "earlier", "history", "last time",
            "remember", "decided", "discussed", "mentioned", "said",
        }

    def classify_query(self, query_text: str) -> QueryType:
        """Classify query to determine routing strategy."""
        query_lower = query_text.lower()

        structural_score = sum(
            1 for kw in self._structural_keywords if kw in query_lower
        )
        historical_score = sum(
            1 for kw in self._historical_keywords if kw in query_lower
        )

        if structural_score >= 2:
            return QueryType.STRUCTURAL
        if historical_score >= 2:
            return QueryType.HISTORICAL
        if structural_score == 1 and historical_score == 1:
            return QueryType.HYBRID
        if "?" in query_text and len(query_text.split()) < 10:
            return QueryType.FACTUAL

        return QueryType.CONTEXTUAL

    async def route(self, query: MemoryQuery) -> MemoryResult:
        """Route query to appropriate memory layers."""
        query_type = self.classify_query(query.query_text)

        if not query.target_layers or query.target_layers == list(MemoryLayer):
            target_layers = self._routing_rules[query_type]
            query.target_layers = target_layers

        logger.debug(
            "Routing query",
            query_type=query_type.value,
            target_layers=[l.value for l in query.target_layers],
        )

        result = await self._manager.retrieve(query)

        logger.info(
            "Query completed",
            query_type=query_type.value,
            results_count=len(result.entries),
            query_time_ms=result.query_time_ms,
        )

        return result

    async def retrieve_for_prompt(
        self,
        query_text: str,
        max_tokens: int = 2000,
        min_score: float = 0.3,
    ) -> str:
        """Retrieve and format memories for LLM prompt injection."""
        query = MemoryQuery(
            query_text=query_text,
            top_k=20,
            min_importance=min_score,
        )

        result = await self.route(query)

        formatted_sections: dict[MemoryLayer, list[str]] = {
            MemoryLayer.WORKING: [],
            MemoryLayer.EPISODIC: [],
            MemoryLayer.SEMANTIC: [],
        }

        for entry, layer in zip(result.entries, result.layer_sources):
            formatted_sections[layer].append(f"- {entry.content}")

        output_parts = []
        token_count = 0
        chars_per_token = 4

        for layer, items in formatted_sections.items():
            if not items:
                continue

            section_header = {
                MemoryLayer.WORKING: "## Current Context",
                MemoryLayer.EPISODIC: "## Relevant History",
                MemoryLayer.SEMANTIC: "## Related Knowledge",
            }[layer]

            section_content = "\n".join(items)
            section_text = f"{section_header}\n{section_content}"
            section_tokens = len(section_text) // chars_per_token

            if token_count + section_tokens > max_tokens:
                available_chars = (max_tokens - token_count) * chars_per_token
                if available_chars > 100:
                    section_text = section_text[:available_chars] + "..."
                    output_parts.append(section_text)
                break

            output_parts.append(section_text)
            token_count += section_tokens

        return "\n\n".join(output_parts)

    def get_routing_stats(self) -> dict:
        """Get statistics about routing decisions."""
        return {
            "routing_rules": {
                qt.value: [l.value for l in layers]
                for qt, layers in self._routing_rules.items()
            },
        }
