"""Evaluation metrics for the memory system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

from memoryforge.core.types import MemoryEntry, MemoryQuery
from memoryforge.memory.manager import MemoryManager

logger = structlog.get_logger()


@dataclass
class MetricResult:
    """Result of a metric evaluation."""

    metric_name: str
    score: float
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InformationRetentionMetric:
    """Measures how well key facts are retained across conversation turns.

    Tests whether important information from early turns can be
    recalled after many subsequent turns.
    """

    def __init__(self, memory_manager: MemoryManager):
        self._manager = memory_manager
        self._injected_facts: list[tuple[str, int]] = []

    def inject_fact(self, fact: str, turn_number: int) -> None:
        """Inject a fact at a specific turn for later testing."""
        self._injected_facts.append((fact, turn_number))

    async def evaluate(
        self,
        current_turn: int,
        retrieval_threshold: float = 0.5,
    ) -> MetricResult:
        """Evaluate information retention rate."""
        if not self._injected_facts:
            return MetricResult(
                metric_name="information_retention",
                score=1.0,
                details={"message": "No facts injected for testing"},
            )

        recalled = 0
        total = len(self._injected_facts)
        recall_details = []

        for fact, turn in self._injected_facts:
            query = MemoryQuery(
                query_text=fact,
                top_k=5,
                min_importance=0.0,
            )
            result = await self._manager.retrieve(query)

            fact_found = False
            best_score = 0.0

            for entry, score in zip(result.entries, result.scores):
                if score >= retrieval_threshold:
                    if fact.lower() in entry.content.lower():
                        fact_found = True
                        best_score = score
                        break
                    best_score = max(best_score, score)

            if fact_found:
                recalled += 1

            recall_details.append({
                "fact": fact[:50] + "..." if len(fact) > 50 else fact,
                "injected_turn": turn,
                "turns_elapsed": current_turn - turn,
                "recalled": fact_found,
                "best_score": best_score,
            })

        retention_rate = recalled / total if total > 0 else 1.0

        return MetricResult(
            metric_name="information_retention",
            score=retention_rate,
            details={
                "total_facts": total,
                "recalled_facts": recalled,
                "current_turn": current_turn,
                "recall_details": recall_details,
            },
        )


class RetrievalAccuracyMetric:
    """Measures retrieval precision and recall.

    Evaluates whether the right memories are retrieved for given queries.
    """

    def __init__(self, memory_manager: MemoryManager):
        self._manager = memory_manager

    async def evaluate(
        self,
        test_cases: list[tuple[str, list[str]]],
    ) -> MetricResult:
        """Evaluate retrieval accuracy.

        Args:
            test_cases: List of (query, expected_content_keywords) tuples
        """
        total_precision = 0.0
        total_recall = 0.0
        case_results = []

        for query_text, expected_keywords in test_cases:
            query = MemoryQuery(query_text=query_text, top_k=10)
            result = await self._manager.retrieve(query)

            retrieved_content = " ".join(e.content.lower() for e in result.entries)

            found_keywords = sum(
                1 for kw in expected_keywords if kw.lower() in retrieved_content
            )

            recall = found_keywords / len(expected_keywords) if expected_keywords else 1.0

            relevant_results = sum(
                1
                for e in result.entries
                if any(kw.lower() in e.content.lower() for kw in expected_keywords)
            )
            precision = relevant_results / len(result.entries) if result.entries else 0.0

            total_precision += precision
            total_recall += recall

            case_results.append({
                "query": query_text,
                "expected_keywords": expected_keywords,
                "precision": precision,
                "recall": recall,
                "results_count": len(result.entries),
            })

        n_cases = len(test_cases)
        avg_precision = total_precision / n_cases if n_cases > 0 else 0.0
        avg_recall = total_recall / n_cases if n_cases > 0 else 0.0

        f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )

        return MetricResult(
            metric_name="retrieval_accuracy",
            score=f1,
            details={
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": f1,
                "test_cases": len(test_cases),
                "case_results": case_results,
            },
        )


class TokenEfficiencyMetric:
    """Measures token usage efficiency.

    Compares token usage of the memory system against baselines.
    """

    def __init__(self, memory_manager: MemoryManager):
        self._manager = memory_manager

    async def evaluate(
        self,
        full_context_tokens: int,
        retrieved_tokens: int,
        task_success_rate: float,
    ) -> MetricResult:
        """Evaluate token efficiency.

        Args:
            full_context_tokens: Tokens needed for full context approach
            retrieved_tokens: Tokens actually used with memory system
            task_success_rate: Success rate on downstream tasks (0-1)
        """
        if full_context_tokens <= 0:
            return MetricResult(
                metric_name="token_efficiency",
                score=0.0,
                details={"error": "Invalid full context tokens"},
            )

        compression_ratio = 1 - (retrieved_tokens / full_context_tokens)

        efficiency_score = compression_ratio * task_success_rate

        tokens_saved = full_context_tokens - retrieved_tokens
        savings_percentage = (tokens_saved / full_context_tokens) * 100

        return MetricResult(
            metric_name="token_efficiency",
            score=efficiency_score,
            details={
                "compression_ratio": compression_ratio,
                "task_success_rate": task_success_rate,
                "full_context_tokens": full_context_tokens,
                "retrieved_tokens": retrieved_tokens,
                "tokens_saved": tokens_saved,
                "savings_percentage": savings_percentage,
            },
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    retention: MetricResult
    accuracy: MetricResult
    efficiency: MetricResult
    overall_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


async def run_benchmark(
    memory_manager: MemoryManager,
    retention_facts: list[tuple[str, int]],
    accuracy_test_cases: list[tuple[str, list[str]]],
    full_context_tokens: int,
    retrieved_tokens: int,
    task_success_rate: float,
    current_turn: int = 100,
) -> BenchmarkResult:
    """Run complete benchmark suite."""
    retention_metric = InformationRetentionMetric(memory_manager)
    for fact, turn in retention_facts:
        retention_metric.inject_fact(fact, turn)

    accuracy_metric = RetrievalAccuracyMetric(memory_manager)
    efficiency_metric = TokenEfficiencyMetric(memory_manager)

    retention_result = await retention_metric.evaluate(current_turn)
    accuracy_result = await accuracy_metric.evaluate(accuracy_test_cases)
    efficiency_result = await efficiency_metric.evaluate(
        full_context_tokens, retrieved_tokens, task_success_rate
    )

    overall_score = (
        retention_result.score * 0.4
        + accuracy_result.score * 0.4
        + efficiency_result.score * 0.2
    )

    return BenchmarkResult(
        retention=retention_result,
        accuracy=accuracy_result,
        efficiency=efficiency_result,
        overall_score=overall_score,
    )
