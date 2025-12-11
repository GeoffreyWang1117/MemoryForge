#!/usr/bin/env python3
"""Benchmark script for evaluating MemoryForge performance."""

import asyncio
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer, MemoryQuery
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.memory.batch import BatchMemoryOperations
from memoryforge.scoring.importance import RuleBasedScorer, ScoringContext

console = Console()


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""

    name: str
    operations: int
    total_time_ms: float
    ops_per_second: float
    avg_latency_ms: float


def generate_random_content(length: int = 100) -> str:
    """Generate random content for testing."""
    words = ["test", "memory", "api", "database", "function", "class", "module",
             "important", "decision", "requirement", "feature", "bug", "fix"]
    return " ".join(random.choices(words, k=length // 5))


def generate_realistic_content() -> str:
    """Generate realistic development conversation content."""
    templates = [
        "We need to implement {feature} for the {component}",
        "Let's use {tech} for {purpose}",
        "The {component} should handle {action} correctly",
        "Decision: {decision} for {reason}",
        "Bug fix: {issue} in {location}",
        "User requirement: {requirement}",
    ]

    features = ["authentication", "caching", "logging", "rate limiting", "validation"]
    components = ["API", "database", "frontend", "backend", "service"]
    techs = ["Redis", "PostgreSQL", "FastAPI", "React", "Docker"]
    purposes = ["caching", "storage", "routing", "rendering", "deployment"]

    template = random.choice(templates)
    return template.format(
        feature=random.choice(features),
        component=random.choice(components),
        tech=random.choice(techs),
        purpose=random.choice(purposes),
        action=random.choice(["errors", "requests", "responses", "events"]),
        decision=f"Use {random.choice(techs)}",
        reason=random.choice(["performance", "scalability", "simplicity"]),
        issue=random.choice(["null pointer", "timeout", "race condition"]),
        location=random.choice(["auth module", "data layer", "API handler"]),
        requirement=random.choice(["must support 1000 users", "needs real-time updates"]),
    )


async def benchmark_store(memory: WorkingMemory, n: int) -> BenchmarkResult:
    """Benchmark memory store operations."""
    entries = [
        MemoryEntry(
            content=generate_realistic_content(),
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=random.uniform(0.3, 0.9)),
        )
        for _ in range(n)
    ]

    start = time.perf_counter()
    for entry in entries:
        await memory.store(entry)
    elapsed = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        name="Store",
        operations=n,
        total_time_ms=elapsed,
        ops_per_second=n / (elapsed / 1000),
        avg_latency_ms=elapsed / n,
    )


async def benchmark_retrieve(memory: WorkingMemory, n: int) -> BenchmarkResult:
    """Benchmark memory retrieve operations."""
    queries = [
        MemoryQuery(
            query_text=random.choice(["database", "api", "authentication", "cache"]),
            target_layers=[MemoryLayer.WORKING],
            top_k=10,
        )
        for _ in range(n)
    ]

    start = time.perf_counter()
    for query in queries:
        await memory.retrieve(query)
    elapsed = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        name="Retrieve",
        operations=n,
        total_time_ms=elapsed,
        ops_per_second=n / (elapsed / 1000),
        avg_latency_ms=elapsed / n,
    )


async def benchmark_scoring(n: int) -> BenchmarkResult:
    """Benchmark importance scoring."""
    scorer = RuleBasedScorer()
    contents = [generate_realistic_content() for _ in range(n)]

    start = time.perf_counter()
    for content in contents:
        ctx = ScoringContext(content=content, role="user")
        await scorer.score(ctx)
    elapsed = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        name="Scoring (Rule)",
        operations=n,
        total_time_ms=elapsed,
        ops_per_second=n / (elapsed / 1000),
        avg_latency_ms=elapsed / n,
    )


async def benchmark_batch_store(memory: WorkingMemory, n: int) -> BenchmarkResult:
    """Benchmark batch store operations."""
    batch_ops = BatchMemoryOperations(memory, batch_size=50)
    contents = [generate_realistic_content() for _ in range(n)]

    start = time.perf_counter()
    await batch_ops.store_batch(contents, auto_score=True)
    elapsed = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        name="Batch Store",
        operations=n,
        total_time_ms=elapsed,
        ops_per_second=n / (elapsed / 1000),
        avg_latency_ms=elapsed / n,
    )


async def benchmark_retention(memory: WorkingMemory) -> dict:
    """Benchmark information retention across many operations."""
    # Store key facts
    key_facts = [
        ("FACT1: Database is PostgreSQL", 0.95),
        ("FACT2: API framework is FastAPI", 0.9),
        ("FACT3: Cache uses Redis", 0.85),
        ("FACT4: Auth is JWT-based", 0.9),
        ("FACT5: Deploy to Kubernetes", 0.8),
    ]

    for content, importance in key_facts:
        entry = MemoryEntry(
            content=content,
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=importance),
            tags=["key_fact"],
        )
        await memory.store(entry)

    # Add noise (many low-importance entries)
    for i in range(100):
        entry = MemoryEntry(
            content=f"Noise entry {i}: {generate_random_content(20)}",
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.3),
        )
        await memory.store(entry)

    # Try to retrieve key facts
    recalled = 0
    for fact, _ in key_facts:
        query = MemoryQuery(
            query_text=fact.split(":")[1].strip(),
            target_layers=[MemoryLayer.WORKING],
            top_k=5,
        )
        result = await memory.retrieve(query)

        for entry in result.entries:
            if fact in entry.content:
                recalled += 1
                break

    retention_rate = recalled / len(key_facts)

    return {
        "key_facts": len(key_facts),
        "noise_entries": 100,
        "recalled": recalled,
        "retention_rate": retention_rate,
    }


async def run_benchmarks():
    """Run all benchmarks."""
    console.print("\n[bold blue]MemoryForge Performance Benchmark[/bold blue]\n")

    results = []

    with Progress() as progress:
        task = progress.add_task("Running benchmarks...", total=5)

        # Store benchmark
        memory = WorkingMemory(max_entries=1000, max_tokens=50000)
        result = await benchmark_store(memory, 500)
        results.append(result)
        progress.advance(task)

        # Retrieve benchmark
        result = await benchmark_retrieve(memory, 500)
        results.append(result)
        progress.advance(task)

        # Scoring benchmark
        result = await benchmark_scoring(500)
        results.append(result)
        progress.advance(task)

        # Batch store benchmark
        memory2 = WorkingMemory(max_entries=1000, max_tokens=50000)
        result = await benchmark_batch_store(memory2, 500)
        results.append(result)
        progress.advance(task)

        # Retention test
        memory3 = WorkingMemory(max_entries=200, max_tokens=20000)
        retention = await benchmark_retention(memory3)
        progress.advance(task)

    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Operation", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Total (ms)", justify="right")
    table.add_column("Ops/sec", justify="right", style="green")
    table.add_column("Avg Latency (ms)", justify="right")

    for r in results:
        table.add_row(
            r.name,
            str(r.operations),
            f"{r.total_time_ms:.2f}",
            f"{r.ops_per_second:.0f}",
            f"{r.avg_latency_ms:.3f}",
        )

    console.print(table)

    # Retention results
    console.print("\n[bold]Information Retention Test:[/bold]")
    console.print(f"  Key facts stored: {retention['key_facts']}")
    console.print(f"  Noise entries: {retention['noise_entries']}")
    console.print(f"  Facts recalled: {retention['recalled']}")
    console.print(f"  Retention rate: [green]{retention['retention_rate']:.0%}[/green]")

    # Memory efficiency
    console.print("\n[bold]Memory Efficiency:[/bold]")
    console.print(f"  Working memory entries: {len(memory.entries)}")
    console.print(f"  Pinned entries: {len(memory._pinned)}")
    console.print(f"  Token count: {memory.token_count}")

    console.print("\n[green]Benchmark completed![/green]")


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
