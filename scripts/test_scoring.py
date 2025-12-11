#!/usr/bin/env python3
"""Test the importance scoring algorithms."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memoryforge.scoring.importance import (
    ScoringContext,
    RuleBasedScorer,
    LLMScorer,
    HybridScorer,
)

console = Console()


async def main():
    console.print("[bold]Testing Importance Scoring Algorithms[/bold]\n")

    # Test cases with expected importance levels
    test_cases = [
        # High importance - decisions and requirements
        ("We decided to use PostgreSQL for the database", "user", "high"),
        ("The API must support JWT authentication", "user", "high"),
        ("Here's the database schema for the users table", "assistant", "high"),

        # Medium importance - context and explanations
        ("Because we need scalability, consider NoSQL options", "assistant", "medium"),
        ("Let me explain how the caching works", "assistant", "medium"),
        ("I'm thinking about using Redis for sessions", "user", "medium"),

        # Low importance - casual and acknowledgments
        ("Okay, sounds good", "user", "low"),
        ("Got it, thanks!", "user", "low"),
        ("Hello, how can I help you today?", "assistant", "low"),
    ]

    rule_scorer = RuleBasedScorer()

    # Test rule-based scorer
    console.print("[bold cyan]Rule-Based Scorer Results:[/bold cyan]\n")

    table = Table()
    table.add_column("Content", style="white", width=50)
    table.add_column("Role", style="cyan", width=10)
    table.add_column("Expected", style="yellow", width=8)
    table.add_column("Score", style="green", width=8)
    table.add_column("Confidence", width=10)

    for content, role, expected in test_cases:
        ctx = ScoringContext(content=content, role=role)
        result = await rule_scorer.score(ctx)

        # Determine actual level
        if result.score >= 0.7:
            actual = "high"
        elif result.score >= 0.5:
            actual = "medium"
        else:
            actual = "low"

        match = "✓" if actual == expected else "✗"

        table.add_row(
            content[:48] + "..." if len(content) > 48 else content,
            role,
            expected,
            f"{result.score:.2f} {match}",
            f"{result.confidence:.2f}",
        )

    console.print(table)

    # Test LLM scorer with one example
    api_key = os.getenv("LLM_OPENAI_API_KEY")
    if api_key:
        console.print("\n[bold cyan]LLM Scorer Test:[/bold cyan]")

        llm_scorer = LLMScorer(api_key=api_key)
        test_content = "We need to implement rate limiting using Redis with a 100 requests/minute limit per user"

        ctx = ScoringContext(content=test_content, role="user")
        result = await llm_scorer.score(ctx)

        console.print(f"\nContent: {test_content}")
        console.print(f"Score: {result.score:.2f}")
        console.print(f"Confidence: {result.confidence:.2f}")
        console.print(f"Reasoning: {result.reasoning}")

        # Test hybrid scorer
        console.print("\n[bold cyan]Hybrid Scorer Test:[/bold cyan]")

        hybrid_scorer = HybridScorer(api_key=api_key)

        test_cases_hybrid = [
            "Let's use FastAPI for the backend",  # Clear high
            "Maybe we should consider caching",  # Ambiguous
            "OK",  # Clear low
        ]

        for content in test_cases_hybrid:
            ctx = ScoringContext(content=content, role="user")
            result = await hybrid_scorer.score(ctx)

            console.print(f"\n'{content}'")
            console.print(f"  Score: {result.score:.2f}, Method: {result.factors.get('method', 'unknown')}")
            if 'rule_score' in result.factors:
                console.print(f"  Rule: {result.factors['rule_score']:.2f}, LLM: {result.factors['llm_score']:.2f}")

    console.print("\n[green]Scoring tests completed![/green]")


if __name__ == "__main__":
    asyncio.run(main())
