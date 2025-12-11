#!/usr/bin/env python3
"""Test the conversation system with memory integration."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memoryforge.conversation import ConversationManager

console = Console()


async def main():
    console.print("[bold]Testing Conversation System with Memory[/bold]\n")

    api_key = os.getenv("LLM_OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: LLM_OPENAI_API_KEY not set[/red]")
        return

    manager = ConversationManager(
        api_key=api_key,
        model="gpt-4o-mini",
        auto_score_importance=True,
    )

    console.print(f"Session: {manager.session_id[:8]}...\n")

    # Simulate a development conversation
    conversation = [
        "I want to build a task management API",
        "Let's use FastAPI with PostgreSQL",
        "We should add user authentication with JWT",
        "What database schema should we use for tasks?",
        "Let's also add priority levels: low, medium, high, urgent",
        "Can you remind me what tech stack we decided on?",
    ]

    for i, message in enumerate(conversation, 1):
        console.print(f"\n[bold blue]Turn {i} - User:[/bold blue] {message}")

        response = await manager.chat(message)

        console.print(f"[bold green]Assistant:[/bold green] {response[:200]}...")

        # Show memory stats
        stats = manager.get_stats()
        console.print(
            f"[dim]Memories: {stats['total_memories']}, "
            f"Pinned: {stats['pinned_memories']}[/dim]"
        )

    # Query memory to verify retention
    console.print("\n" + "=" * 60)
    console.print("[bold]Memory Retention Test[/bold]")
    console.print("=" * 60)

    queries = ["database", "authentication", "tech stack", "priority"]

    for query in queries:
        results = await manager.query_memory(query, top_k=3)
        console.print(f"\n[cyan]Query: '{query}'[/cyan]")
        if results:
            for r in results:
                console.print(f"  [{r['importance']:.2f}] {r['content'][:60]}...")
        else:
            console.print("  [yellow]No results[/yellow]")

    # Final stats
    console.print("\n" + "=" * 60)
    console.print("[bold]Final Session Statistics[/bold]")
    console.print("=" * 60)

    stats = manager.get_stats()
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Turns", str(stats["turn_count"]))
    table.add_row("Total Memories", str(stats["total_memories"]))
    table.add_row("Pinned (High Importance)", str(stats["pinned_memories"]))
    table.add_row("Estimated Tokens", str(stats["token_count"]))

    console.print(table)
    console.print("\n[green]Test completed successfully![/green]")


if __name__ == "__main__":
    asyncio.run(main())
