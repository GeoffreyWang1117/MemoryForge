#!/usr/bin/env python3
"""Interactive conversation demo with memory integration."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memoryforge.conversation import ConversationManager

console = Console()


def print_banner():
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           MemoryForge - Interactive Conversation Demo         ║
╠═══════════════════════════════════════════════════════════════╣
║  Commands:                                                    ║
║    /memory <query>  - Search your memories                    ║
║    /add <content>   - Manually add a memory                   ║
║    /stats           - Show session statistics                 ║
║    /clear           - Clear all memories                      ║
║    /quit            - Exit the demo                           ║
╚═══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


async def main():
    print_banner()

    # Check for API key
    api_key = os.getenv("LLM_OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: LLM_OPENAI_API_KEY not set in environment[/red]")
        return

    # Initialize conversation manager
    console.print("[dim]Initializing memory system...[/dim]")
    manager = ConversationManager(
        api_key=api_key,
        model="gpt-4o-mini",
        max_context_tokens=4000,
        auto_score_importance=True,
    )
    console.print(f"[green]Session started: {manager.session_id[:8]}...[/green]\n")

    # Pre-populate with some context
    await manager.add_memory(
        "User is building a hierarchical memory system for LLM agents",
        importance=0.9,
        tags=["project", "context"],
    )

    while True:
        try:
            # Get user input
            console.print("[bold blue]You:[/bold blue] ", end="")
            user_input = input().strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input[1:].split(" ", 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command == "quit" or command == "exit":
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break

                elif command == "memory":
                    if not args:
                        console.print("[yellow]Usage: /memory <query>[/yellow]")
                        continue

                    results = await manager.query_memory(args, top_k=5)
                    if results:
                        table = Table(title="Memory Search Results")
                        table.add_column("Score", style="cyan", width=8)
                        table.add_column("Content", style="white")
                        table.add_column("Tags", style="green")

                        for r in results:
                            table.add_row(
                                f"{r['importance']:.2f}",
                                r["content"][:60] + "..." if len(r["content"]) > 60 else r["content"],
                                ", ".join(r["tags"]),
                            )
                        console.print(table)
                    else:
                        console.print("[yellow]No matching memories found[/yellow]")

                elif command == "add":
                    if not args:
                        console.print("[yellow]Usage: /add <content>[/yellow]")
                        continue

                    entry_id = await manager.add_memory(args, importance=0.8)
                    console.print(f"[green]Memory added: {entry_id[:8]}...[/green]")

                elif command == "stats":
                    stats = manager.get_stats()
                    table = Table(title="Session Statistics")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")

                    table.add_row("Session ID", stats["session_id"][:16] + "...")
                    table.add_row("Turn Count", str(stats["turn_count"]))
                    table.add_row("Total Memories", str(stats["total_memories"]))
                    table.add_row("Pinned Memories", str(stats["pinned_memories"]))
                    table.add_row("Est. Token Usage", str(stats["token_count"]))

                    console.print(table)

                elif command == "clear":
                    await manager.clear_memory()
                    console.print("[green]Memory cleared[/green]")

                else:
                    console.print(f"[red]Unknown command: {command}[/red]")

                continue

            # Regular conversation
            console.print("[bold green]Assistant:[/bold green] ", end="")

            # Stream the response
            async for chunk in manager.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print()  # New line after response

            # Show memory stats periodically
            if manager.turn_count % 5 == 0:
                stats = manager.get_stats()
                console.print(
                    f"\n[dim]Memory: {stats['total_memories']} entries, "
                    f"{stats['pinned_memories']} pinned, "
                    f"~{stats['token_count']} tokens[/dim]\n"
                )

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())
