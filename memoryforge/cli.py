"""Command-line interface for MemoryForge."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from memoryforge.analysis.extractor import CodeEntityExtractor
from memoryforge.analytics import MemoryAnalyzer
from memoryforge.config import get_settings
from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer, MemoryQuery
from memoryforge.export import MemoryExporter, ExportFormat
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.session import SessionManager, SessionConfig
from memoryforge.storage.sqlite import SQLiteMemoryStore

console = Console()
logger = structlog.get_logger()


def print_banner():
    """Print the MemoryForge banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗  ║
║   ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝  ║
║   ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝   ║
║   ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝    ║
║   ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║     ║
║   ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝     ║
║                                                              ║
║         Hierarchical Context Memory System v0.1.0            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold blue")


def cmd_analyze(args: list[str]) -> int:
    """Analyze a codebase and extract semantic entities."""
    if not args:
        console.print("[red]Error: Please provide a path to analyze[/red]")
        return 1

    path = Path(args[0])
    if not path.exists():
        console.print(f"[red]Error: Path not found: {path}[/red]")
        return 1

    async def run_analysis():
        extractor = CodeEntityExtractor()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing codebase...", total=None)
            summary = await extractor.analyze_codebase(path)
            progress.update(task, completed=True)

        # Display results
        table = Table(title="Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Files", str(summary.get("total_files", 0)))
        table.add_row("Total Classes", str(summary.get("total_classes", 0)))
        table.add_row("Total Functions", str(summary.get("total_functions", 0)))
        table.add_row("Async Functions", str(summary.get("async_functions", 0)))
        table.add_row("Methods", str(summary.get("methods", 0)))
        table.add_row("Total Imports", str(summary.get("total_imports", 0)))
        table.add_row("Total Entities", str(summary.get("total_entities", 0)))
        table.add_row("Total Relations", str(summary.get("total_relations", 0)))

        console.print(table)

        # Show entity types
        if summary.get("entity_types"):
            console.print("\n[bold]Entity Types:[/bold]")
            for etype, count in summary["entity_types"].items():
                console.print(f"  - {etype}: {count}")

        # Show relation types
        if summary.get("relation_types"):
            console.print("\n[bold]Relation Types:[/bold]")
            for rtype, count in summary["relation_types"].items():
                console.print(f"  - {rtype}: {count}")

        # Offer to generate documentation
        if len(args) > 1 and args[1] == "--docs":
            docs = extractor.generate_documentation()
            output_path = path / "CODE_DOCS.md" if path.is_dir() else path.with_suffix(".md")
            with open(output_path, "w") as f:
                f.write(docs)
            console.print(f"\n[green]Documentation saved to: {output_path}[/green]")

    asyncio.run(run_analysis())
    return 0


def cmd_memory(args: list[str]) -> int:
    """Interact with the memory system."""
    if not args:
        console.print("[yellow]Memory commands:[/yellow]")
        console.print("  store <content>   - Store a memory entry")
        console.print("  query <text>      - Query memories")
        console.print("  list              - List all memories")
        console.print("  clear             - Clear all memories")
        return 0

    subcmd = args[0]
    wm = WorkingMemory()

    async def run_memory_cmd():
        if subcmd == "store":
            if len(args) < 2:
                console.print("[red]Error: Please provide content to store[/red]")
                return 1

            content = " ".join(args[1:])
            entry = MemoryEntry(
                content=content,
                layer=MemoryLayer.WORKING,
                importance=ImportanceScore(base_score=0.7),
            )
            await wm.store(entry)
            console.print(f"[green]Stored memory: {entry.id}[/green]")

        elif subcmd == "query":
            if len(args) < 2:
                console.print("[red]Error: Please provide a query[/red]")
                return 1

            query_text = " ".join(args[1:])
            query = MemoryQuery(query_text=query_text, top_k=5)
            result = await wm.retrieve(query)

            if not result.entries:
                console.print("[yellow]No matching memories found[/yellow]")
            else:
                table = Table(title="Query Results")
                table.add_column("Score", style="cyan", width=8)
                table.add_column("Content", style="white")
                table.add_column("Tags", style="green")

                for entry, score in zip(result.entries, result.scores):
                    table.add_row(
                        f"{score:.2f}",
                        entry.content[:80] + "..." if len(entry.content) > 80 else entry.content,
                        ", ".join(entry.tags) if entry.tags else "-",
                    )

                console.print(table)

        elif subcmd == "list":
            entries = wm.entries
            if not entries:
                console.print("[yellow]No memories stored[/yellow]")
            else:
                table = Table(title=f"Memory Entries ({len(entries)} total)")
                table.add_column("ID", style="dim", width=12)
                table.add_column("Importance", style="cyan", width=10)
                table.add_column("Content", style="white")

                for entry in entries:
                    table.add_row(
                        str(entry.id)[:8],
                        f"{entry.importance.effective_score:.2f}",
                        entry.content[:60] + "..." if len(entry.content) > 60 else entry.content,
                    )

                console.print(table)

        elif subcmd == "clear":
            await wm.clear()
            console.print("[green]Memory cleared[/green]")

        else:
            console.print(f"[red]Unknown command: {subcmd}[/red]")
            return 1

        return 0

    return asyncio.run(run_memory_cmd())


def cmd_config(args: list[str]) -> int:
    """Show or modify configuration."""
    settings = get_settings()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("App Name", settings.app_name)
    table.add_row("Debug", str(settings.debug))
    table.add_row("Log Level", settings.log_level)
    table.add_row("Qdrant Host", f"{settings.qdrant.host}:{settings.qdrant.port}")
    table.add_row("Neo4j URI", settings.neo4j.uri)
    table.add_row("LLM Provider", settings.llm.provider)
    table.add_row("Embedding Model", settings.llm.embedding_model)
    table.add_row("Working Memory Max Entries", str(settings.memory.working_max_entries))
    table.add_row("Working Memory Max Tokens", str(settings.memory.working_max_tokens))

    console.print(table)
    return 0


def cmd_export(args: list[str]) -> int:
    """Export memories to various formats."""
    if not args:
        console.print("[yellow]Export commands:[/yellow]")
        console.print("  json <output>     - Export to JSON")
        console.print("  markdown <output> - Export to Markdown")
        console.print("  csv <output>      - Export to CSV")
        console.print("  html <output>     - Export to HTML")
        return 0

    format_arg = args[0].lower()
    format_map = {
        "json": ExportFormat.JSON,
        "markdown": ExportFormat.MARKDOWN,
        "md": ExportFormat.MARKDOWN,
        "csv": ExportFormat.CSV,
        "html": ExportFormat.HTML,
    }

    if format_arg not in format_map:
        console.print(f"[red]Unknown format: {format_arg}[/red]")
        return 1

    output_path = args[1] if len(args) > 1 else f"memories.{format_arg}"

    store = SQLiteMemoryStore("memoryforge.db")
    entries = store.get_all(limit=1000)

    if not entries:
        console.print("[yellow]No memories to export[/yellow]")
        return 0

    exporter = MemoryExporter()
    exporter.export_to_file(entries, output_path, format_map[format_arg])

    console.print(f"[green]Exported {len(entries)} memories to {output_path}[/green]")
    return 0


def cmd_session(args: list[str]) -> int:
    """Manage memory sessions."""
    if not args:
        console.print("[yellow]Session commands:[/yellow]")
        console.print("  list              - List all sessions")
        console.print("  create <name>     - Create a new session")
        console.print("  delete <id>       - Delete a session")
        console.print("  stats             - Show session statistics")
        return 0

    subcmd = args[0]
    store = SQLiteMemoryStore("memoryforge.db")

    if subcmd == "list":
        sessions = store.get_sessions()
        if not sessions:
            console.print("[yellow]No sessions found[/yellow]")
            return 0

        table = Table(title="Sessions")
        table.add_column("ID", style="dim", width=12)
        table.add_column("Name", style="cyan")
        table.add_column("Memories", style="green")
        table.add_column("Created", style="white")

        for session in sessions:
            table.add_row(
                session["id"][:8] + "...",
                session.get("name", "-") or "-",
                str(session.get("memory_count", 0)),
                session["created_at"][:16],
            )

        console.print(table)

    elif subcmd == "create":
        if len(args) < 2:
            console.print("[red]Error: Please provide a session name[/red]")
            return 1

        name = " ".join(args[1:])
        from uuid import uuid4
        session_id = str(uuid4())
        store.create_session(session_id, name)
        console.print(f"[green]Created session: {session_id[:8]}... ({name})[/green]")

    elif subcmd == "delete":
        if len(args) < 2:
            console.print("[red]Error: Please provide a session ID[/red]")
            return 1

        session_id = args[1]
        # Find matching session
        sessions = store.get_sessions()
        matching = [s for s in sessions if s["id"].startswith(session_id)]

        if not matching:
            console.print(f"[red]Session not found: {session_id}[/red]")
            return 1

        full_id = matching[0]["id"]
        if store.delete_session(full_id):
            console.print(f"[green]Deleted session: {full_id[:8]}...[/green]")
        else:
            console.print("[red]Failed to delete session[/red]")
            return 1

    elif subcmd == "stats":
        stats = store.get_stats()
        table = Table(title="Storage Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Memories", str(stats["total_memories"]))
        table.add_row("Sessions", str(stats["session_count"]))
        table.add_row("Avg Importance", f"{stats['avg_importance']:.3f}")
        table.add_row("Database Size", f"{stats['db_size_bytes'] / 1024:.1f} KB")

        for layer, count in stats.get("by_layer", {}).items():
            table.add_row(f"  {layer.capitalize()} Layer", str(count))

        console.print(table)

    else:
        console.print(f"[red]Unknown subcommand: {subcmd}[/red]")
        return 1

    return 0


def cmd_analytics(args: list[str]) -> int:
    """View memory analytics and insights."""
    store = SQLiteMemoryStore("memoryforge.db")
    entries = store.get_all(limit=1000)

    if not entries:
        console.print("[yellow]No memories to analyze[/yellow]")
        return 0

    analyzer = MemoryAnalyzer()
    report = analyzer.analyze(entries)
    recommendations = analyzer.get_recommendations(report)

    # Summary
    console.print(Panel.fit(
        f"[bold]Memory Analytics Report[/bold]\n\n"
        f"Total Memories: {report.total_memories}\n"
        f"Avg Importance: {report.avg_importance:.2f}\n"
        f"Avg Content Length: {report.avg_content_length:.0f} chars",
        title="Summary",
        border_style="blue",
    ))

    # Importance distribution
    console.print("\n[bold]Importance Distribution:[/bold]")
    for level, count in report.importance_distribution.items():
        bar = "█" * int(count / max(report.importance_distribution.values(), default=1) * 20)
        console.print(f"  {level.capitalize():10} {bar} {count}")

    # Top tags
    if report.top_tags:
        console.print("\n[bold]Top Tags:[/bold]")
        for tag, count in report.top_tags[:5]:
            console.print(f"  • {tag}: {count}")

    # Topic clusters
    if report.topic_clusters:
        console.print("\n[bold]Topic Clusters:[/bold]")
        for cluster in report.topic_clusters[:3]:
            console.print(f"  • {cluster.topic} ({cluster.memory_count} memories)")
            if cluster.keywords:
                console.print(f"    Related: {', '.join(cluster.keywords[:3])}")

    # Usage patterns
    if report.usage_patterns:
        console.print("\n[bold]Usage Patterns:[/bold]")
        for pattern in report.usage_patterns:
            console.print(f"  • {pattern.description}")

    # Recommendations
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in recommendations:
            priority_color = {"high": "red", "medium": "yellow", "low": "green"}.get(
                rec["priority"], "white"
            )
            console.print(f"  [{priority_color}]{rec['priority'].upper()}[/{priority_color}] {rec['title']}")
            console.print(f"       {rec['action']}")

    return 0


def cmd_compress(args: list[str]) -> int:
    """Compress old memories."""
    from memoryforge.compression import CompressionPipeline, CompressionConfig, SummaryCompressor

    store = SQLiteMemoryStore("memoryforge.db")
    entries = store.get_all(limit=1000)

    if not entries:
        console.print("[yellow]No memories to compress[/yellow]")
        return 0

    target = int(args[0]) if args else len(entries) // 2

    async def run_compression():
        config = CompressionConfig(min_age_hours=1, importance_threshold=0.9)
        pipeline = CompressionPipeline(config)
        pipeline.add_strategy(SummaryCompressor())

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Compressing memories...", total=None)
            result = await pipeline.auto_compress(entries, max_entries=target)
            progress.update(task, completed=True)

        console.print(f"\n[bold]Compression Results:[/bold]")
        console.print(f"  Original: {result.original_count} memories")
        console.print(f"  Compressed: {result.compressed_count} memories")
        console.print(f"  Removed: {len(result.removed_ids)} entries")
        console.print(f"  Token savings: {result.token_savings}")

        return result

    asyncio.run(run_compression())
    return 0


def cmd_help(args: list[str]) -> int:
    """Show help information."""
    help_text = """
## MemoryForge CLI

### Commands

| Command | Description |
|---------|-------------|
| `analyze <path>` | Analyze a codebase and extract entities |
| `memory <subcmd>` | Interact with memory system |
| `export <format>` | Export memories to various formats |
| `session <subcmd>` | Manage memory sessions |
| `analytics` | View memory analytics and insights |
| `compress [target]` | Compress old memories |
| `config` | Show current configuration |
| `help` | Show this help message |

### Memory Subcommands

| Subcommand | Description |
|------------|-------------|
| `store <content>` | Store a new memory entry |
| `query <text>` | Search memories by content |
| `list` | List all stored memories |
| `clear` | Clear all memories |

### Export Formats

| Format | Description |
|--------|-------------|
| `json` | JSON format with full metadata |
| `markdown` | Human-readable Markdown |
| `csv` | CSV spreadsheet format |
| `html` | Styled HTML document |

### Examples

```bash
# Analyze a Python project
memoryforge analyze ./myproject

# Store a memory
memoryforge memory store "Important decision: Use PostgreSQL"

# Export to markdown
memoryforge export markdown memories.md

# View analytics
memoryforge analytics

# Compress to 50 memories
memoryforge compress 50

# List sessions
memoryforge session list
```
"""
    console.print(Markdown(help_text))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    argv = argv or sys.argv[1:]

    if not argv:
        print_banner()
        return cmd_help([])

    command = argv[0]
    args = argv[1:]

    commands = {
        "analyze": cmd_analyze,
        "memory": cmd_memory,
        "export": cmd_export,
        "session": cmd_session,
        "analytics": cmd_analytics,
        "compress": cmd_compress,
        "config": cmd_config,
        "help": cmd_help,
        "--help": cmd_help,
        "-h": cmd_help,
    }

    if command in commands:
        return commands[command](args)
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Run 'memoryforge help' for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
