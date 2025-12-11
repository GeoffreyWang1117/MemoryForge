"""Export formats for memory data."""

import csv
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TextIO

import structlog

from memoryforge.core.types import MemoryEntry

logger = structlog.get_logger()


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    CSV = "csv"
    HTML = "html"


@dataclass
class ExportOptions:
    """Options for memory export."""

    # Include metadata
    include_metadata: bool = True

    # Include embeddings (can be large)
    include_embeddings: bool = False

    # Include timestamps
    include_timestamps: bool = True

    # Include importance scores
    include_importance: bool = True

    # Content truncation (0 = no truncation)
    max_content_length: int = 0

    # Custom title for export
    title: str = "MemoryForge Export"

    # Add statistics section
    include_stats: bool = True


class BaseExporter(ABC):
    """Base class for memory exporters."""

    def __init__(self, options: ExportOptions | None = None):
        self.options = options or ExportOptions()

    @abstractmethod
    def export(self, entries: list[MemoryEntry]) -> str:
        """Export memories to string format."""
        pass

    def export_to_file(self, entries: list[MemoryEntry], path: Path | str) -> None:
        """Export memories to a file."""
        content = self.export(entries)
        Path(path).write_text(content, encoding="utf-8")
        logger.info(f"Exported {len(entries)} memories to {path}")

    def _truncate_content(self, content: str) -> str:
        """Truncate content if needed."""
        if self.options.max_content_length > 0:
            if len(content) > self.options.max_content_length:
                return content[: self.options.max_content_length] + "..."
        return content

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for export."""
        return dt.isoformat()


class JSONExporter(BaseExporter):
    """Export memories to JSON format."""

    def export(self, entries: list[MemoryEntry]) -> str:
        """Export memories to JSON string."""
        data = {
            "version": "1.0",
            "format": "memoryforge",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "title": self.options.title,
            "count": len(entries),
        }

        if self.options.include_stats:
            data["stats"] = self._generate_stats(entries)

        memories = []
        for entry in entries:
            mem = {
                "id": str(entry.id),
                "content": self._truncate_content(entry.content),
                "layer": entry.layer.value,
                "tags": entry.tags,
            }

            if self.options.include_importance:
                mem["importance"] = {
                    "base_score": entry.importance.base_score,
                    "effective_score": entry.importance.effective_score,
                    "access_count": entry.importance.access_count,
                }

            if self.options.include_timestamps:
                mem["created_at"] = self._format_datetime(entry.created_at)
                mem["updated_at"] = self._format_datetime(entry.updated_at)

            if self.options.include_metadata:
                mem["metadata"] = entry.metadata

            if self.options.include_embeddings and entry.embedding:
                mem["embedding"] = entry.embedding

            memories.append(mem)

        data["memories"] = memories
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _generate_stats(self, entries: list[MemoryEntry]) -> dict:
        """Generate statistics for the export."""
        if not entries:
            return {"total": 0}

        from collections import Counter

        layer_counts = Counter(e.layer.value for e in entries)
        importances = [e.importance.effective_score for e in entries]

        return {
            "total": len(entries),
            "by_layer": dict(layer_counts),
            "avg_importance": round(sum(importances) / len(importances), 3),
            "tagged_count": sum(1 for e in entries if e.tags),
        }


class MarkdownExporter(BaseExporter):
    """Export memories to Markdown format."""

    def export(self, entries: list[MemoryEntry]) -> str:
        """Export memories to Markdown string."""
        lines = [
            f"# {self.options.title}",
            "",
            f"*Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            f"*Total memories: {len(entries)}*",
            "",
        ]

        if self.options.include_stats:
            lines.extend(self._generate_stats_section(entries))

        # Group by layer
        by_layer: dict[str, list[MemoryEntry]] = {}
        for entry in entries:
            layer = entry.layer.value
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append(entry)

        # Export each layer
        for layer, layer_entries in sorted(by_layer.items()):
            lines.append(f"## {layer.capitalize()} Memory")
            lines.append("")

            for i, entry in enumerate(layer_entries, 1):
                lines.extend(self._format_entry(entry, i))

        return "\n".join(lines)

    def _format_entry(self, entry: MemoryEntry, index: int) -> list[str]:
        """Format a single memory entry."""
        lines = [f"### {index}. Memory `{str(entry.id)[:8]}...`", ""]

        # Content
        content = self._truncate_content(entry.content)
        lines.append(content)
        lines.append("")

        # Details
        details = []

        if self.options.include_importance:
            score = entry.importance.effective_score
            details.append(f"**Importance:** {score:.2f}")

        if entry.tags:
            tags_str = ", ".join(f"`{t}`" for t in entry.tags)
            details.append(f"**Tags:** {tags_str}")

        if self.options.include_timestamps:
            details.append(
                f"**Created:** {entry.created_at.strftime('%Y-%m-%d %H:%M')}"
            )

        if details:
            lines.append(" | ".join(details))
            lines.append("")

        lines.append("---")
        lines.append("")

        return lines

    def _generate_stats_section(self, entries: list[MemoryEntry]) -> list[str]:
        """Generate statistics section."""
        if not entries:
            return []

        from collections import Counter

        layer_counts = Counter(e.layer.value for e in entries)
        importances = [e.importance.effective_score for e in entries]

        lines = [
            "## Statistics",
            "",
            f"- **Total memories:** {len(entries)}",
            f"- **Average importance:** {sum(importances) / len(importances):.2f}",
            "",
            "### By Layer",
            "",
        ]

        for layer, count in sorted(layer_counts.items()):
            lines.append(f"- {layer.capitalize()}: {count}")

        lines.extend(["", "---", ""])
        return lines


class CSVExporter(BaseExporter):
    """Export memories to CSV format."""

    def export(self, entries: list[MemoryEntry]) -> str:
        """Export memories to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        # Header
        headers = ["id", "content", "layer"]

        if self.options.include_importance:
            headers.extend(["importance_base", "importance_effective", "access_count"])

        headers.append("tags")

        if self.options.include_timestamps:
            headers.extend(["created_at", "updated_at"])

        if self.options.include_metadata:
            headers.append("metadata")

        writer.writerow(headers)

        # Data rows
        for entry in entries:
            row = [
                str(entry.id),
                self._truncate_content(entry.content),
                entry.layer.value,
            ]

            if self.options.include_importance:
                row.extend([
                    entry.importance.base_score,
                    entry.importance.effective_score,
                    entry.importance.access_count,
                ])

            row.append(";".join(entry.tags))

            if self.options.include_timestamps:
                row.extend([
                    self._format_datetime(entry.created_at),
                    self._format_datetime(entry.updated_at),
                ])

            if self.options.include_metadata:
                row.append(json.dumps(entry.metadata))

            writer.writerow(row)

        return output.getvalue()


class HTMLExporter(BaseExporter):
    """Export memories to HTML format."""

    def export(self, entries: list[MemoryEntry]) -> str:
        """Export memories to HTML string."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"  <title>{self.options.title}</title>",
            "  <style>",
            self._get_styles(),
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>{self.options.title}</h1>",
            f"  <p class='meta'>Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>",
            f"  <p class='meta'>Total memories: {len(entries)}</p>",
        ]

        if self.options.include_stats:
            html_parts.append(self._generate_stats_html(entries))

        # Memory cards
        html_parts.append("  <div class='memories'>")

        for entry in entries:
            html_parts.append(self._format_entry_html(entry))

        html_parts.extend([
            "  </div>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def _get_styles(self) -> str:
        """Get CSS styles for HTML export."""
        return """
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 20px;
      background: #f5f5f5;
    }
    h1 { color: #333; }
    .meta { color: #666; font-size: 0.9em; }
    .stats {
      background: white;
      padding: 15px;
      border-radius: 8px;
      margin: 20px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .memory {
      background: white;
      padding: 20px;
      margin: 15px 0;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .memory-id { font-size: 0.8em; color: #999; }
    .memory-content {
      margin: 10px 0;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .memory-meta { display: flex; gap: 15px; flex-wrap: wrap; font-size: 0.85em; }
    .tag {
      display: inline-block;
      background: #e0e7ff;
      color: #3730a3;
      padding: 2px 8px;
      border-radius: 4px;
      margin-right: 5px;
    }
    .importance {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .importance-high { background: #fee2e2; color: #991b1b; }
    .importance-medium { background: #fef3c7; color: #92400e; }
    .importance-low { background: #d1fae5; color: #065f46; }
    .layer {
      text-transform: uppercase;
      font-size: 0.75em;
      font-weight: bold;
      color: #6b7280;
    }
"""

    def _format_entry_html(self, entry: MemoryEntry) -> str:
        """Format a single entry as HTML."""
        content = self._truncate_content(entry.content)
        # Escape HTML
        content = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        importance = entry.importance.effective_score
        if importance >= 0.7:
            imp_class = "importance-high"
        elif importance >= 0.4:
            imp_class = "importance-medium"
        else:
            imp_class = "importance-low"

        tags_html = "".join(f"<span class='tag'>{t}</span>" for t in entry.tags)

        parts = [
            f"    <div class='memory'>",
            f"      <span class='layer'>{entry.layer.value}</span>",
            f"      <span class='memory-id'>ID: {str(entry.id)[:8]}...</span>",
            f"      <div class='memory-content'>{content}</div>",
            f"      <div class='memory-meta'>",
        ]

        if self.options.include_importance:
            parts.append(
                f"        <span class='importance {imp_class}'>Importance: {importance:.2f}</span>"
            )

        if entry.tags:
            parts.append(f"        <span>{tags_html}</span>")

        if self.options.include_timestamps:
            parts.append(
                f"        <span>Created: {entry.created_at.strftime('%Y-%m-%d %H:%M')}</span>"
            )

        parts.extend([
            f"      </div>",
            f"    </div>",
        ])

        return "\n".join(parts)

    def _generate_stats_html(self, entries: list[MemoryEntry]) -> str:
        """Generate statistics as HTML."""
        if not entries:
            return ""

        from collections import Counter

        layer_counts = Counter(e.layer.value for e in entries)
        importances = [e.importance.effective_score for e in entries]
        avg_imp = sum(importances) / len(importances)

        layers_html = ", ".join(
            f"{layer}: {count}" for layer, count in sorted(layer_counts.items())
        )

        return f"""
  <div class='stats'>
    <strong>Statistics:</strong>
    Average importance: {avg_imp:.2f} |
    Layers: {layers_html}
  </div>
"""


class MemoryExporter:
    """Unified interface for memory export.

    Supports multiple export formats and provides a simple API.
    """

    _exporters = {
        ExportFormat.JSON: JSONExporter,
        ExportFormat.MARKDOWN: MarkdownExporter,
        ExportFormat.CSV: CSVExporter,
        ExportFormat.HTML: HTMLExporter,
    }

    def __init__(self, options: ExportOptions | None = None):
        """Initialize exporter.

        Args:
            options: Export options (applies to all formats)
        """
        self.options = options or ExportOptions()

    def export(
        self,
        entries: list[MemoryEntry],
        format: ExportFormat = ExportFormat.JSON,
    ) -> str:
        """Export memories to specified format.

        Args:
            entries: Memory entries to export
            format: Export format

        Returns:
            Exported content as string
        """
        exporter_class = self._exporters.get(format)
        if not exporter_class:
            raise ValueError(f"Unsupported format: {format}")

        exporter = exporter_class(self.options)
        return exporter.export(entries)

    def export_to_file(
        self,
        entries: list[MemoryEntry],
        path: Path | str,
        format: ExportFormat | None = None,
    ) -> None:
        """Export memories to a file.

        Args:
            entries: Memory entries to export
            path: Output file path
            format: Export format (auto-detected from extension if not provided)
        """
        path = Path(path)

        # Auto-detect format from extension
        if format is None:
            ext = path.suffix.lower()
            format_map = {
                ".json": ExportFormat.JSON,
                ".md": ExportFormat.MARKDOWN,
                ".markdown": ExportFormat.MARKDOWN,
                ".csv": ExportFormat.CSV,
                ".html": ExportFormat.HTML,
                ".htm": ExportFormat.HTML,
            }
            format = format_map.get(ext, ExportFormat.JSON)

        content = self.export(entries, format)
        path.write_text(content, encoding="utf-8")

        logger.info(
            "Memories exported",
            path=str(path),
            format=format.value,
            count=len(entries),
        )

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """Get list of supported export formats."""
        return [f.value for f in ExportFormat]
