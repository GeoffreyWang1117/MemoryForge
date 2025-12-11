"""Compression strategies for memory consolidation."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import structlog

from memoryforge.compression.pipeline import (
    CompressionConfig,
    CompressionLevel,
    CompressionStrategy,
)
from memoryforge.core.types import ImportanceScore, MemoryEntry, MemoryLayer

logger = structlog.get_logger()


class SummaryCompressor(CompressionStrategy):
    """Compresses memories by generating summaries.

    This strategy groups related memories and creates summary entries
    that capture the key information from multiple memories.
    """

    def __init__(self, summarizer: "LLMSummarizer | None" = None):
        """Initialize with optional LLM summarizer.

        Args:
            summarizer: LLM-based summarizer (if None, uses extractive summary)
        """
        self._summarizer = summarizer

    def can_compress(self, entries: list[MemoryEntry]) -> bool:
        """Check if we have enough entries to summarize."""
        return len(entries) >= 2

    async def compress(
        self,
        entries: list[MemoryEntry],
        config: CompressionConfig,
    ) -> list[MemoryEntry]:
        """Compress entries by summarizing them."""
        if len(entries) < 2:
            return entries

        # Group by tags for more coherent summaries
        groups = self._group_by_tags(entries)

        compressed = []
        for tag_key, group_entries in groups.items():
            if len(group_entries) == 1:
                compressed.append(group_entries[0])
                continue

            # Generate summary
            if self._summarizer:
                summary = await self._summarizer.summarize(
                    [e.content for e in group_entries]
                )
            else:
                summary = self._extractive_summary(group_entries, config.level)

            # Create compressed entry
            avg_importance = sum(
                e.importance.base_score for e in group_entries
            ) / len(group_entries)

            compressed_entry = MemoryEntry(
                id=uuid4(),
                content=summary,
                layer=MemoryLayer.EPISODIC,
                importance=ImportanceScore(
                    base_score=min(avg_importance + 0.1, 1.0),  # Boost slightly
                    access_count=sum(e.importance.access_count for e in group_entries),
                ),
                tags=list(set(tag for e in group_entries for tag in e.tags)) + ["compressed"],
                metadata={
                    "compression": {
                        "type": "summary",
                        "source_count": len(group_entries),
                        "source_ids": [str(e.id) for e in group_entries],
                        "compressed_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )
            compressed.append(compressed_entry)

        logger.debug(
            "Summary compression complete",
            original=len(entries),
            compressed=len(compressed),
        )
        return compressed

    def _group_by_tags(
        self,
        entries: list[MemoryEntry],
    ) -> dict[str, list[MemoryEntry]]:
        """Group entries by their primary tag."""
        groups: dict[str, list[MemoryEntry]] = defaultdict(list)

        for entry in entries:
            if entry.tags:
                key = entry.tags[0]
            else:
                key = "_untagged"
            groups[key].append(entry)

        return groups

    def _extractive_summary(
        self,
        entries: list[MemoryEntry],
        level: CompressionLevel,
    ) -> str:
        """Create an extractive summary without LLM."""
        # Sort by importance
        sorted_entries = sorted(
            entries,
            key=lambda e: e.importance.effective_score,
            reverse=True,
        )

        # Determine how many to keep based on level
        if level == CompressionLevel.LIGHT:
            keep_ratio = 0.7
        elif level == CompressionLevel.MODERATE:
            keep_ratio = 0.5
        else:  # AGGRESSIVE
            keep_ratio = 0.3

        keep_count = max(1, int(len(sorted_entries) * keep_ratio))
        kept = sorted_entries[:keep_count]

        # Build summary
        summary_parts = [
            f"[Consolidated from {len(entries)} memories]",
        ]

        for entry in kept:
            # Truncate long content
            content = entry.content
            if level == CompressionLevel.AGGRESSIVE and len(content) > 100:
                content = content[:100] + "..."
            elif level == CompressionLevel.MODERATE and len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"• {content}")

        return "\n".join(summary_parts)


class ClusterCompressor(CompressionStrategy):
    """Compresses memories by clustering similar content.

    Uses embedding similarity to group related memories and
    create representative entries for each cluster.
    """

    def __init__(
        self,
        embedder: "BaseEmbedder | None" = None,
        similarity_threshold: float = 0.8,
    ):
        """Initialize with optional embedder.

        Args:
            embedder: Embedding generator
            similarity_threshold: Minimum similarity to cluster together
        """
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold

    def can_compress(self, entries: list[MemoryEntry]) -> bool:
        """Check if we have enough entries and embeddings."""
        if len(entries) < 2:
            return False
        # Check if entries have embeddings or we have an embedder
        return self._embedder is not None or all(e.embedding for e in entries)

    async def compress(
        self,
        entries: list[MemoryEntry],
        config: CompressionConfig,
    ) -> list[MemoryEntry]:
        """Compress entries by clustering."""
        if len(entries) < 2:
            return entries

        # Get or generate embeddings
        embeddings = []
        for entry in entries:
            if entry.embedding:
                embeddings.append(entry.embedding)
            elif self._embedder:
                emb = await self._embedder.embed(entry.content)
                embeddings.append(emb)
            else:
                # Can't cluster without embeddings
                return entries

        # Simple clustering using cosine similarity
        clusters = self._cluster_by_similarity(entries, embeddings)

        # Create representative entry for each cluster
        compressed = []
        for cluster in clusters:
            if len(cluster) == 1:
                compressed.append(cluster[0])
            else:
                representative = self._create_representative(cluster, config.level)
                compressed.append(representative)

        logger.debug(
            "Cluster compression complete",
            original=len(entries),
            clusters=len(clusters),
            compressed=len(compressed),
        )
        return compressed

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _cluster_by_similarity(
        self,
        entries: list[MemoryEntry],
        embeddings: list[list[float]],
    ) -> list[list[MemoryEntry]]:
        """Cluster entries by embedding similarity."""
        n = len(entries)
        assigned = [False] * n
        clusters: list[list[MemoryEntry]] = []

        for i in range(n):
            if assigned[i]:
                continue

            cluster = [entries[i]]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue

                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._similarity_threshold:
                    cluster.append(entries[j])
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _create_representative(
        self,
        cluster: list[MemoryEntry],
        level: CompressionLevel,
    ) -> MemoryEntry:
        """Create a representative entry for a cluster."""
        # Find the most important entry as the base
        sorted_cluster = sorted(
            cluster,
            key=lambda e: e.importance.effective_score,
            reverse=True,
        )
        base_entry = sorted_cluster[0]

        # Combine content based on compression level
        if level == CompressionLevel.AGGRESSIVE:
            content = base_entry.content
        elif level == CompressionLevel.MODERATE:
            contents = [base_entry.content]
            for entry in sorted_cluster[1:2]:  # Add 1 more
                contents.append(f"Related: {entry.content[:100]}...")
            content = "\n".join(contents)
        else:  # LIGHT
            contents = [base_entry.content]
            for entry in sorted_cluster[1:]:
                contents.append(f"Related: {entry.content[:150]}")
            content = "\n".join(contents)

        # Average embeddings for the cluster
        avg_embedding = None
        if all(e.embedding for e in cluster):
            dim = len(cluster[0].embedding)
            avg_embedding = [0.0] * dim
            for entry in cluster:
                for i, val in enumerate(entry.embedding):
                    avg_embedding[i] += val
            avg_embedding = [v / len(cluster) for v in avg_embedding]

        return MemoryEntry(
            id=uuid4(),
            content=content,
            layer=MemoryLayer.EPISODIC,
            importance=ImportanceScore(
                base_score=max(e.importance.base_score for e in cluster),
                access_count=sum(e.importance.access_count for e in cluster),
            ),
            embedding=avg_embedding,
            tags=list(set(tag for e in cluster for tag in e.tags)) + ["clustered"],
            metadata={
                "compression": {
                    "type": "cluster",
                    "cluster_size": len(cluster),
                    "source_ids": [str(e.id) for e in cluster],
                    "compressed_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )


class TimeWindowCompressor(CompressionStrategy):
    """Compresses memories by time windows.

    Groups memories into time windows (e.g., hourly, daily) and
    creates consolidated entries for each window.
    """

    def __init__(self, window_hours: int = 24):
        """Initialize with time window size.

        Args:
            window_hours: Size of each time window in hours
        """
        self._window_hours = window_hours

    def can_compress(self, entries: list[MemoryEntry]) -> bool:
        """Check if entries span multiple time windows."""
        if len(entries) < 2:
            return False

        # Check if entries span at least one window
        if not entries:
            return False

        times = [e.created_at for e in entries]
        time_span = max(times) - min(times)
        return time_span >= timedelta(hours=self._window_hours)

    async def compress(
        self,
        entries: list[MemoryEntry],
        config: CompressionConfig,
    ) -> list[MemoryEntry]:
        """Compress entries by time window."""
        if len(entries) < 2:
            return entries

        # Group by time window
        windows = self._group_by_window(entries)

        compressed = []
        for window_key, window_entries in windows.items():
            if len(window_entries) == 1:
                compressed.append(window_entries[0])
            else:
                consolidated = self._consolidate_window(
                    window_key,
                    window_entries,
                    config.level,
                )
                compressed.append(consolidated)

        logger.debug(
            "Time window compression complete",
            original=len(entries),
            windows=len(windows),
            compressed=len(compressed),
        )
        return compressed

    def _group_by_window(
        self,
        entries: list[MemoryEntry],
    ) -> dict[str, list[MemoryEntry]]:
        """Group entries by time window."""
        windows: dict[str, list[MemoryEntry]] = defaultdict(list)

        for entry in entries:
            # Calculate window start time
            hours = entry.created_at.hour
            window_start = hours - (hours % self._window_hours)
            window_key = entry.created_at.strftime(f"%Y-%m-%d_{window_start:02d}h")
            windows[window_key].append(entry)

        return windows

    def _consolidate_window(
        self,
        window_key: str,
        entries: list[MemoryEntry],
        level: CompressionLevel,
    ) -> MemoryEntry:
        """Create a consolidated entry for a time window."""
        # Sort by importance
        sorted_entries = sorted(
            entries,
            key=lambda e: e.importance.effective_score,
            reverse=True,
        )

        # Build content based on compression level
        if level == CompressionLevel.AGGRESSIVE:
            # Just keep the most important
            content = f"[{window_key}] {sorted_entries[0].content}"
        elif level == CompressionLevel.MODERATE:
            # Keep top few
            parts = [f"[{window_key}] Session summary:"]
            for entry in sorted_entries[:3]:
                parts.append(f"• {entry.content[:150]}")
            content = "\n".join(parts)
        else:  # LIGHT
            parts = [f"[{window_key}] Session ({len(entries)} events):"]
            for entry in sorted_entries[:5]:
                parts.append(f"• {entry.content[:200]}")
            if len(sorted_entries) > 5:
                parts.append(f"... and {len(sorted_entries) - 5} more")
            content = "\n".join(parts)

        # Calculate average importance
        avg_importance = sum(e.importance.base_score for e in entries) / len(entries)

        # Get time range
        times = [e.created_at for e in entries]

        return MemoryEntry(
            id=uuid4(),
            content=content,
            layer=MemoryLayer.EPISODIC,
            importance=ImportanceScore(
                base_score=min(avg_importance + 0.1, 1.0),
                access_count=sum(e.importance.access_count for e in entries),
            ),
            tags=list(set(tag for e in entries for tag in e.tags)) + ["time_consolidated"],
            metadata={
                "compression": {
                    "type": "time_window",
                    "window": window_key,
                    "window_hours": self._window_hours,
                    "source_count": len(entries),
                    "source_ids": [str(e.id) for e in entries],
                    "time_range_start": min(times).isoformat(),
                    "time_range_end": max(times).isoformat(),
                    "compressed_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )


class DeduplicationCompressor(CompressionStrategy):
    """Removes duplicate or near-duplicate memories."""

    def __init__(self, similarity_threshold: float = 0.95):
        """Initialize with similarity threshold.

        Args:
            similarity_threshold: Threshold above which entries are considered duplicates
        """
        self._similarity_threshold = similarity_threshold

    def can_compress(self, entries: list[MemoryEntry]) -> bool:
        """Check if we have multiple entries to deduplicate."""
        return len(entries) >= 2

    async def compress(
        self,
        entries: list[MemoryEntry],
        config: CompressionConfig,
    ) -> list[MemoryEntry]:
        """Remove duplicate entries."""
        if len(entries) < 2:
            return entries

        # Use content hashing for exact duplicates
        seen_content: dict[str, MemoryEntry] = {}
        unique_entries: list[MemoryEntry] = []

        for entry in entries:
            # Normalize content for comparison
            normalized = entry.content.strip().lower()

            if normalized not in seen_content:
                seen_content[normalized] = entry
                unique_entries.append(entry)
            else:
                # Keep the one with higher importance
                existing = seen_content[normalized]
                if entry.importance.effective_score > existing.importance.effective_score:
                    unique_entries.remove(existing)
                    unique_entries.append(entry)
                    seen_content[normalized] = entry

        logger.debug(
            "Deduplication complete",
            original=len(entries),
            unique=len(unique_entries),
            removed=len(entries) - len(unique_entries),
        )
        return unique_entries
