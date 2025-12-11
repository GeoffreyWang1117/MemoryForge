# MemoryForge

Hierarchical Context Memory System for Multi-Agent LLM Collaboration.

## Problem

LLM context windows are limited. During long multi-agent collaboration sessions, early context is lost. Simple truncation causes information loss, while full retrieval introduces noise and increases token costs.

## Solution

MemoryForge implements a three-layer memory architecture:

| Layer | Purpose | Storage | Access Pattern |
|-------|---------|---------|----------------|
| **Working Memory** | Current task context | In-memory | Sliding window + importance scoring |
| **Episodic Memory** | Session history | Qdrant (Vector DB) | LLM summaries + semantic retrieval |
| **Semantic Memory** | Project knowledge | Neo4j (Graph DB) | Code structure + relationships |

## Installation

```bash
# Clone and setup
git clone https://github.com/your-org/memoryforge.git
cd memoryforge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

```python
import asyncio
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore, MemoryQuery

async def main():
    # Create working memory
    wm = WorkingMemory(max_entries=100, max_tokens=8000)

    # Store a memory
    entry = MemoryEntry(
        content="User wants to build a REST API",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.8),
        tags=["requirement"],
    )
    await wm.store(entry)

    # Query memories
    query = MemoryQuery(query_text="API", top_k=5)
    result = await wm.retrieve(query)

    for entry in result.entries:
        print(entry.content)

asyncio.run(main())
```

## Starting the Databases

```bash
# Start Qdrant and Neo4j
docker-compose up -d

# Verify services
curl http://localhost:6333/health  # Qdrant
curl http://localhost:7474         # Neo4j Browser
```

## Project Structure

```
memoryforge/
├── core/           # Base types and interfaces
├── memory/         # Memory layer implementations
│   ├── working/    # Sliding window + importance scoring
│   ├── episodic/   # LLM summaries + vector search
│   └── semantic/   # Knowledge graph
├── retrieval/      # Query routing
├── storage/        # Qdrant & Neo4j backends
├── llm/            # Embeddings & summarization
├── utils/          # Token counting, etc.
└── evaluation/     # Metrics & benchmarks
```

## Configuration

Environment variables (see `.env.example`):

```bash
# LLM Provider
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-...

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password
```

## Evaluation Metrics

- **Information Retention**: Recall of key facts after 100+ turns
- **Retrieval Accuracy**: Precision/recall of semantic search
- **Token Efficiency**: Compression ratio vs. full context

## Development

```bash
# Run tests
pytest tests/

# Type checking
mypy memoryforge

# Linting
ruff check memoryforge

# Format code
ruff format memoryforge
```

## License

MIT
