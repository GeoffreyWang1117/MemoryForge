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

## REST API

Start the API server:

```bash
# Using uvicorn
uvicorn memoryforge.api.app:app --reload --port 8000

# Or using the Makefile
make serve
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/memory/store` | Store a memory |
| POST | `/api/v1/memory/query` | Query memories |
| GET | `/api/v1/memory/list` | List all memories |
| DELETE | `/api/v1/memory/{id}` | Delete a memory |
| GET | `/api/v1/memory/stats` | Memory statistics |
| POST | `/api/v1/persistence/save` | Save to SQLite |
| GET | `/api/v1/persistence/load` | Load from SQLite |

### Example API Usage

```bash
# Store a memory
curl -X POST http://localhost:8000/api/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers Python", "importance": 0.8, "tags": ["preference"]}'

# Query memories
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "python", "top_k": 5}'

# Get statistics
curl http://localhost:8000/api/v1/memory/stats
```

## CLI Commands

MemoryForge includes a rich CLI for interactive use:

```bash
# Run the CLI
python -m memoryforge.cli

# Available commands:
#   analyze <path>    - Analyze a codebase and extract entities
#   store <content>   - Store a memory entry
#   query <text>      - Query memories
#   list              - List all memories
#   stats             - Show memory statistics
#   export <format>   - Export memories (json/csv/markdown)
#   session           - Manage sessions
#   help              - Show help
```

### CLI Examples

```bash
# Analyze your codebase
python -m memoryforge.cli analyze ./src --docs

# Store a memory
python -m memoryforge.cli store "Important: Use async/await for all IO"

# Query memories
python -m memoryforge.cli query "async patterns"

# Export to JSON
python -m memoryforge.cli export json --output memories.json
```

## Features

### Memory Management
- **Auto-tagging**: Automatic extraction of keywords, entities, and topics
- **Importance Scoring**: Rule-based and LLM-based importance assessment
- **Deduplication**: Detect and merge duplicate memories
- **Compression**: Multiple strategies for memory compression

### Session Management
- Create, switch, and archive sessions
- Import/export session data
- Session-scoped memory isolation

### Analytics
- Memory usage statistics
- Topic clustering
- Access pattern analysis
- Retention metrics

### Monitoring
- Health checks for all components
- Prometheus-compatible metrics
- Detailed logging with structlog

## Development

```bash
# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=memoryforge --cov-report=html

# Type checking
mypy memoryforge

# Linting
ruff check memoryforge

# Format code
ruff format memoryforge
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│              (FastAPI + WebSocket + Auth)                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Memory Manager                           │
│              (Query Router + Consolidation)                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐    ┌──────▼──────┐
│    Working    │    │    Episodic     │    │   Semantic  │
│    Memory     │    │    Memory       │    │   Memory    │
│  (In-Memory)  │    │   (Qdrant)      │    │  (Neo4j)    │
└───────────────┘    └─────────────────┘    └─────────────┘
```

## License

MIT
