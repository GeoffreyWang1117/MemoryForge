# MemoryForge

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Hierarchical Context Memory System for Multi-Agent LLM Collaboration

MemoryForge is a sophisticated memory management system designed for large language model (LLM) applications. It solves the fundamental challenge of context window limitations through a three-layer hierarchical memory architecture.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-473%20passed-brightgreen.svg)]()

### The Problem

LLM context windows are limited. During long multi-agent collaboration sessions:
- Early context is lost due to truncation
- Simple sliding windows cause important information loss
- Full retrieval introduces noise and increases token costs
- No intelligent memory consolidation between sessions

### The Solution

MemoryForge implements a three-layer memory architecture inspired by human cognitive systems:

| Layer | Purpose | Storage | Access Pattern |
|-------|---------|---------|----------------|
| **Working Memory** | Current task context | In-memory | Sliding window + importance scoring |
| **Episodic Memory** | Session history | Qdrant (Vector DB) | LLM summaries + semantic retrieval |
| **Semantic Memory** | Project knowledge | Neo4j (Graph DB) | Code structure + relationships |

### Key Features

- **🧠 Intelligent Memory Management**: Automatic importance scoring, deduplication, and compression
- **🔍 Semantic Search**: Vector-based retrieval with hybrid keyword boosting
- **📊 Analytics & Insights**: Memory usage statistics, topic clustering, access patterns
- **🔌 REST API & WebSocket**: Real-time memory updates and streaming
- **💾 Multiple Storage Backends**: SQLite, Qdrant, Neo4j support
- **🎯 Event Hooks**: Extensible hook system for memory lifecycle events
- **📦 Backup & Restore**: Compressed backups with verification
- **🖥️ Rich CLI**: Interactive command-line interface

### Installation

```bash
# Clone the repository
git clone https://github.com/GeoffreyWang1117/MemoryForge.git
cd MemoryForge

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

### Quick Start

```python
import asyncio
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore, MemoryQuery

async def main():
    # Create working memory
    wm = WorkingMemory(max_entries=100, max_tokens=8000)

    # Store a memory
    entry = MemoryEntry(
        content="User wants to build a REST API with FastAPI",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.8),
        tags=["requirement", "api"],
    )
    await wm.store(entry)

    # Query memories
    query = MemoryQuery(query_text="API", top_k=5)
    result = await wm.retrieve(query)

    for entry in result.entries:
        print(f"[{entry.importance.effective_score:.2f}] {entry.content}")

asyncio.run(main())
```

### Starting the Services

```bash
# Start Qdrant and Neo4j with Docker
docker-compose up -d

# Verify services
curl http://localhost:6333/health  # Qdrant
curl http://localhost:7474         # Neo4j Browser

# Start the API server
uvicorn memoryforge.api.app:app --reload --port 8000
```

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/memory/store` | Store a memory |
| POST | `/api/v1/memory/query` | Query memories |
| GET | `/api/v1/memory/list` | List all memories |
| DELETE | `/api/v1/memory/{id}` | Delete a memory |
| GET | `/api/v1/memory/stats` | Memory statistics |
| POST | `/api/v1/sessions` | Create a session |
| GET | `/api/v1/sessions` | List sessions |

### CLI Commands

```bash
# Run the CLI
memoryforge help

# Analyze a codebase
memoryforge analyze ./src --docs

# Memory operations
memoryforge memory store "Important decision: Use PostgreSQL"
memoryforge memory query "database"
memoryforge memory list

# Export memories
memoryforge export json memories.json
memoryforge export markdown memories.md

# Session management
memoryforge session list
memoryforge session create "Project Alpha"

# View analytics
memoryforge analytics
```

### Project Structure

```
memoryforge/
├── core/           # Base types, interfaces, and exceptions
├── memory/         # Memory layer implementations
│   ├── working/    # Sliding window + importance scoring
│   ├── episodic/   # LLM summaries + vector search
│   └── semantic/   # Knowledge graph
├── retrieval/      # Semantic search and caching
├── storage/        # SQLite, Qdrant, Neo4j backends
├── context/        # LLM context building
├── hooks/          # Event system
├── backup/         # Backup and restore
├── api/            # FastAPI REST endpoints
├── cli.py          # Command-line interface
└── config.py       # Configuration management
```

### Configuration

All settings can be configured via environment variables:

```bash
# LLM Provider
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-...
LLM_ANTHROPIC_API_KEY=sk-ant-...

# Memory Settings
MEMORY_WORKING_MAX_ENTRIES=100
MEMORY_WORKING_MAX_TOKENS=8000
MEMORY_WORKING_IMPORTANCE_THRESHOLD=0.5

# Retrieval Settings
RETRIEVAL_MIN_SIMILARITY=0.5
RETRIEVAL_SEMANTIC_WEIGHT=0.7

# Storage
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### Development

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=memoryforge --cov-report=html

# Type checking
mypy memoryforge

# Linting and formatting
ruff check memoryforge
ruff format memoryforge
```

### Architecture

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

### License

MIT License - see [LICENSE](LICENSE) for details.

---

<a name="chinese"></a>
## 中文

### 多智能体LLM协作的分层上下文记忆系统

MemoryForge 是一个专为大语言模型（LLM）应用设计的高级记忆管理系统。它通过三层分层记忆架构解决了上下文窗口限制的根本性挑战。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-473%20passed-brightgreen.svg)]()

### 问题背景

LLM 的上下文窗口是有限的。在长时间的多智能体协作会话中：
- 早期上下文因截断而丢失
- 简单的滑动窗口会导致重要信息丢失
- 完整检索会引入噪声并增加 token 成本
- 会话之间缺乏智能的记忆整合机制

### 解决方案

MemoryForge 实现了一个受人类认知系统启发的三层记忆架构：

| 层级 | 用途 | 存储方式 | 访问模式 |
|------|------|----------|----------|
| **工作记忆** | 当前任务上下文 | 内存 | 滑动窗口 + 重要性评分 |
| **情景记忆** | 会话历史 | Qdrant（向量数据库） | LLM摘要 + 语义检索 |
| **语义记忆** | 项目知识 | Neo4j（图数据库） | 代码结构 + 关系图谱 |

### 核心特性

- **🧠 智能记忆管理**：自动重要性评分、去重和压缩
- **🔍 语义搜索**：基于向量的检索，支持混合关键词增强
- **📊 分析与洞察**：记忆使用统计、主题聚类、访问模式分析
- **🔌 REST API 与 WebSocket**：实时记忆更新和流式传输
- **💾 多存储后端支持**：SQLite、Qdrant、Neo4j
- **🎯 事件钩子**：可扩展的记忆生命周期事件钩子系统
- **📦 备份与恢复**：压缩备份，支持完整性验证
- **🖥️ 丰富的命令行界面**：交互式 CLI 工具

### 安装

```bash
# 克隆仓库
git clone https://github.com/GeoffreyWang1117/MemoryForge.git
cd MemoryForge

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或者: .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e ".[dev]"

# 复制环境配置模板
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

### 快速开始

```python
import asyncio
from memoryforge.memory.working.memory import WorkingMemory
from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore, MemoryQuery

async def main():
    # 创建工作记忆
    wm = WorkingMemory(max_entries=100, max_tokens=8000)

    # 存储记忆
    entry = MemoryEntry(
        content="用户想要使用 FastAPI 构建 REST API",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.8),
        tags=["需求", "api"],
    )
    await wm.store(entry)

    # 查询记忆
    query = MemoryQuery(query_text="API", top_k=5)
    result = await wm.retrieve(query)

    for entry in result.entries:
        print(f"[{entry.importance.effective_score:.2f}] {entry.content}")

asyncio.run(main())
```

### 启动服务

```bash
# 使用 Docker 启动 Qdrant 和 Neo4j
docker-compose up -d

# 验证服务
curl http://localhost:6333/health  # Qdrant
curl http://localhost:7474         # Neo4j 浏览器

# 启动 API 服务器
uvicorn memoryforge.api.app:app --reload --port 8000
```

### REST API 接口

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/memory/store` | 存储记忆 |
| POST | `/api/v1/memory/query` | 查询记忆 |
| GET | `/api/v1/memory/list` | 列出所有记忆 |
| DELETE | `/api/v1/memory/{id}` | 删除记忆 |
| GET | `/api/v1/memory/stats` | 记忆统计 |
| POST | `/api/v1/sessions` | 创建会话 |
| GET | `/api/v1/sessions` | 列出会话 |

### 命令行工具

```bash
# 运行 CLI
memoryforge help

# 分析代码库
memoryforge analyze ./src --docs

# 记忆操作
memoryforge memory store "重要决定：使用 PostgreSQL"
memoryforge memory query "数据库"
memoryforge memory list

# 导出记忆
memoryforge export json memories.json
memoryforge export markdown memories.md

# 会话管理
memoryforge session list
memoryforge session create "项目 Alpha"

# 查看分析
memoryforge analytics
```

### 项目结构

```
memoryforge/
├── core/           # 基础类型、接口和异常
├── memory/         # 记忆层实现
│   ├── working/    # 滑动窗口 + 重要性评分
│   ├── episodic/   # LLM摘要 + 向量搜索
│   └── semantic/   # 知识图谱
├── retrieval/      # 语义搜索和缓存
├── storage/        # SQLite、Qdrant、Neo4j 后端
├── context/        # LLM 上下文构建
├── hooks/          # 事件系统
├── backup/         # 备份和恢复
├── api/            # FastAPI REST 端点
├── cli.py          # 命令行界面
└── config.py       # 配置管理
```

### 配置说明

所有设置都可以通过环境变量配置：

```bash
# LLM 提供商
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-...
LLM_ANTHROPIC_API_KEY=sk-ant-...

# 记忆设置
MEMORY_WORKING_MAX_ENTRIES=100
MEMORY_WORKING_MAX_TOKENS=8000
MEMORY_WORKING_IMPORTANCE_THRESHOLD=0.5

# 检索设置
RETRIEVAL_MIN_SIMILARITY=0.5
RETRIEVAL_SEMANTIC_WEIGHT=0.7

# 存储
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### 开发指南

```bash
# 运行所有测试
pytest tests/

# 运行测试并生成覆盖率报告
pytest tests/ --cov=memoryforge --cov-report=html

# 类型检查
mypy memoryforge

# 代码检查和格式化
ruff check memoryforge
ruff format memoryforge
```

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        API 层                               │
│              (FastAPI + WebSocket + 认证)                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      记忆管理器                              │
│                (查询路由 + 记忆整合)                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐    ┌──────▼──────┐
│   工作记忆     │    │    情景记忆     │    │   语义记忆   │
│   (内存)      │    │   (Qdrant)     │    │   (Neo4j)   │
└───────────────┘    └─────────────────┘    └─────────────┘
```

### 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## Documentation | 文档

- [API Reference | API 参考](docs/api.md)
- [Architecture Guide | 架构指南](docs/architecture.md)
- [Configuration Guide | 配置指南](docs/configuration.md)
- [Quick Start Tutorial | 快速入门教程](docs/quickstart.md)

## Contributing | 贡献

We welcome contributions! Please see our contributing guidelines.

欢迎贡献代码！请查看我们的贡献指南。

## Support | 支持

- GitHub Issues: [Report a bug | 报告问题](https://github.com/GeoffreyWang1117/MemoryForge/issues)
- Documentation: [Online Docs | 在线文档](https://geoffreywang1117.github.io/MemoryForge/)
