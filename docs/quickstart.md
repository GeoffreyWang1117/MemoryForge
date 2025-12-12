# Quick Start Guide | 快速入门指南

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Docker (optional, for Qdrant and Neo4j)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/GeoffreyWang1117/MemoryForge.git
cd MemoryForge
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or just core dependencies
pip install -e .
```

#### 4. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
nano .env  # or your preferred editor
```

### Basic Usage

#### Working Memory

```python
import asyncio
from memoryforge.memory.working import WorkingMemory
from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore, MemoryQuery

async def main():
    # Create working memory instance
    wm = WorkingMemory(max_entries=100, max_tokens=8000)

    # Store a memory
    entry = MemoryEntry(
        content="The user wants to build a REST API using FastAPI",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.8),
        tags=["requirement", "api", "fastapi"],
    )
    await wm.store(entry)

    # Store another memory
    entry2 = MemoryEntry(
        content="The project should use PostgreSQL for database",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.7),
        tags=["requirement", "database"],
    )
    await wm.store(entry2)

    # Query memories
    query = MemoryQuery(query_text="API", top_k=5)
    result = await wm.retrieve(query)

    print("Found memories:")
    for entry in result.entries:
        print(f"  [{entry.importance.effective_score:.2f}] {entry.content}")

    # Get statistics
    stats = wm.get_stats()
    print(f"\nTotal memories: {stats['total_entries']}")
    print(f"Total tokens: {stats['total_tokens']}")

asyncio.run(main())
```

#### Using Tags

```python
import asyncio
from memoryforge.memory.working import WorkingMemory
from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore, MemoryQuery

async def main():
    wm = WorkingMemory()

    # Store memories with different tags
    memories = [
        ("User prefers dark mode", ["preference", "ui"]),
        ("Authentication should use JWT", ["security", "auth"]),
        ("API rate limit: 100 req/min", ["security", "api"]),
        ("Use React for frontend", ["preference", "frontend"]),
    ]

    for content, tags in memories:
        entry = MemoryEntry(
            content=content,
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.6),
            tags=tags,
        )
        await wm.store(entry)

    # Query by tags
    query = MemoryQuery(query_text="", top_k=10, tags=["security"])
    result = await wm.retrieve(query)

    print("Security-related memories:")
    for entry in result.entries:
        print(f"  - {entry.content} (tags: {entry.tags})")

asyncio.run(main())
```

### Starting Services

#### Option 1: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

This starts:
- **Qdrant** on port 6333 (vector database)
- **Neo4j** on port 7474/7687 (graph database)

#### Option 2: Manual Setup

**Qdrant:**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Neo4j:**
```bash
docker run -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:5
```

### Running the API Server

```bash
# Start the FastAPI server
uvicorn memoryforge.api.app:app --reload --host 0.0.0.0 --port 8000

# The API will be available at:
# - http://localhost:8000 (API)
# - http://localhost:8000/docs (Swagger UI)
# - http://localhost:8000/redoc (ReDoc)
```

### API Examples

#### Store a Memory

```bash
curl -X POST http://localhost:8000/api/v1/memory/store \
    -H "Content-Type: application/json" \
    -d '{
        "content": "User prefers Python for backend development",
        "importance": 0.8,
        "tags": ["preference", "technology"]
    }'
```

#### Query Memories

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
    -H "Content-Type: application/json" \
    -d '{
        "query": "programming language preferences",
        "top_k": 5
    }'
```

#### List All Memories

```bash
curl http://localhost:8000/api/v1/memory/list
```

#### Get Statistics

```bash
curl http://localhost:8000/api/v1/memory/stats
```

### CLI Usage

```bash
# Show help
memoryforge help

# Store a memory
memoryforge memory store "Important decision: Use PostgreSQL for database"

# Query memories
memoryforge memory query "database"

# List all memories
memoryforge memory list

# Export to JSON
memoryforge export json memories.json

# Session management
memoryforge session create "Project Alpha"
memoryforge session list

# View analytics
memoryforge analytics
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_working_memory.py

# Run with coverage
pytest tests/ --cov=memoryforge --cov-report=html
```

### Next Steps

1. Read the [API Reference](api.md) for complete endpoint documentation
2. Check the [Architecture Guide](architecture.md) to understand the system design
3. Review the [Configuration Guide](configuration.md) for all settings

---

<a name="chinese"></a>
## 中文

### 前置要求

- Python 3.10 或更高版本
- pip（Python 包管理器）
- Docker（可选，用于 Qdrant 和 Neo4j）

### 安装

#### 1. 克隆仓库

```bash
git clone https://github.com/GeoffreyWang1117/MemoryForge.git
cd MemoryForge
```

#### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活 (Linux/Mac)
source .venv/bin/activate

# 激活 (Windows)
.venv\Scripts\activate
```

#### 3. 安装依赖

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 或只安装核心依赖
pip install -e .
```

#### 4. 配置环境

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置文件
nano .env  # 或使用你喜欢的编辑器
```

### 基本用法

#### 工作记忆

```python
import asyncio
from memoryforge.memory.working import WorkingMemory
from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore, MemoryQuery

async def main():
    # 创建工作记忆实例
    wm = WorkingMemory(max_entries=100, max_tokens=8000)

    # 存储记忆
    entry = MemoryEntry(
        content="用户想要使用 FastAPI 构建 REST API",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.8),
        tags=["需求", "api", "fastapi"],
    )
    await wm.store(entry)

    # 存储另一个记忆
    entry2 = MemoryEntry(
        content="项目应使用 PostgreSQL 作为数据库",
        layer=MemoryLayer.WORKING,
        importance=ImportanceScore(base_score=0.7),
        tags=["需求", "数据库"],
    )
    await wm.store(entry2)

    # 查询记忆
    query = MemoryQuery(query_text="API", top_k=5)
    result = await wm.retrieve(query)

    print("找到的记忆：")
    for entry in result.entries:
        print(f"  [{entry.importance.effective_score:.2f}] {entry.content}")

    # 获取统计信息
    stats = wm.get_stats()
    print(f"\n总记忆数: {stats['total_entries']}")
    print(f"总令牌数: {stats['total_tokens']}")

asyncio.run(main())
```

#### 使用标签

```python
import asyncio
from memoryforge.memory.working import WorkingMemory
from memoryforge.core.types import MemoryEntry, MemoryLayer, ImportanceScore, MemoryQuery

async def main():
    wm = WorkingMemory()

    # 使用不同标签存储记忆
    memories = [
        ("用户偏好深色模式", ["偏好", "界面"]),
        ("认证应使用 JWT", ["安全", "认证"]),
        ("API 速率限制: 100 请求/分钟", ["安全", "api"]),
        ("前端使用 React", ["偏好", "前端"]),
    ]

    for content, tags in memories:
        entry = MemoryEntry(
            content=content,
            layer=MemoryLayer.WORKING,
            importance=ImportanceScore(base_score=0.6),
            tags=tags,
        )
        await wm.store(entry)

    # 按标签查询
    query = MemoryQuery(query_text="", top_k=10, tags=["安全"])
    result = await wm.retrieve(query)

    print("安全相关的记忆：")
    for entry in result.entries:
        print(f"  - {entry.content} (标签: {entry.tags})")

asyncio.run(main())
```

### 启动服务

#### 方式一：Docker Compose（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 检查状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

这将启动：
- **Qdrant** 端口 6333（向量数据库）
- **Neo4j** 端口 7474/7687（图数据库）

#### 方式二：手动设置

**Qdrant：**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Neo4j：**
```bash
docker run -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:5
```

### 运行 API 服务器

```bash
# 启动 FastAPI 服务器
uvicorn memoryforge.api.app:app --reload --host 0.0.0.0 --port 8000

# API 将在以下地址可用：
# - http://localhost:8000（API）
# - http://localhost:8000/docs（Swagger UI）
# - http://localhost:8000/redoc（ReDoc）
```

### API 示例

#### 存储记忆

```bash
curl -X POST http://localhost:8000/api/v1/memory/store \
    -H "Content-Type: application/json" \
    -d '{
        "content": "用户偏好使用 Python 进行后端开发",
        "importance": 0.8,
        "tags": ["偏好", "技术"]
    }'
```

#### 查询记忆

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
    -H "Content-Type: application/json" \
    -d '{
        "query": "编程语言偏好",
        "top_k": 5
    }'
```

#### 列出所有记忆

```bash
curl http://localhost:8000/api/v1/memory/list
```

#### 获取统计信息

```bash
curl http://localhost:8000/api/v1/memory/stats
```

### CLI 使用方法

```bash
# 显示帮助
memoryforge help

# 存储记忆
memoryforge memory store "重要决定：使用 PostgreSQL 作为数据库"

# 查询记忆
memoryforge memory query "数据库"

# 列出所有记忆
memoryforge memory list

# 导出到 JSON
memoryforge export json memories.json

# 会话管理
memoryforge session create "项目 Alpha"
memoryforge session list

# 查看分析
memoryforge analytics
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 详细输出
pytest tests/ -v

# 运行特定测试文件
pytest tests/unit/test_working_memory.py

# 生成覆盖率报告
pytest tests/ --cov=memoryforge --cov-report=html
```

### 后续步骤

1. 阅读 [API 参考](api.md) 获取完整的端点文档
2. 查看 [架构指南](architecture.md) 了解系统设计
3. 参阅 [配置指南](configuration.md) 了解所有设置选项
