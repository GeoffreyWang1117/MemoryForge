# Architecture Guide | 架构指南

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

MemoryForge implements a three-layer hierarchical memory architecture inspired by human cognitive systems. This design enables efficient context management for LLM applications by organizing memories based on their temporal relevance and semantic importance.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│                  (FastAPI + WebSocket + Auth)                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Memory Manager                             │
│                 (Query Router + Consolidation)                  │
└─────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Working Memory │  │ Episodic Memory │  │ Semantic Memory │
│   (In-Memory)   │  │    (Qdrant)     │  │    (Neo4j)      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│              (SQLite + Qdrant + Neo4j + Backup)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Layers

#### Working Memory

Working memory handles the current task context and recent interactions.

**Characteristics:**
- **Storage**: In-memory (fast access)
- **Capacity**: Limited (configurable max entries and tokens)
- **Lifetime**: Current session
- **Access Pattern**: Sliding window with importance scoring

**Key Features:**
- Automatic importance decay over time
- Priority-based eviction when capacity is reached
- Fast retrieval for active context

**Implementation:**
```python
from memoryforge.memory.working import WorkingMemory

wm = WorkingMemory(
    max_entries=100,
    max_tokens=8000,
    importance_threshold=0.3
)
```

#### Episodic Memory

Episodic memory stores session history and past interactions.

**Characteristics:**
- **Storage**: Qdrant (vector database)
- **Capacity**: Large (disk-based)
- **Lifetime**: Persistent across sessions
- **Access Pattern**: LLM summaries + semantic retrieval

**Key Features:**
- Vector embeddings for semantic search
- LLM-generated summaries for compression
- Hybrid search (semantic + keyword)

**Implementation:**
```python
from memoryforge.memory.episodic import EpisodicMemory

em = EpisodicMemory(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="memories"
)
```

#### Semantic Memory

Semantic memory maintains project knowledge and code structure.

**Characteristics:**
- **Storage**: Neo4j (graph database)
- **Capacity**: Very large
- **Lifetime**: Permanent project knowledge
- **Access Pattern**: Graph traversal + relationship queries

**Key Features:**
- Code entity relationships
- Concept hierarchies
- Cross-reference navigation

### Data Flow

```
User Input
    │
    ▼
┌───────────────┐
│ Context Build │◄────── Working Memory (recent context)
└───────────────┘◄────── Episodic Memory (relevant history)
    │           ◄────── Semantic Memory (knowledge graph)
    ▼
┌───────────────┐
│  LLM Request  │
└───────────────┘
    │
    ▼
┌───────────────┐
│ LLM Response  │
└───────────────┘
    │
    ▼
┌───────────────┐
│ Memory Store  │────► Working Memory (immediate storage)
└───────────────┘      │
                       ▼ (consolidation triggers)
                 ┌─────────────┐
                 │ Episodic    │ (when working memory is full)
                 └─────────────┘
                       │
                       ▼ (knowledge extraction)
                 ┌─────────────┐
                 │ Semantic    │ (structured knowledge)
                 └─────────────┘
```

### Core Components

#### MemoryEntry

The fundamental unit of storage:

```python
@dataclass
class MemoryEntry:
    id: UUID
    content: str
    layer: MemoryLayer
    importance: ImportanceScore
    tags: list[str]
    metadata: dict
    embedding: list[float] | None
    created_at: datetime
    updated_at: datetime
```

#### ImportanceScore

Adaptive importance scoring:

```python
@dataclass
class ImportanceScore:
    base_score: float      # Initial importance (0.0-1.0)
    recency_boost: float   # Time-based boost
    access_boost: float    # Frequency-based boost

    @property
    def effective_score(self) -> float:
        return min(1.0, self.base_score + self.recency_boost + self.access_boost)
```

#### MemoryQuery

Flexible query interface:

```python
@dataclass
class MemoryQuery:
    query_text: str
    top_k: int = 10
    min_score: float = 0.0
    tags: list[str] | None = None
    layer: MemoryLayer | None = None
    time_range: tuple[datetime, datetime] | None = None
```

### Event System

MemoryForge uses an event-driven architecture for extensibility:

```python
class EventType(Enum):
    MEMORY_CREATED = "memory.created"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_DELETED = "memory.deleted"
    MEMORY_ACCESSED = "memory.accessed"
    CONSOLIDATION_STARTED = "consolidation.started"
    CONSOLIDATION_COMPLETED = "consolidation.completed"
```

**Hook Registration:**
```python
from memoryforge.hooks import HookRegistry, EventType

registry = HookRegistry()

@registry.on(EventType.MEMORY_CREATED)
async def on_memory_created(event):
    print(f"New memory: {event.data['content']}")
```

### Consolidation Process

Memory consolidation moves data between layers:

```
Working Memory (full)
        │
        ▼ (importance scoring)
┌───────────────────┐
│ Select Important  │
│    Memories       │
└───────────────────┘
        │
        ▼ (LLM summarization)
┌───────────────────┐
│ Generate Summary  │
└───────────────────┘
        │
        ▼ (vector embedding)
┌───────────────────┐
│ Create Embedding  │
└───────────────────┘
        │
        ▼
Episodic Memory
```

### Storage Backends

| Backend | Purpose | Technology |
|---------|---------|------------|
| SQLite | Local persistence | sqlite3 |
| Qdrant | Vector search | Qdrant DB |
| Neo4j | Graph relationships | Neo4j |

### API Design

RESTful endpoints follow these conventions:

- `GET /api/v1/resource` - List resources
- `POST /api/v1/resource` - Create resource
- `GET /api/v1/resource/{id}` - Get single resource
- `PUT /api/v1/resource/{id}` - Update resource
- `DELETE /api/v1/resource/{id}` - Delete resource

WebSocket for real-time updates:
- `ws://localhost:8000/ws` - Memory event stream

### Security

- API key authentication (optional)
- Rate limiting per endpoint
- Input validation with Pydantic
- SQL injection prevention

### Performance Considerations

1. **Caching**: LRU cache for frequently accessed memories
2. **Batch Operations**: Bulk inserts and updates
3. **Lazy Loading**: Embeddings generated on demand
4. **Index Optimization**: Proper indexes on SQLite tables

---

<a name="chinese"></a>
## 中文

### 概述

MemoryForge 实现了一个受人类认知系统启发的三层分层记忆架构。该设计通过根据时间相关性和语义重要性组织记忆，为 LLM 应用提供高效的上下文管理。

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          API 层                                  │
│                  (FastAPI + WebSocket + 认证)                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        记忆管理器                                │
│                   (查询路由 + 记忆整合)                          │
└─────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    工作记忆     │  │    情景记忆     │  │    语义记忆     │
│    (内存)      │  │   (Qdrant)     │  │    (Neo4j)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         存储层                                   │
│              (SQLite + Qdrant + Neo4j + 备份)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 记忆层级

#### 工作记忆

工作记忆处理当前任务上下文和最近的交互。

**特性：**
- **存储**：内存（快速访问）
- **容量**：有限（可配置的最大条目数和令牌数）
- **生命周期**：当前会话
- **访问模式**：滑动窗口 + 重要性评分

**核心功能：**
- 随时间自动衰减重要性
- 达到容量时基于优先级的驱逐
- 活跃上下文的快速检索

**实现示例：**
```python
from memoryforge.memory.working import WorkingMemory

wm = WorkingMemory(
    max_entries=100,
    max_tokens=8000,
    importance_threshold=0.3
)
```

#### 情景记忆

情景记忆存储会话历史和过去的交互。

**特性：**
- **存储**：Qdrant（向量数据库）
- **容量**：大（基于磁盘）
- **生命周期**：跨会话持久化
- **访问模式**：LLM 摘要 + 语义检索

**核心功能：**
- 用于语义搜索的向量嵌入
- LLM 生成的摘要用于压缩
- 混合搜索（语义 + 关键词）

**实现示例：**
```python
from memoryforge.memory.episodic import EpisodicMemory

em = EpisodicMemory(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="memories"
)
```

#### 语义记忆

语义记忆维护项目知识和代码结构。

**特性：**
- **存储**：Neo4j（图数据库）
- **容量**：非常大
- **生命周期**：永久项目知识
- **访问模式**：图遍历 + 关系查询

**核心功能：**
- 代码实体关系
- 概念层次结构
- 交叉引用导航

### 数据流

```
用户输入
    │
    ▼
┌───────────────┐
│  上下文构建   │◄────── 工作记忆（最近上下文）
└───────────────┘◄────── 情景记忆（相关历史）
    │           ◄────── 语义记忆（知识图谱）
    ▼
┌───────────────┐
│   LLM 请求    │
└───────────────┘
    │
    ▼
┌───────────────┐
│   LLM 响应    │
└───────────────┘
    │
    ▼
┌───────────────┐
│   记忆存储    │────► 工作记忆（立即存储）
└───────────────┘      │
                       ▼ （整合触发）
                 ┌─────────────┐
                 │  情景记忆   │ （当工作记忆已满时）
                 └─────────────┘
                       │
                       ▼ （知识提取）
                 ┌─────────────┐
                 │  语义记忆   │ （结构化知识）
                 └─────────────┘
```

### 核心组件

#### MemoryEntry（记忆条目）

存储的基本单元：

```python
@dataclass
class MemoryEntry:
    id: UUID              # 唯一标识符
    content: str          # 记忆内容
    layer: MemoryLayer    # 记忆层级
    importance: ImportanceScore  # 重要性评分
    tags: list[str]       # 标签
    metadata: dict        # 元数据
    embedding: list[float] | None  # 向量嵌入
    created_at: datetime  # 创建时间
    updated_at: datetime  # 更新时间
```

#### ImportanceScore（重要性评分）

自适应重要性评分：

```python
@dataclass
class ImportanceScore:
    base_score: float      # 初始重要性 (0.0-1.0)
    recency_boost: float   # 基于时间的加成
    access_boost: float    # 基于频率的加成

    @property
    def effective_score(self) -> float:
        return min(1.0, self.base_score + self.recency_boost + self.access_boost)
```

#### MemoryQuery（记忆查询）

灵活的查询接口：

```python
@dataclass
class MemoryQuery:
    query_text: str                    # 查询文本
    top_k: int = 10                    # 返回结果数
    min_score: float = 0.0             # 最小分数
    tags: list[str] | None = None      # 标签过滤
    layer: MemoryLayer | None = None   # 层级过滤
    time_range: tuple[datetime, datetime] | None = None  # 时间范围
```

### 事件系统

MemoryForge 使用事件驱动架构实现可扩展性：

```python
class EventType(Enum):
    MEMORY_CREATED = "memory.created"       # 记忆创建
    MEMORY_UPDATED = "memory.updated"       # 记忆更新
    MEMORY_DELETED = "memory.deleted"       # 记忆删除
    MEMORY_ACCESSED = "memory.accessed"     # 记忆访问
    CONSOLIDATION_STARTED = "consolidation.started"    # 整合开始
    CONSOLIDATION_COMPLETED = "consolidation.completed" # 整合完成
```

**钩子注册：**
```python
from memoryforge.hooks import HookRegistry, EventType

registry = HookRegistry()

@registry.on(EventType.MEMORY_CREATED)
async def on_memory_created(event):
    print(f"新记忆: {event.data['content']}")
```

### 整合过程

记忆整合在层级之间移动数据：

```
工作记忆（已满）
        │
        ▼ （重要性评分）
┌───────────────────┐
│   选择重要记忆    │
└───────────────────┘
        │
        ▼ （LLM 摘要）
┌───────────────────┐
│    生成摘要      │
└───────────────────┘
        │
        ▼ （向量嵌入）
┌───────────────────┐
│   创建嵌入向量    │
└───────────────────┘
        │
        ▼
    情景记忆
```

### 存储后端

| 后端 | 用途 | 技术 |
|------|------|------|
| SQLite | 本地持久化 | sqlite3 |
| Qdrant | 向量搜索 | Qdrant DB |
| Neo4j | 图关系 | Neo4j |

### API 设计

RESTful 端点遵循以下约定：

- `GET /api/v1/resource` - 列出资源
- `POST /api/v1/resource` - 创建资源
- `GET /api/v1/resource/{id}` - 获取单个资源
- `PUT /api/v1/resource/{id}` - 更新资源
- `DELETE /api/v1/resource/{id}` - 删除资源

WebSocket 用于实时更新：
- `ws://localhost:8000/ws` - 记忆事件流

### 安全性

- API 密钥认证（可选）
- 每个端点的速率限制
- 使用 Pydantic 进行输入验证
- SQL 注入防护

### 性能考虑

1. **缓存**：频繁访问记忆的 LRU 缓存
2. **批量操作**：批量插入和更新
3. **延迟加载**：按需生成嵌入
4. **索引优化**：SQLite 表的适当索引
