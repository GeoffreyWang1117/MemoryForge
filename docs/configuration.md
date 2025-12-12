# Configuration Guide | 配置指南

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

MemoryForge uses environment variables for configuration. All settings can be defined in a `.env` file or exported as shell environment variables.

### Quick Setup

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
nano .env
```

### Configuration Categories

---

### LLM Provider Settings

Configure the LLM provider for summarization and embeddings.

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `LLM_PROVIDER` | Primary LLM provider | `openai` | `openai`, `anthropic`, `local` |
| `LLM_MODEL` | Model name | `gpt-4` | Depends on provider |
| `LLM_TEMPERATURE` | Response randomness | `0.7` | `0.0` - `2.0` |
| `LLM_MAX_TOKENS` | Max response tokens | `2000` | Provider limit |

#### OpenAI Settings

```bash
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-your-api-key
LLM_OPENAI_ORG_ID=org-optional
LLM_MODEL=gpt-4
```

#### Anthropic Settings

```bash
LLM_PROVIDER=anthropic
LLM_ANTHROPIC_API_KEY=sk-ant-your-api-key
LLM_MODEL=claude-3-sonnet-20240229
```

#### Local LLM Settings

```bash
LLM_PROVIDER=local
LLM_LOCAL_URL=http://localhost:11434
LLM_MODEL=llama2
```

---

### Memory Settings

Configure memory layer behavior.

#### Working Memory

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_WORKING_MAX_ENTRIES` | Maximum number of entries | `100` |
| `MEMORY_WORKING_MAX_TOKENS` | Maximum total tokens | `8000` |
| `MEMORY_WORKING_IMPORTANCE_THRESHOLD` | Minimum importance to keep | `0.3` |
| `MEMORY_WORKING_DECAY_RATE` | Importance decay rate per hour | `0.1` |

```bash
MEMORY_WORKING_MAX_ENTRIES=100
MEMORY_WORKING_MAX_TOKENS=8000
MEMORY_WORKING_IMPORTANCE_THRESHOLD=0.3
MEMORY_WORKING_DECAY_RATE=0.1
```

#### Episodic Memory

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_EPISODIC_SUMMARY_THRESHOLD` | Entries before summarization | `50` |
| `MEMORY_EPISODIC_COMPRESSION_RATIO` | Target compression ratio | `0.3` |

```bash
MEMORY_EPISODIC_SUMMARY_THRESHOLD=50
MEMORY_EPISODIC_COMPRESSION_RATIO=0.3
```

#### Semantic Memory

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_SEMANTIC_MAX_DEPTH` | Maximum graph traversal depth | `3` |
| `MEMORY_SEMANTIC_MIN_RELEVANCE` | Minimum relevance score | `0.5` |

```bash
MEMORY_SEMANTIC_MAX_DEPTH=3
MEMORY_SEMANTIC_MIN_RELEVANCE=0.5
```

---

### Retrieval Settings

Configure search and retrieval behavior.

| Variable | Description | Default |
|----------|-------------|---------|
| `RETRIEVAL_DEFAULT_TOP_K` | Default results to return | `10` |
| `RETRIEVAL_MIN_SIMILARITY` | Minimum similarity score | `0.5` |
| `RETRIEVAL_SEMANTIC_WEIGHT` | Weight for semantic search | `0.7` |
| `RETRIEVAL_KEYWORD_WEIGHT` | Weight for keyword search | `0.3` |
| `RETRIEVAL_RERANK_ENABLED` | Enable result reranking | `true` |

```bash
RETRIEVAL_DEFAULT_TOP_K=10
RETRIEVAL_MIN_SIMILARITY=0.5
RETRIEVAL_SEMANTIC_WEIGHT=0.7
RETRIEVAL_KEYWORD_WEIGHT=0.3
RETRIEVAL_RERANK_ENABLED=true
```

---

### Embedding Settings

Configure embedding generation.

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | Embedding provider | `openai` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | Vector dimension | `1536` |
| `EMBEDDING_BATCH_SIZE` | Batch size for generation | `100` |

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=100
```

---

### Storage Settings

#### SQLite

| Variable | Description | Default |
|----------|-------------|---------|
| `SQLITE_PATH` | Database file path | `data/memory.db` |
| `SQLITE_TIMEOUT` | Connection timeout (seconds) | `30` |

```bash
SQLITE_PATH=data/memory.db
SQLITE_TIMEOUT=30
```

#### Qdrant

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `QDRANT_GRPC_PORT` | Qdrant gRPC port | `6334` |
| `QDRANT_API_KEY` | API key (if enabled) | - |
| `QDRANT_COLLECTION` | Collection name | `memories` |

```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=
QDRANT_COLLECTION=memories
```

#### Neo4j

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Username | `neo4j` |
| `NEO4J_PASSWORD` | Password | `password` |
| `NEO4J_DATABASE` | Database name | `neo4j` |

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

---

### API Settings

Configure the REST API server.

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Server host | `0.0.0.0` |
| `API_PORT` | Server port | `8000` |
| `API_DEBUG` | Debug mode | `false` |
| `API_RELOAD` | Auto-reload on changes | `false` |
| `API_WORKERS` | Number of workers | `1` |
| `API_CORS_ORIGINS` | Allowed CORS origins | `*` |

```bash
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_RELOAD=false
API_WORKERS=1
API_CORS_ORIGINS=*
```

#### Authentication

| Variable | Description | Default |
|----------|-------------|---------|
| `API_AUTH_ENABLED` | Enable authentication | `false` |
| `API_KEY` | API key for authentication | - |
| `API_KEY_HEADER` | Header name for API key | `X-API-Key` |

```bash
API_AUTH_ENABLED=true
API_KEY=your-secret-api-key
API_KEY_HEADER=X-API-Key
```

#### Rate Limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `API_RATE_LIMIT_ENABLED` | Enable rate limiting | `true` |
| `API_RATE_LIMIT_REQUESTS` | Requests per window | `100` |
| `API_RATE_LIMIT_WINDOW` | Window duration | `minute` |

```bash
API_RATE_LIMIT_ENABLED=true
API_RATE_LIMIT_REQUESTS=100
API_RATE_LIMIT_WINDOW=minute
```

---

### Backup Settings

Configure backup and restore behavior.

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKUP_DIR` | Backup directory | `backups/` |
| `BACKUP_RETENTION_DAYS` | Days to keep backups | `30` |
| `BACKUP_COMPRESSION` | Enable compression | `true` |
| `BACKUP_AUTO_ENABLED` | Enable automatic backups | `false` |
| `BACKUP_AUTO_INTERVAL` | Auto backup interval | `24h` |

```bash
BACKUP_DIR=backups/
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
BACKUP_AUTO_ENABLED=false
BACKUP_AUTO_INTERVAL=24h
```

---

### Logging Settings

Configure logging behavior.

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log format | `text` |
| `LOG_FILE` | Log file path | - |
| `LOG_MAX_SIZE` | Max log file size | `10MB` |
| `LOG_BACKUP_COUNT` | Number of log backups | `5` |

```bash
LOG_LEVEL=INFO
LOG_FORMAT=text
LOG_FILE=logs/memoryforge.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5
```

Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

Log formats: `text`, `json`

---

### Example Configurations

#### Development

```bash
# .env.development
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-dev-key

MEMORY_WORKING_MAX_ENTRIES=50
MEMORY_WORKING_MAX_TOKENS=4000

API_DEBUG=true
API_RELOAD=true
LOG_LEVEL=DEBUG
```

#### Production

```bash
# .env.production
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-prod-key

MEMORY_WORKING_MAX_ENTRIES=200
MEMORY_WORKING_MAX_TOKENS=16000

QDRANT_HOST=qdrant.example.com
QDRANT_API_KEY=qdrant-secret

NEO4J_URI=bolt://neo4j.example.com:7687
NEO4J_PASSWORD=strong-password

API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_AUTH_ENABLED=true
API_KEY=production-api-key

BACKUP_AUTO_ENABLED=true
BACKUP_AUTO_INTERVAL=6h

LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/memoryforge/app.log
```

#### Docker

```bash
# .env.docker
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-docker-key

QDRANT_HOST=qdrant
QDRANT_PORT=6333

NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=docker-password

SQLITE_PATH=/data/memory.db
BACKUP_DIR=/data/backups
```

---

### Programmatic Configuration

You can also configure MemoryForge programmatically:

```python
from memoryforge.config import Settings

# Create settings instance
settings = Settings(
    llm_provider="openai",
    llm_openai_api_key="sk-...",
    memory_working_max_entries=100,
    memory_working_max_tokens=8000,
    api_port=8000,
)

# Access settings
print(settings.memory_working_max_entries)  # 100
```

---

<a name="chinese"></a>
## 中文

### 概述

MemoryForge 使用环境变量进行配置。所有设置都可以在 `.env` 文件中定义，或作为 shell 环境变量导出。

### 快速设置

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置
nano .env
```

### 配置分类

---

### LLM 提供商设置

配置用于摘要和嵌入的 LLM 提供商。

| 变量 | 描述 | 默认值 | 选项 |
|------|------|--------|------|
| `LLM_PROVIDER` | 主要 LLM 提供商 | `openai` | `openai`, `anthropic`, `local` |
| `LLM_MODEL` | 模型名称 | `gpt-4` | 取决于提供商 |
| `LLM_TEMPERATURE` | 响应随机性 | `0.7` | `0.0` - `2.0` |
| `LLM_MAX_TOKENS` | 最大响应令牌数 | `2000` | 提供商限制 |

#### OpenAI 设置

```bash
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-your-api-key
LLM_OPENAI_ORG_ID=org-optional
LLM_MODEL=gpt-4
```

#### Anthropic 设置

```bash
LLM_PROVIDER=anthropic
LLM_ANTHROPIC_API_KEY=sk-ant-your-api-key
LLM_MODEL=claude-3-sonnet-20240229
```

#### 本地 LLM 设置

```bash
LLM_PROVIDER=local
LLM_LOCAL_URL=http://localhost:11434
LLM_MODEL=llama2
```

---

### 记忆设置

配置记忆层行为。

#### 工作记忆

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `MEMORY_WORKING_MAX_ENTRIES` | 最大条目数 | `100` |
| `MEMORY_WORKING_MAX_TOKENS` | 最大总令牌数 | `8000` |
| `MEMORY_WORKING_IMPORTANCE_THRESHOLD` | 保留的最低重要性 | `0.3` |
| `MEMORY_WORKING_DECAY_RATE` | 每小时重要性衰减率 | `0.1` |

```bash
MEMORY_WORKING_MAX_ENTRIES=100
MEMORY_WORKING_MAX_TOKENS=8000
MEMORY_WORKING_IMPORTANCE_THRESHOLD=0.3
MEMORY_WORKING_DECAY_RATE=0.1
```

#### 情景记忆

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `MEMORY_EPISODIC_SUMMARY_THRESHOLD` | 触发摘要的条目数 | `50` |
| `MEMORY_EPISODIC_COMPRESSION_RATIO` | 目标压缩比 | `0.3` |

```bash
MEMORY_EPISODIC_SUMMARY_THRESHOLD=50
MEMORY_EPISODIC_COMPRESSION_RATIO=0.3
```

#### 语义记忆

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `MEMORY_SEMANTIC_MAX_DEPTH` | 最大图遍历深度 | `3` |
| `MEMORY_SEMANTIC_MIN_RELEVANCE` | 最小相关性分数 | `0.5` |

```bash
MEMORY_SEMANTIC_MAX_DEPTH=3
MEMORY_SEMANTIC_MIN_RELEVANCE=0.5
```

---

### 检索设置

配置搜索和检索行为。

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `RETRIEVAL_DEFAULT_TOP_K` | 默认返回结果数 | `10` |
| `RETRIEVAL_MIN_SIMILARITY` | 最小相似度分数 | `0.5` |
| `RETRIEVAL_SEMANTIC_WEIGHT` | 语义搜索权重 | `0.7` |
| `RETRIEVAL_KEYWORD_WEIGHT` | 关键词搜索权重 | `0.3` |
| `RETRIEVAL_RERANK_ENABLED` | 启用结果重排序 | `true` |

```bash
RETRIEVAL_DEFAULT_TOP_K=10
RETRIEVAL_MIN_SIMILARITY=0.5
RETRIEVAL_SEMANTIC_WEIGHT=0.7
RETRIEVAL_KEYWORD_WEIGHT=0.3
RETRIEVAL_RERANK_ENABLED=true
```

---

### 嵌入设置

配置嵌入生成。

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `EMBEDDING_PROVIDER` | 嵌入提供商 | `openai` |
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | 向量维度 | `1536` |
| `EMBEDDING_BATCH_SIZE` | 生成批次大小 | `100` |

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=100
```

---

### 存储设置

#### SQLite

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `SQLITE_PATH` | 数据库文件路径 | `data/memory.db` |
| `SQLITE_TIMEOUT` | 连接超时（秒） | `30` |

```bash
SQLITE_PATH=data/memory.db
SQLITE_TIMEOUT=30
```

#### Qdrant

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `QDRANT_HOST` | Qdrant 服务器主机 | `localhost` |
| `QDRANT_PORT` | Qdrant 服务器端口 | `6333` |
| `QDRANT_GRPC_PORT` | Qdrant gRPC 端口 | `6334` |
| `QDRANT_API_KEY` | API 密钥（如果启用） | - |
| `QDRANT_COLLECTION` | 集合名称 | `memories` |

```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=
QDRANT_COLLECTION=memories
```

#### Neo4j

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `NEO4J_URI` | Neo4j 连接 URI | `bolt://localhost:7687` |
| `NEO4J_USER` | 用户名 | `neo4j` |
| `NEO4J_PASSWORD` | 密码 | `password` |
| `NEO4J_DATABASE` | 数据库名称 | `neo4j` |

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

---

### API 设置

配置 REST API 服务器。

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `API_HOST` | 服务器主机 | `0.0.0.0` |
| `API_PORT` | 服务器端口 | `8000` |
| `API_DEBUG` | 调试模式 | `false` |
| `API_RELOAD` | 更改时自动重载 | `false` |
| `API_WORKERS` | 工作进程数 | `1` |
| `API_CORS_ORIGINS` | 允许的 CORS 源 | `*` |

```bash
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_RELOAD=false
API_WORKERS=1
API_CORS_ORIGINS=*
```

#### 认证

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `API_AUTH_ENABLED` | 启用认证 | `false` |
| `API_KEY` | 用于认证的 API 密钥 | - |
| `API_KEY_HEADER` | API 密钥的请求头名称 | `X-API-Key` |

```bash
API_AUTH_ENABLED=true
API_KEY=your-secret-api-key
API_KEY_HEADER=X-API-Key
```

#### 速率限制

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `API_RATE_LIMIT_ENABLED` | 启用速率限制 | `true` |
| `API_RATE_LIMIT_REQUESTS` | 每个窗口的请求数 | `100` |
| `API_RATE_LIMIT_WINDOW` | 窗口持续时间 | `minute` |

```bash
API_RATE_LIMIT_ENABLED=true
API_RATE_LIMIT_REQUESTS=100
API_RATE_LIMIT_WINDOW=minute
```

---

### 备份设置

配置备份和恢复行为。

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `BACKUP_DIR` | 备份目录 | `backups/` |
| `BACKUP_RETENTION_DAYS` | 备份保留天数 | `30` |
| `BACKUP_COMPRESSION` | 启用压缩 | `true` |
| `BACKUP_AUTO_ENABLED` | 启用自动备份 | `false` |
| `BACKUP_AUTO_INTERVAL` | 自动备份间隔 | `24h` |

```bash
BACKUP_DIR=backups/
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
BACKUP_AUTO_ENABLED=false
BACKUP_AUTO_INTERVAL=24h
```

---

### 日志设置

配置日志行为。

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_FORMAT` | 日志格式 | `text` |
| `LOG_FILE` | 日志文件路径 | - |
| `LOG_MAX_SIZE` | 最大日志文件大小 | `10MB` |
| `LOG_BACKUP_COUNT` | 日志备份数量 | `5` |

```bash
LOG_LEVEL=INFO
LOG_FORMAT=text
LOG_FILE=logs/memoryforge.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5
```

日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`

日志格式：`text`、`json`

---

### 配置示例

#### 开发环境

```bash
# .env.development
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-dev-key

MEMORY_WORKING_MAX_ENTRIES=50
MEMORY_WORKING_MAX_TOKENS=4000

API_DEBUG=true
API_RELOAD=true
LOG_LEVEL=DEBUG
```

#### 生产环境

```bash
# .env.production
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-prod-key

MEMORY_WORKING_MAX_ENTRIES=200
MEMORY_WORKING_MAX_TOKENS=16000

QDRANT_HOST=qdrant.example.com
QDRANT_API_KEY=qdrant-secret

NEO4J_URI=bolt://neo4j.example.com:7687
NEO4J_PASSWORD=strong-password

API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_AUTH_ENABLED=true
API_KEY=production-api-key

BACKUP_AUTO_ENABLED=true
BACKUP_AUTO_INTERVAL=6h

LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/memoryforge/app.log
```

#### Docker 环境

```bash
# .env.docker
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-docker-key

QDRANT_HOST=qdrant
QDRANT_PORT=6333

NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=docker-password

SQLITE_PATH=/data/memory.db
BACKUP_DIR=/data/backups
```

---

### 编程方式配置

您也可以通过编程方式配置 MemoryForge：

```python
from memoryforge.config import Settings

# 创建设置实例
settings = Settings(
    llm_provider="openai",
    llm_openai_api_key="sk-...",
    memory_working_max_entries=100,
    memory_working_max_tokens=8000,
    api_port=8000,
)

# 访问设置
print(settings.memory_working_max_entries)  # 100
```
