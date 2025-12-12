# API Reference | API 参考

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

MemoryForge provides a RESTful API for memory management operations. The API is built with FastAPI and supports both synchronous and asynchronous operations.

**Base URL**: `http://localhost:8000`

**API Version**: `v1`

### Authentication

Currently, the API supports optional API key authentication.

```bash
# With API key
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/memory/list
```

### Common Response Format

All API responses follow this structure:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation successful",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": { ... }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Endpoints

#### Health Check

```http
GET /health
```

Check system health and component status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "components": {
    "memory": "ok",
    "database": "ok"
  }
}
```

---

#### Memory Operations

##### Store Memory

```http
POST /api/v1/memory/store
```

Store a new memory entry.

**Request Body:**
```json
{
  "content": "User prefers Python for backend development",
  "importance": 0.8,
  "tags": ["preference", "technology"],
  "metadata": {
    "source": "conversation",
    "context": "project planning"
  }
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| content | string | Yes | Memory content text |
| importance | float | No | Importance score (0.0-1.0), default: 0.5 |
| tags | string[] | No | Tags for categorization |
| metadata | object | No | Additional metadata |

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "User prefers Python for backend development",
    "importance": 0.8,
    "tags": ["preference", "technology"],
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

---

##### Query Memories

```http
POST /api/v1/memory/query
```

Search memories using semantic similarity.

**Request Body:**
```json
{
  "query": "programming language preferences",
  "top_k": 5,
  "min_score": 0.5,
  "tags": ["preference"],
  "layer": "working"
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Search query text |
| top_k | int | No | Maximum results to return (default: 10) |
| min_score | float | No | Minimum similarity score (0.0-1.0) |
| tags | string[] | No | Filter by tags |
| layer | string | No | Filter by memory layer (working/episodic/semantic) |

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "entry": {
          "id": "550e8400-e29b-41d4-a716-446655440000",
          "content": "User prefers Python for backend development",
          "importance": 0.8,
          "tags": ["preference", "technology"]
        },
        "score": 0.92
      }
    ],
    "total": 1,
    "query_time_ms": 45
  }
}
```

---

##### List Memories

```http
GET /api/v1/memory/list
```

List all stored memories with pagination.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 50 | Maximum entries to return |
| offset | int | 0 | Number of entries to skip |
| layer | string | - | Filter by memory layer |
| sort_by | string | created_at | Sort field |
| order | string | desc | Sort order (asc/desc) |

**Response:**
```json
{
  "success": true,
  "data": {
    "entries": [...],
    "total": 150,
    "limit": 50,
    "offset": 0
  }
}
```

---

##### Get Memory by ID

```http
GET /api/v1/memory/{id}
```

Retrieve a specific memory entry.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "User prefers Python",
    "layer": "working",
    "importance": 0.8,
    "tags": ["preference"],
    "metadata": {},
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

---

##### Delete Memory

```http
DELETE /api/v1/memory/{id}
```

Delete a memory entry.

**Response:**
```json
{
  "success": true,
  "message": "Memory deleted successfully"
}
```

---

##### Memory Statistics

```http
GET /api/v1/memory/stats
```

Get memory usage statistics.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_memories": 150,
    "by_layer": {
      "working": 50,
      "episodic": 80,
      "semantic": 20
    },
    "avg_importance": 0.65,
    "total_tokens": 45000,
    "session_count": 5
  }
}
```

---

#### Session Operations

##### Create Session

```http
POST /api/v1/sessions
```

Create a new memory session.

**Request Body:**
```json
{
  "name": "Project Alpha Planning",
  "metadata": {
    "project": "alpha",
    "phase": "planning"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "session-123",
    "name": "Project Alpha Planning",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

---

##### List Sessions

```http
GET /api/v1/sessions
```

List all sessions.

**Response:**
```json
{
  "success": true,
  "data": {
    "sessions": [
      {
        "id": "session-123",
        "name": "Project Alpha Planning",
        "memory_count": 25,
        "created_at": "2024-01-15T10:30:00Z"
      }
    ]
  }
}
```

---

##### Delete Session

```http
DELETE /api/v1/sessions/{id}
```

Delete a session and its memories.

---

#### Persistence Operations

##### Save to Storage

```http
POST /api/v1/persistence/save
```

Save current memories to persistent storage.

**Request Body:**
```json
{
  "path": "backup.db",
  "session_id": "session-123"
}
```

---

##### Load from Storage

```http
GET /api/v1/persistence/load
```

Load memories from persistent storage.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| path | string | Storage file path |
| session_id | string | Optional session filter |

---

### WebSocket API

MemoryForge supports real-time memory updates via WebSocket.

**Endpoint**: `ws://localhost:8000/ws`

**Message Types:**

```json
// Subscribe to memory events
{
  "type": "subscribe",
  "events": ["memory.created", "memory.updated", "memory.deleted"]
}

// Memory event notification
{
  "type": "event",
  "event_type": "memory.created",
  "data": {
    "id": "...",
    "content": "..."
  }
}
```

---

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 422 | Invalid request parameters |
| NOT_FOUND | 404 | Resource not found |
| UNAUTHORIZED | 401 | Authentication required |
| RATE_LIMITED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error |

---

<a name="chinese"></a>
## 中文

### 概述

MemoryForge 提供用于记忆管理操作的 RESTful API。该 API 基于 FastAPI 构建，支持同步和异步操作。

**基础 URL**: `http://localhost:8000`

**API 版本**: `v1`

### 认证

目前，API 支持可选的 API 密钥认证。

```bash
# 使用 API 密钥
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/memory/list
```

### 通用响应格式

所有 API 响应遵循以下结构：

```json
{
  "success": true,
  "data": { ... },
  "message": "操作成功",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

错误响应：

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数无效",
    "details": { ... }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### 接口列表

#### 健康检查

```http
GET /health
```

检查系统健康状态和组件状态。

**响应:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "components": {
    "memory": "ok",
    "database": "ok"
  }
}
```

---

#### 记忆操作

##### 存储记忆

```http
POST /api/v1/memory/store
```

存储新的记忆条目。

**请求体:**
```json
{
  "content": "用户偏好使用 Python 进行后端开发",
  "importance": 0.8,
  "tags": ["偏好", "技术"],
  "metadata": {
    "source": "对话",
    "context": "项目规划"
  }
}
```

**参数:**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| content | string | 是 | 记忆内容文本 |
| importance | float | 否 | 重要性评分 (0.0-1.0)，默认: 0.5 |
| tags | string[] | 否 | 分类标签 |
| metadata | object | 否 | 额外元数据 |

**响应:**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "用户偏好使用 Python 进行后端开发",
    "importance": 0.8,
    "tags": ["偏好", "技术"],
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

---

##### 查询记忆

```http
POST /api/v1/memory/query
```

使用语义相似度搜索记忆。

**请求体:**
```json
{
  "query": "编程语言偏好",
  "top_k": 5,
  "min_score": 0.5,
  "tags": ["偏好"],
  "layer": "working"
}
```

**参数:**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| query | string | 是 | 搜索查询文本 |
| top_k | int | 否 | 返回的最大结果数（默认: 10） |
| min_score | float | 否 | 最小相似度分数 (0.0-1.0) |
| tags | string[] | 否 | 按标签过滤 |
| layer | string | 否 | 按记忆层过滤 (working/episodic/semantic) |

**响应:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "entry": {
          "id": "550e8400-e29b-41d4-a716-446655440000",
          "content": "用户偏好使用 Python 进行后端开发",
          "importance": 0.8,
          "tags": ["偏好", "技术"]
        },
        "score": 0.92
      }
    ],
    "total": 1,
    "query_time_ms": 45
  }
}
```

---

##### 列出记忆

```http
GET /api/v1/memory/list
```

列出所有存储的记忆，支持分页。

**查询参数:**
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| limit | int | 50 | 返回的最大条目数 |
| offset | int | 0 | 跳过的条目数 |
| layer | string | - | 按记忆层过滤 |
| sort_by | string | created_at | 排序字段 |
| order | string | desc | 排序顺序 (asc/desc) |

---

##### 获取单条记忆

```http
GET /api/v1/memory/{id}
```

获取特定的记忆条目。

---

##### 删除记忆

```http
DELETE /api/v1/memory/{id}
```

删除记忆条目。

---

##### 记忆统计

```http
GET /api/v1/memory/stats
```

获取记忆使用统计信息。

**响应:**
```json
{
  "success": true,
  "data": {
    "total_memories": 150,
    "by_layer": {
      "working": 50,
      "episodic": 80,
      "semantic": 20
    },
    "avg_importance": 0.65,
    "total_tokens": 45000,
    "session_count": 5
  }
}
```

---

#### 会话操作

##### 创建会话

```http
POST /api/v1/sessions
```

创建新的记忆会话。

**请求体:**
```json
{
  "name": "项目 Alpha 规划",
  "metadata": {
    "project": "alpha",
    "phase": "planning"
  }
}
```

---

##### 列出会话

```http
GET /api/v1/sessions
```

列出所有会话。

---

##### 删除会话

```http
DELETE /api/v1/sessions/{id}
```

删除会话及其记忆。

---

#### 持久化操作

##### 保存到存储

```http
POST /api/v1/persistence/save
```

将当前记忆保存到持久化存储。

---

##### 从存储加载

```http
GET /api/v1/persistence/load
```

从持久化存储加载记忆。

---

### WebSocket API

MemoryForge 支持通过 WebSocket 实时更新记忆。

**端点**: `ws://localhost:8000/ws`

**消息类型:**

```json
// 订阅记忆事件
{
  "type": "subscribe",
  "events": ["memory.created", "memory.updated", "memory.deleted"]
}

// 记忆事件通知
{
  "type": "event",
  "event_type": "memory.created",
  "data": {
    "id": "...",
    "content": "..."
  }
}
```

---

### 错误代码

| 代码 | HTTP 状态 | 描述 |
|------|-----------|------|
| VALIDATION_ERROR | 422 | 请求参数无效 |
| NOT_FOUND | 404 | 资源未找到 |
| UNAUTHORIZED | 401 | 需要认证 |
| RATE_LIMITED | 429 | 请求过于频繁 |
| INTERNAL_ERROR | 500 | 服务器错误 |
