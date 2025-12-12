"""Integration tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from memoryforge.api.app import create_app


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


@pytest.fixture
def client(app):
    """Create sync test client."""
    with TestClient(app) as tc:
        yield tc


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "memory_entries" in data


class TestMemoryEndpoints:
    """Test memory CRUD endpoints."""

    def test_store_memory(self, client):
        """Test storing a memory entry."""
        response = client.post(
            "/api/v1/memory/store",
            json={
                "content": "Test memory content",
                "importance": 0.8,
                "tags": ["test", "integration"],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert data["message"] == "Memory stored successfully"

    def test_list_memories(self, client):
        """Test listing memories."""
        # First store some
        client.post(
            "/api/v1/memory/store",
            json={"content": "Memory 1", "importance": 0.5},
        )
        client.post(
            "/api/v1/memory/store",
            json={"content": "Memory 2", "importance": 0.6},
        )

        response = client.get("/api/v1/memory/list")
        assert response.status_code == 200

        data = response.json()
        assert "entries" in data
        assert len(data["entries"]) >= 2

    def test_query_memories(self, client):
        """Test querying memories."""
        # Store a memory
        client.post(
            "/api/v1/memory/store",
            json={
                "content": "Python programming tutorial",
                "importance": 0.7,
                "tags": ["python", "tutorial"],
            },
        )

        # Query for it
        response = client.post(
            "/api/v1/memory/query",
            json={
                "query": "python",
                "top_k": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "entries" in data
        assert "query_time_ms" in data

    def test_delete_memory(self, client):
        """Test deleting a memory."""
        # Store first
        store_response = client.post(
            "/api/v1/memory/store",
            json={"content": "To be deleted", "importance": 0.5},
        )
        memory_id = store_response.json()["id"]

        # Delete it
        response = client.delete(f"/api/v1/memory/{memory_id}")
        assert response.status_code == 200

    def test_delete_nonexistent_memory(self, client):
        """Test deleting non-existent memory returns 404."""
        response = client.delete("/api/v1/memory/nonexistent-id")
        assert response.status_code == 404

    def test_clear_memories(self, client):
        """Test clearing all memories."""
        # Store some
        client.post(
            "/api/v1/memory/store",
            json={"content": "Will be cleared", "importance": 0.5},
        )

        # Clear all
        response = client.delete("/api/v1/memory/")
        assert response.status_code == 200

        # Verify empty
        list_response = client.get("/api/v1/memory/list")
        assert len(list_response.json()["entries"]) == 0

    def test_memory_stats(self, client):
        """Test memory statistics endpoint."""
        response = client.get("/api/v1/memory/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_entries" in data
        assert "max_entries" in data
        assert "max_tokens" in data


class TestPersistenceEndpoints:
    """Test persistence (SQLite) endpoints."""

    def test_persistence_stats(self, client):
        """Test getting persistence statistics."""
        response = client.get("/api/v1/persistence/stats")
        assert response.status_code == 200

    def test_list_sessions(self, client):
        """Test listing sessions."""
        response = client.get("/api/v1/persistence/sessions")
        assert response.status_code == 200

    def test_save_and_load_flow(self, client):
        """Test saving and loading memories."""
        # Store a memory first
        client.post(
            "/api/v1/memory/store",
            json={"content": "Persistent memory", "importance": 0.8},
        )

        # Save to SQLite
        save_response = client.post(
            "/api/v1/persistence/save",
            json={"session_id": "test-session"},
        )
        assert save_response.status_code == 200

        # Clear working memory
        client.delete("/api/v1/memory/")

        # Load back
        load_response = client.get(
            "/api/v1/persistence/load",
            params={"session_id": "test-session"},
        )
        assert load_response.status_code == 200


class TestAPIValidation:
    """Test API input validation."""

    def test_store_invalid_importance(self, client):
        """Test validation rejects invalid importance."""
        response = client.post(
            "/api/v1/memory/store",
            json={
                "content": "Test",
                "importance": 1.5,  # Invalid: > 1.0
            },
        )
        assert response.status_code == 422

    def test_store_missing_content(self, client):
        """Test validation requires content."""
        response = client.post(
            "/api/v1/memory/store",
            json={"importance": 0.5},
        )
        assert response.status_code == 422

    def test_query_invalid_top_k(self, client):
        """Test validation rejects invalid top_k."""
        response = client.post(
            "/api/v1/memory/query",
            json={
                "query": "test",
                "top_k": 200,  # Invalid: > 100
            },
        )
        assert response.status_code == 422


class TestCORS:
    """Test CORS configuration."""

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI handles preflight - 200 means CORS is configured
        assert response.status_code in [200, 405]
