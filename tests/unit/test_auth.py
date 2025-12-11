"""Tests for API authentication functionality."""

import pytest
from datetime import datetime, timezone, timedelta

from memoryforge.api.auth import (
    APIKeyManager,
    TokenManager,
    AuthConfig,
    AuthMethod,
    Permission,
    APIKey,
    AuthToken,
    AuthenticatedUser,
)


class TestAPIKeyManager:
    """Tests for APIKeyManager."""

    def test_generate_key(self):
        """Test API key generation."""
        manager = APIKeyManager()

        plaintext, api_key = manager.generate_key(
            name="Test Key",
            permissions=[Permission.READ, Permission.WRITE],
        )

        assert plaintext.startswith("mf_")
        assert len(plaintext) > 20
        assert api_key.name == "Test Key"
        assert Permission.READ in api_key.permissions

    def test_validate_key_success(self):
        """Test successful key validation."""
        manager = APIKeyManager()
        plaintext, api_key = manager.generate_key(name="Test")

        validated = manager.validate_key(plaintext)

        assert validated is not None
        assert validated.key_id == api_key.key_id

    def test_validate_key_invalid(self):
        """Test invalid key validation."""
        manager = APIKeyManager()

        validated = manager.validate_key("invalid_key")

        assert validated is None

    def test_validate_key_revoked(self):
        """Test revoked key validation."""
        manager = APIKeyManager()
        plaintext, api_key = manager.generate_key(name="Test")

        manager.revoke_key(api_key.key_id)
        validated = manager.validate_key(plaintext)

        assert validated is None

    def test_validate_key_expired(self):
        """Test expired key validation."""
        manager = APIKeyManager()
        plaintext, api_key = manager.generate_key(
            name="Test",
            expires_in_days=0,  # Expires immediately
        )

        # Set expiration to past
        api_key.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        validated = manager.validate_key(plaintext)

        assert validated is None

    def test_revoke_key(self):
        """Test key revocation."""
        manager = APIKeyManager()
        _, api_key = manager.generate_key(name="Test")

        result = manager.revoke_key(api_key.key_id)

        assert result is True
        assert api_key.is_active is False

    def test_revoke_nonexistent_key(self):
        """Test revoking nonexistent key."""
        manager = APIKeyManager()

        result = manager.revoke_key("nonexistent")

        assert result is False

    def test_get_key_info(self):
        """Test getting key info."""
        manager = APIKeyManager()
        _, api_key = manager.generate_key(name="Test")

        info = manager.get_key_info(api_key.key_id)

        assert info is not None
        assert info.name == "Test"

    def test_list_keys(self):
        """Test listing all keys."""
        manager = APIKeyManager()
        manager.generate_key(name="Key 1")
        manager.generate_key(name="Key 2")

        keys = manager.list_keys()

        assert len(keys) == 2

    def test_key_last_used_updated(self):
        """Test that last_used is updated on validation."""
        manager = APIKeyManager()
        plaintext, api_key = manager.generate_key(name="Test")

        assert api_key.last_used is None

        manager.validate_key(plaintext)

        assert api_key.last_used is not None


class TestTokenManager:
    """Tests for TokenManager."""

    def test_create_token(self):
        """Test token creation."""
        manager = TokenManager()

        token = manager.create_token(
            user_id="user123",
            permissions=[Permission.READ],
        )

        assert token.token_id is not None
        assert token.user_id == "user123"
        assert token.refresh_token is not None

    def test_validate_token_success(self):
        """Test successful token validation."""
        manager = TokenManager()
        token = manager.create_token(
            user_id="user123",
            permissions=[Permission.READ],
        )

        validated = manager.validate_token(token.token_id)

        assert validated is not None
        assert validated.user_id == "user123"

    def test_validate_token_invalid(self):
        """Test invalid token validation."""
        manager = TokenManager()

        validated = manager.validate_token("invalid_token")

        assert validated is None

    def test_validate_token_expired(self):
        """Test expired token validation."""
        manager = TokenManager()
        token = manager.create_token(
            user_id="user123",
            permissions=[Permission.READ],
        )

        # Set expiration to past
        token.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        validated = manager.validate_token(token.token_id)

        assert validated is None

    def test_refresh_token(self):
        """Test token refresh."""
        manager = TokenManager()
        original_token = manager.create_token(
            user_id="user123",
            permissions=[Permission.READ, Permission.WRITE],
        )

        new_token = manager.refresh_token(original_token.refresh_token)

        assert new_token is not None
        assert new_token.user_id == "user123"
        assert new_token.token_id != original_token.token_id

    def test_refresh_invalid_token(self):
        """Test refreshing invalid token."""
        manager = TokenManager()

        new_token = manager.refresh_token("invalid_refresh")

        assert new_token is None

    def test_revoke_token(self):
        """Test token revocation."""
        manager = TokenManager()
        token = manager.create_token(
            user_id="user123",
            permissions=[Permission.READ],
        )

        result = manager.revoke_token(token.token_id)

        assert result is True
        assert manager.validate_token(token.token_id) is None


class TestAuthenticatedUser:
    """Tests for AuthenticatedUser."""

    def test_has_permission(self):
        """Test permission checking."""
        user = AuthenticatedUser(
            user_id="user123",
            auth_method=AuthMethod.API_KEY,
            permissions=[Permission.READ, Permission.WRITE],
        )

        assert user.has_permission(Permission.READ) is True
        assert user.has_permission(Permission.WRITE) is True
        assert user.has_permission(Permission.DELETE) is False

    def test_admin_has_all_permissions(self):
        """Test that admin has all permissions."""
        user = AuthenticatedUser(
            user_id="admin",
            auth_method=AuthMethod.API_KEY,
            permissions=[Permission.ADMIN],
        )

        assert user.has_permission(Permission.READ) is True
        assert user.has_permission(Permission.WRITE) is True
        assert user.has_permission(Permission.DELETE) is True

    def test_can_read(self):
        """Test can_read convenience method."""
        user = AuthenticatedUser(
            user_id="user123",
            auth_method=AuthMethod.API_KEY,
            permissions=[Permission.READ],
        )

        assert user.can_read() is True
        assert user.can_write() is False

    def test_can_write(self):
        """Test can_write convenience method."""
        user = AuthenticatedUser(
            user_id="user123",
            auth_method=AuthMethod.API_KEY,
            permissions=[Permission.WRITE],
        )

        assert user.can_write() is True

    def test_can_delete(self):
        """Test can_delete convenience method."""
        user = AuthenticatedUser(
            user_id="user123",
            auth_method=AuthMethod.API_KEY,
            permissions=[Permission.DELETE],
        )

        assert user.can_delete() is True

    def test_is_admin(self):
        """Test is_admin check."""
        admin = AuthenticatedUser(
            user_id="admin",
            auth_method=AuthMethod.API_KEY,
            permissions=[Permission.ADMIN],
        )
        regular = AuthenticatedUser(
            user_id="user",
            auth_method=AuthMethod.API_KEY,
            permissions=[Permission.READ],
        )

        assert admin.is_admin() is True
        assert regular.is_admin() is False


class TestAPIKey:
    """Tests for APIKey dataclass."""

    def test_to_dict(self):
        """Test APIKey serialization."""
        api_key = APIKey(
            key_id="test123",
            key_hash="hash",
            name="Test Key",
            permissions=[Permission.READ, Permission.WRITE],
        )

        data = api_key.to_dict()

        assert data["key_id"] == "test123"
        assert data["name"] == "Test Key"
        assert "read" in data["permissions"]
        assert "write" in data["permissions"]
        assert data["is_active"] is True


class TestAuthToken:
    """Tests for AuthToken dataclass."""

    def test_is_expired_false(self):
        """Test token not expired."""
        token = AuthToken(
            token_id="token123",
            user_id="user123",
            permissions=[Permission.READ],
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert token.is_expired() is False

    def test_is_expired_true(self):
        """Test token expired."""
        token = AuthToken(
            token_id="token123",
            user_id="user123",
            permissions=[Permission.READ],
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert token.is_expired() is True

    def test_to_dict(self):
        """Test AuthToken serialization."""
        token = AuthToken(
            token_id="token123",
            user_id="user123",
            permissions=[Permission.READ],
        )

        data = token.to_dict()

        assert data["token_id"] == "token123"
        assert data["user_id"] == "user123"
        assert "read" in data["permissions"]


class TestAuthConfig:
    """Tests for AuthConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AuthConfig()

        assert config.enabled is True
        assert "/health" in config.public_paths
        assert config.api_key_header == "X-API-Key"

    def test_custom_config(self):
        """Test custom configuration."""
        config = AuthConfig(
            enabled=False,
            api_key_header="X-Custom-Key",
        )

        assert config.enabled is False
        assert config.api_key_header == "X-Custom-Key"


class TestPermission:
    """Tests for Permission enum."""

    def test_all_permissions_defined(self):
        """Test all permissions are defined."""
        permissions = list(Permission)

        assert Permission.READ in permissions
        assert Permission.WRITE in permissions
        assert Permission.DELETE in permissions
        assert Permission.ADMIN in permissions


class TestAuthMethod:
    """Tests for AuthMethod enum."""

    def test_all_methods_defined(self):
        """Test all auth methods are defined."""
        methods = list(AuthMethod)

        assert AuthMethod.API_KEY in methods
        assert AuthMethod.BEARER_TOKEN in methods
        assert AuthMethod.BASIC in methods
