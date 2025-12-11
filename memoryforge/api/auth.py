"""Authentication middleware for the MemoryForge API."""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Awaitable

import structlog
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()


class AuthMethod(Enum):
    """Authentication methods."""

    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC = "basic"


class Permission(Enum):
    """API permissions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class AuthConfig:
    """Configuration for authentication."""

    # Enable authentication
    enabled: bool = True

    # Allow unauthenticated access to certain paths
    public_paths: list[str] = field(default_factory=lambda: [
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    ])

    # API key settings
    api_key_header: str = "X-API-Key"
    api_key_query_param: str = "api_key"

    # Token settings
    token_expiry_hours: int = 24
    token_refresh_hours: int = 168  # 7 days

    # Rate limiting per auth level
    rate_limits: dict[str, int] = field(default_factory=lambda: {
        "anonymous": 10,
        "authenticated": 100,
        "admin": 1000,
    })


@dataclass
class APIKey:
    """An API key with associated permissions."""

    key_id: str
    key_hash: str  # Store hash, not plaintext
    name: str
    permissions: list[Permission]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    last_used: datetime | None = None
    is_active: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "name": self.name,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "is_active": self.is_active,
        }


@dataclass
class AuthToken:
    """An authentication token."""

    token_id: str
    user_id: str
    permissions: list[Permission]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24))
    refresh_token: str | None = None

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "user_id": self.user_id,
            "permissions": [p.value for p in self.permissions],
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass
class AuthenticatedUser:
    """Represents an authenticated user/client."""

    user_id: str
    auth_method: AuthMethod
    permissions: list[Permission]
    metadata: dict = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return Permission.ADMIN in self.permissions or permission in self.permissions

    def can_read(self) -> bool:
        return self.has_permission(Permission.READ)

    def can_write(self) -> bool:
        return self.has_permission(Permission.WRITE)

    def can_delete(self) -> bool:
        return self.has_permission(Permission.DELETE)

    def is_admin(self) -> bool:
        return Permission.ADMIN in self.permissions


class APIKeyManager:
    """Manages API keys for authentication."""

    def __init__(self):
        """Initialize API key manager."""
        self._keys: dict[str, APIKey] = {}
        self._key_id_to_hash: dict[str, str] = {}

    def generate_key(
        self,
        name: str,
        permissions: list[Permission] | None = None,
        expires_in_days: int | None = None,
    ) -> tuple[str, APIKey]:
        """Generate a new API key.

        Args:
            name: Key name/description
            permissions: Permissions for this key
            expires_in_days: Days until expiration

        Returns:
            Tuple of (plaintext_key, APIKey object)
        """
        # Generate secure random key
        plaintext_key = f"mf_{secrets.token_urlsafe(32)}"
        key_id = secrets.token_hex(8)
        key_hash = self._hash_key(plaintext_key)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions or [Permission.READ, Permission.WRITE],
            expires_at=expires_at,
        )

        self._keys[key_hash] = api_key
        self._key_id_to_hash[key_id] = key_hash

        logger.info("API key generated", key_id=key_id, name=name)

        return plaintext_key, api_key

    def validate_key(self, plaintext_key: str) -> APIKey | None:
        """Validate an API key.

        Args:
            plaintext_key: The key to validate

        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = self._hash_key(plaintext_key)
        api_key = self._keys.get(key_hash)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at and datetime.now(timezone.utc) > api_key.expires_at:
            return None

        # Update last used
        api_key.last_used = datetime.now(timezone.utc)

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: ID of key to revoke

        Returns:
            True if revoked
        """
        key_hash = self._key_id_to_hash.get(key_id)
        if key_hash and key_hash in self._keys:
            self._keys[key_hash].is_active = False
            logger.info("API key revoked", key_id=key_id)
            return True
        return False

    def get_key_info(self, key_id: str) -> APIKey | None:
        """Get API key info by ID.

        Args:
            key_id: Key ID

        Returns:
            APIKey or None
        """
        key_hash = self._key_id_to_hash.get(key_id)
        if key_hash:
            return self._keys.get(key_hash)
        return None

    def list_keys(self) -> list[APIKey]:
        """List all API keys.

        Returns:
            List of API keys
        """
        return list(self._keys.values())

    def _hash_key(self, plaintext_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(plaintext_key.encode()).hexdigest()


class TokenManager:
    """Manages authentication tokens."""

    def __init__(self, config: AuthConfig | None = None):
        """Initialize token manager.

        Args:
            config: Auth configuration
        """
        self._config = config or AuthConfig()
        self._tokens: dict[str, AuthToken] = {}
        self._refresh_tokens: dict[str, str] = {}  # refresh_token -> token_id
        self._secret = secrets.token_hex(32)

    def create_token(
        self,
        user_id: str,
        permissions: list[Permission],
    ) -> AuthToken:
        """Create a new authentication token.

        Args:
            user_id: User identifier
            permissions: Token permissions

        Returns:
            Authentication token
        """
        token_id = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        token = AuthToken(
            token_id=token_id,
            user_id=user_id,
            permissions=permissions,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=self._config.token_expiry_hours),
            refresh_token=refresh_token,
        )

        self._tokens[token_id] = token
        self._refresh_tokens[refresh_token] = token_id

        logger.debug("Token created", user_id=user_id)

        return token

    def validate_token(self, token_id: str) -> AuthToken | None:
        """Validate an authentication token.

        Args:
            token_id: Token to validate

        Returns:
            AuthToken if valid, None otherwise
        """
        token = self._tokens.get(token_id)

        if not token:
            return None

        if token.is_expired():
            self._cleanup_token(token_id)
            return None

        return token

    def refresh_token(self, refresh_token: str) -> AuthToken | None:
        """Refresh an authentication token.

        Args:
            refresh_token: Refresh token

        Returns:
            New AuthToken or None
        """
        token_id = self._refresh_tokens.get(refresh_token)
        if not token_id:
            return None

        old_token = self._tokens.get(token_id)
        if not old_token:
            return None

        # Create new token
        new_token = self.create_token(
            old_token.user_id,
            old_token.permissions,
        )

        # Cleanup old token
        self._cleanup_token(token_id)

        return new_token

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token.

        Args:
            token_id: Token to revoke

        Returns:
            True if revoked
        """
        return self._cleanup_token(token_id)

    def _cleanup_token(self, token_id: str) -> bool:
        """Remove a token and its refresh token."""
        token = self._tokens.pop(token_id, None)
        if token and token.refresh_token:
            self._refresh_tokens.pop(token.refresh_token, None)
        return token is not None


class AuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for authentication."""

    def __init__(
        self,
        app,
        config: AuthConfig | None = None,
        key_manager: APIKeyManager | None = None,
        token_manager: TokenManager | None = None,
    ):
        """Initialize authentication middleware.

        Args:
            app: FastAPI application
            config: Auth configuration
            key_manager: API key manager
            token_manager: Token manager
        """
        super().__init__(app)
        self._config = config or AuthConfig()
        self._key_manager = key_manager or APIKeyManager()
        self._token_manager = token_manager or TokenManager(self._config)

        # Statistics
        self._auth_attempts = 0
        self._auth_successes = 0
        self._auth_failures = 0

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request with authentication.

        Args:
            request: Incoming request
            call_next: Next handler

        Returns:
            Response
        """
        if not self._config.enabled:
            return await call_next(request)

        # Check public paths
        path = request.url.path
        if self._is_public_path(path):
            return await call_next(request)

        self._auth_attempts += 1

        # Try to authenticate
        user = await self._authenticate(request)

        if not user:
            self._auth_failures += 1
            return self._unauthorized_response()

        self._auth_successes += 1

        # Store user in request state
        request.state.user = user

        # Process request
        response = await call_next(request)

        return response

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        for public_path in self._config.public_paths:
            if path == public_path or path.startswith(public_path + "/"):
                return True
        return False

    async def _authenticate(self, request: Request) -> AuthenticatedUser | None:
        """Attempt to authenticate the request.

        Args:
            request: Incoming request

        Returns:
            AuthenticatedUser or None
        """
        # Try API key in header
        api_key = request.headers.get(self._config.api_key_header)
        if api_key:
            return self._auth_api_key(api_key)

        # Try API key in query param
        api_key = request.query_params.get(self._config.api_key_query_param)
        if api_key:
            return self._auth_api_key(api_key)

        # Try Bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return self._auth_bearer_token(token)

        return None

    def _auth_api_key(self, api_key: str) -> AuthenticatedUser | None:
        """Authenticate using API key.

        Args:
            api_key: API key

        Returns:
            AuthenticatedUser or None
        """
        validated = self._key_manager.validate_key(api_key)
        if not validated:
            return None

        return AuthenticatedUser(
            user_id=f"key:{validated.key_id}",
            auth_method=AuthMethod.API_KEY,
            permissions=validated.permissions,
            metadata={"key_name": validated.name},
        )

    def _auth_bearer_token(self, token: str) -> AuthenticatedUser | None:
        """Authenticate using Bearer token.

        Args:
            token: Bearer token

        Returns:
            AuthenticatedUser or None
        """
        validated = self._token_manager.validate_token(token)
        if not validated:
            return None

        return AuthenticatedUser(
            user_id=validated.user_id,
            auth_method=AuthMethod.BEARER_TOKEN,
            permissions=validated.permissions,
        )

    def _unauthorized_response(self) -> Response:
        """Create unauthorized response."""
        return Response(
            content='{"detail": "Unauthorized"}',
            status_code=401,
            headers={
                "Content-Type": "application/json",
                "WWW-Authenticate": "Bearer",
            },
        )

    def get_stats(self) -> dict:
        """Get authentication statistics."""
        return {
            "enabled": self._config.enabled,
            "auth_attempts": self._auth_attempts,
            "auth_successes": self._auth_successes,
            "auth_failures": self._auth_failures,
            "success_rate": (
                self._auth_successes / self._auth_attempts
                if self._auth_attempts > 0 else 0
            ),
            "api_keys_count": len(self._key_manager.list_keys()),
        }


def require_permission(permission: Permission):
    """Decorator to require specific permission.

    Args:
        permission: Required permission

    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user = getattr(request.state, "user", None)
            if not user:
                raise HTTPException(status_code=401, detail="Unauthorized")

            if not user.has_permission(permission):
                raise HTTPException(status_code=403, detail="Forbidden")

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


def get_current_user(request: Request) -> AuthenticatedUser | None:
    """Get the current authenticated user from request.

    Args:
        request: FastAPI request

    Returns:
        AuthenticatedUser or None
    """
    return getattr(request.state, "user", None)
