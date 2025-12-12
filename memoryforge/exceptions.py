"""Custom exceptions for MemoryForge.

This module defines a hierarchy of exceptions for better error handling
throughout the MemoryForge system.
"""

from typing import Any


class MemoryForgeError(Exception):
    """Base exception for all MemoryForge errors."""

    def __init__(self, message: str = "", details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Storage Exceptions
# =============================================================================


class StorageError(MemoryForgeError):
    """Base exception for storage-related errors."""

    pass


class StorageConnectionError(StorageError):
    """Failed to connect to storage backend."""

    def __init__(self, backend: str, message: str = "", details: dict | None = None):
        self.backend = backend
        super().__init__(
            message or f"Failed to connect to {backend} storage",
            details={"backend": backend, **(details or {})},
        )


class StorageWriteError(StorageError):
    """Failed to write to storage."""

    def __init__(
        self, operation: str, message: str = "", details: dict | None = None
    ):
        self.operation = operation
        super().__init__(
            message or f"Storage write operation failed: {operation}",
            details={"operation": operation, **(details or {})},
        )


class StorageReadError(StorageError):
    """Failed to read from storage."""

    def __init__(self, operation: str, message: str = "", details: dict | None = None):
        self.operation = operation
        super().__init__(
            message or f"Storage read operation failed: {operation}",
            details={"operation": operation, **(details or {})},
        )


class EntryNotFoundError(StorageError):
    """Memory entry not found."""

    def __init__(self, entry_id: str, message: str = "", details: dict | None = None):
        self.entry_id = entry_id
        super().__init__(
            message or f"Memory entry not found: {entry_id}",
            details={"entry_id": entry_id, **(details or {})},
        )


class SessionNotFoundError(StorageError):
    """Session not found."""

    def __init__(self, session_id: str, message: str = "", details: dict | None = None):
        self.session_id = session_id
        super().__init__(
            message or f"Session not found: {session_id}",
            details={"session_id": session_id, **(details or {})},
        )


# =============================================================================
# Memory Exceptions
# =============================================================================


class MemoryError(MemoryForgeError):
    """Base exception for memory-related errors."""

    pass


class MemoryCapacityError(MemoryError):
    """Memory capacity exceeded."""

    def __init__(
        self,
        layer: str,
        current: int,
        maximum: int,
        message: str = "",
        details: dict | None = None,
    ):
        self.layer = layer
        self.current = current
        self.maximum = maximum
        super().__init__(
            message or f"{layer} memory capacity exceeded: {current}/{maximum}",
            details={
                "layer": layer,
                "current": current,
                "maximum": maximum,
                **(details or {}),
            },
        )


class InvalidMemoryLayerError(MemoryError):
    """Invalid memory layer specified."""

    def __init__(self, layer: str, message: str = "", details: dict | None = None):
        self.layer = layer
        super().__init__(
            message or f"Invalid memory layer: {layer}",
            details={"layer": layer, **(details or {})},
        )


class MemoryValidationError(MemoryError):
    """Memory entry validation failed."""

    def __init__(
        self, field: str, reason: str, message: str = "", details: dict | None = None
    ):
        self.field = field
        self.reason = reason
        super().__init__(
            message or f"Validation failed for {field}: {reason}",
            details={"field": field, "reason": reason, **(details or {})},
        )


# =============================================================================
# Retrieval Exceptions
# =============================================================================


class RetrievalError(MemoryForgeError):
    """Base exception for retrieval-related errors."""

    pass


class EmbeddingError(RetrievalError):
    """Failed to generate embedding."""

    def __init__(
        self,
        content_preview: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.content_preview = content_preview[:100] if content_preview else ""
        super().__init__(
            message or "Failed to generate embedding",
            details={"content_preview": self.content_preview, **(details or {})},
        )


class SearchError(RetrievalError):
    """Search operation failed."""

    def __init__(
        self,
        query: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.query = query
        super().__init__(
            message or "Search operation failed",
            details={"query": query[:100] if query else "", **(details or {})},
        )


class IndexError(RetrievalError):
    """Index operation failed."""

    def __init__(
        self,
        operation: str,
        message: str = "",
        details: dict | None = None,
    ):
        self.operation = operation
        super().__init__(
            message or f"Index operation failed: {operation}",
            details={"operation": operation, **(details or {})},
        )


# =============================================================================
# Consolidation Exceptions
# =============================================================================


class ConsolidationError(MemoryForgeError):
    """Base exception for consolidation-related errors."""

    pass


class CompressionError(ConsolidationError):
    """Memory compression failed."""

    def __init__(
        self,
        strategy: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.strategy = strategy
        super().__init__(
            message or f"Compression failed using strategy: {strategy}",
            details={"strategy": strategy, **(details or {})},
        )


class SummarizationError(ConsolidationError):
    """Memory summarization failed."""

    def __init__(
        self,
        level: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.level = level
        super().__init__(
            message or f"Summarization failed at level: {level}",
            details={"level": level, **(details or {})},
        )


# =============================================================================
# LLM Exceptions
# =============================================================================


class LLMError(MemoryForgeError):
    """Base exception for LLM-related errors."""

    pass


class LLMProviderError(LLMError):
    """LLM provider error."""

    def __init__(
        self,
        provider: str,
        message: str = "",
        details: dict | None = None,
    ):
        self.provider = provider
        super().__init__(
            message or f"LLM provider error: {provider}",
            details={"provider": provider, **(details or {})},
        )


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    def __init__(
        self,
        provider: str,
        retry_after: int | None = None,
        message: str = "",
        details: dict | None = None,
    ):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            message or f"Rate limit exceeded for {provider}",
            details={
                "provider": provider,
                "retry_after": retry_after,
                **(details or {}),
            },
        )


class LLMTokenLimitError(LLMError):
    """LLM token limit exceeded."""

    def __init__(
        self,
        tokens_requested: int,
        tokens_limit: int,
        message: str = "",
        details: dict | None = None,
    ):
        self.tokens_requested = tokens_requested
        self.tokens_limit = tokens_limit
        super().__init__(
            message or f"Token limit exceeded: {tokens_requested}/{tokens_limit}",
            details={
                "tokens_requested": tokens_requested,
                "tokens_limit": tokens_limit,
                **(details or {}),
            },
        )


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(MemoryForgeError):
    """Base exception for configuration errors."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration value."""

    def __init__(
        self,
        key: str,
        value: Any,
        reason: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.key = key
        self.value = value
        self.reason = reason
        super().__init__(
            message or f"Invalid configuration for {key}: {reason}",
            details={"key": key, "value": str(value), "reason": reason, **(details or {})},
        )


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing."""

    def __init__(
        self,
        key: str,
        message: str = "",
        details: dict | None = None,
    ):
        self.key = key
        super().__init__(
            message or f"Missing required configuration: {key}",
            details={"key": key, **(details or {})},
        )


# =============================================================================
# Backup Exceptions
# =============================================================================


class BackupError(MemoryForgeError):
    """Base exception for backup-related errors."""

    pass


class BackupCreateError(BackupError):
    """Failed to create backup."""

    def __init__(
        self,
        reason: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.reason = reason
        super().__init__(
            message or f"Failed to create backup: {reason}",
            details={"reason": reason, **(details or {})},
        )


class BackupRestoreError(BackupError):
    """Failed to restore from backup."""

    def __init__(
        self,
        backup_path: str = "",
        reason: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.backup_path = backup_path
        self.reason = reason
        super().__init__(
            message or f"Failed to restore backup: {reason}",
            details={"backup_path": backup_path, "reason": reason, **(details or {})},
        )


class BackupVerificationError(BackupError):
    """Backup verification failed."""

    def __init__(
        self,
        backup_path: str = "",
        reason: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.backup_path = backup_path
        self.reason = reason
        super().__init__(
            message or f"Backup verification failed: {reason}",
            details={"backup_path": backup_path, "reason": reason, **(details or {})},
        )


# =============================================================================
# Hook Exceptions
# =============================================================================


class HookError(MemoryForgeError):
    """Base exception for hook-related errors."""

    pass


class HookExecutionError(HookError):
    """Hook execution failed."""

    def __init__(
        self,
        hook_name: str,
        event_type: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.hook_name = hook_name
        self.event_type = event_type
        super().__init__(
            message or f"Hook '{hook_name}' execution failed",
            details={
                "hook_name": hook_name,
                "event_type": event_type,
                **(details or {}),
            },
        )


class HookRegistrationError(HookError):
    """Failed to register hook."""

    def __init__(
        self,
        hook_name: str,
        reason: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.hook_name = hook_name
        self.reason = reason
        super().__init__(
            message or f"Failed to register hook '{hook_name}': {reason}",
            details={"hook_name": hook_name, "reason": reason, **(details or {})},
        )


# =============================================================================
# API Exceptions
# =============================================================================


class APIError(MemoryForgeError):
    """Base exception for API-related errors."""

    status_code: int = 500

    def __init__(
        self,
        message: str = "",
        status_code: int | None = None,
        details: dict | None = None,
    ):
        if status_code:
            self.status_code = status_code
        super().__init__(message or "API error occurred", details)


class AuthenticationError(APIError):
    """Authentication failed."""

    status_code = 401

    def __init__(
        self,
        message: str = "",
        details: dict | None = None,
    ):
        super().__init__(message or "Authentication failed", details=details)


class AuthorizationError(APIError):
    """Authorization failed (forbidden)."""

    status_code = 403

    def __init__(
        self,
        resource: str = "",
        message: str = "",
        details: dict | None = None,
    ):
        self.resource = resource
        super().__init__(
            message or f"Access denied to resource: {resource}",
            details={"resource": resource, **(details or {})},
        )


class RateLimitError(APIError):
    """API rate limit exceeded."""

    status_code = 429

    def __init__(
        self,
        limit: int,
        window: str = "",
        retry_after: int | None = None,
        message: str = "",
        details: dict | None = None,
    ):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(
            message or f"Rate limit exceeded: {limit} requests per {window}",
            details={
                "limit": limit,
                "window": window,
                "retry_after": retry_after,
                **(details or {}),
            },
        )


class ValidationError(APIError):
    """Request validation failed."""

    status_code = 422

    def __init__(
        self,
        errors: list[dict] | None = None,
        message: str = "",
        details: dict | None = None,
    ):
        self.errors = errors or []
        super().__init__(
            message or "Request validation failed",
            details={"validation_errors": self.errors, **(details or {})},
        )
