"""Unit tests for custom exceptions."""

import pytest

from memoryforge.exceptions import (
    # Base
    MemoryForgeError,
    # Storage
    StorageError,
    StorageConnectionError,
    StorageWriteError,
    StorageReadError,
    EntryNotFoundError,
    SessionNotFoundError,
    # Memory
    MemoryError,
    MemoryCapacityError,
    InvalidMemoryLayerError,
    MemoryValidationError,
    # Retrieval
    RetrievalError,
    EmbeddingError,
    SearchError,
    IndexError,
    # Consolidation
    ConsolidationError,
    CompressionError,
    SummarizationError,
    # LLM
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTokenLimitError,
    # Configuration
    ConfigurationError,
    InvalidConfigurationError,
    MissingConfigurationError,
    # Backup
    BackupError,
    BackupCreateError,
    BackupRestoreError,
    BackupVerificationError,
    # Hook
    HookError,
    HookExecutionError,
    HookRegistrationError,
    # API
    APIError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ValidationError,
)


class TestMemoryForgeError:
    """Tests for base exception."""

    def test_basic_creation(self):
        """Test creating a basic exception."""
        error = MemoryForgeError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_with_details(self):
        """Test creating exception with details."""
        error = MemoryForgeError("Error", details={"key": "value"})
        assert error.details == {"key": "value"}

    def test_to_dict(self):
        """Test converting to dictionary."""
        error = MemoryForgeError("Test error", details={"extra": "info"})
        result = error.to_dict()

        assert result["error"] == "MemoryForgeError"
        assert result["message"] == "Test error"
        assert result["details"] == {"extra": "info"}

    def test_inheritance(self):
        """Test that it inherits from Exception."""
        error = MemoryForgeError("Test")
        assert isinstance(error, Exception)


class TestStorageExceptions:
    """Tests for storage-related exceptions."""

    def test_storage_connection_error(self):
        """Test StorageConnectionError."""
        error = StorageConnectionError("sqlite")
        assert "sqlite" in str(error)
        assert error.backend == "sqlite"
        assert error.details["backend"] == "sqlite"

    def test_storage_write_error(self):
        """Test StorageWriteError."""
        error = StorageWriteError("insert", details={"table": "memories"})
        assert "insert" in str(error)
        assert error.operation == "insert"
        assert error.details["table"] == "memories"

    def test_storage_read_error(self):
        """Test StorageReadError."""
        error = StorageReadError("select")
        assert "select" in str(error)
        assert error.operation == "select"

    def test_entry_not_found_error(self):
        """Test EntryNotFoundError."""
        error = EntryNotFoundError("abc-123")
        assert "abc-123" in str(error)
        assert error.entry_id == "abc-123"

    def test_session_not_found_error(self):
        """Test SessionNotFoundError."""
        error = SessionNotFoundError("session-xyz")
        assert "session-xyz" in str(error)
        assert error.session_id == "session-xyz"

    def test_inheritance(self):
        """Test inheritance chain."""
        error = EntryNotFoundError("id")
        assert isinstance(error, StorageError)
        assert isinstance(error, MemoryForgeError)


class TestMemoryExceptions:
    """Tests for memory-related exceptions."""

    def test_memory_capacity_error(self):
        """Test MemoryCapacityError."""
        error = MemoryCapacityError("working", 100, 50)
        assert "working" in str(error)
        assert "100" in str(error)
        assert error.layer == "working"
        assert error.current == 100
        assert error.maximum == 50

    def test_invalid_memory_layer_error(self):
        """Test InvalidMemoryLayerError."""
        error = InvalidMemoryLayerError("unknown")
        assert "unknown" in str(error)
        assert error.layer == "unknown"

    def test_memory_validation_error(self):
        """Test MemoryValidationError."""
        error = MemoryValidationError("content", "cannot be empty")
        assert "content" in str(error)
        assert error.field == "content"
        assert error.reason == "cannot be empty"


class TestRetrievalExceptions:
    """Tests for retrieval-related exceptions."""

    def test_embedding_error(self):
        """Test EmbeddingError."""
        error = EmbeddingError("long content " * 20)
        assert error.content_preview
        assert len(error.content_preview) <= 100

    def test_search_error(self):
        """Test SearchError."""
        error = SearchError("test query")
        assert error.query == "test query"

    def test_index_error(self):
        """Test IndexError."""
        error = IndexError("rebuild")
        assert error.operation == "rebuild"


class TestConsolidationExceptions:
    """Tests for consolidation-related exceptions."""

    def test_compression_error(self):
        """Test CompressionError."""
        error = CompressionError("aggressive")
        assert error.strategy == "aggressive"

    def test_summarization_error(self):
        """Test SummarizationError."""
        error = SummarizationError("brief")
        assert error.level == "brief"


class TestLLMExceptions:
    """Tests for LLM-related exceptions."""

    def test_llm_provider_error(self):
        """Test LLMProviderError."""
        error = LLMProviderError("openai")
        assert error.provider == "openai"

    def test_llm_rate_limit_error(self):
        """Test LLMRateLimitError."""
        error = LLMRateLimitError("anthropic", retry_after=60)
        assert error.provider == "anthropic"
        assert error.retry_after == 60

    def test_llm_token_limit_error(self):
        """Test LLMTokenLimitError."""
        error = LLMTokenLimitError(10000, 8000)
        assert error.tokens_requested == 10000
        assert error.tokens_limit == 8000


class TestConfigurationExceptions:
    """Tests for configuration-related exceptions."""

    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        error = InvalidConfigurationError("max_tokens", -1, "must be positive")
        assert error.key == "max_tokens"
        assert error.value == -1
        assert error.reason == "must be positive"

    def test_missing_configuration_error(self):
        """Test MissingConfigurationError."""
        error = MissingConfigurationError("api_key")
        assert error.key == "api_key"


class TestBackupExceptions:
    """Tests for backup-related exceptions."""

    def test_backup_create_error(self):
        """Test BackupCreateError."""
        error = BackupCreateError("disk full")
        assert error.reason == "disk full"

    def test_backup_restore_error(self):
        """Test BackupRestoreError."""
        error = BackupRestoreError("/path/to/backup.json", "corrupted")
        assert error.backup_path == "/path/to/backup.json"
        assert error.reason == "corrupted"

    def test_backup_verification_error(self):
        """Test BackupVerificationError."""
        error = BackupVerificationError("/path/backup.json", "checksum mismatch")
        assert error.backup_path == "/path/backup.json"
        assert error.reason == "checksum mismatch"


class TestHookExceptions:
    """Tests for hook-related exceptions."""

    def test_hook_execution_error(self):
        """Test HookExecutionError."""
        error = HookExecutionError("my_hook", "memory.created")
        assert error.hook_name == "my_hook"
        assert error.event_type == "memory.created"

    def test_hook_registration_error(self):
        """Test HookRegistrationError."""
        error = HookRegistrationError("bad_hook", "invalid callback")
        assert error.hook_name == "bad_hook"
        assert error.reason == "invalid callback"


class TestAPIExceptions:
    """Tests for API-related exceptions."""

    def test_api_error_default_status(self):
        """Test APIError with default status."""
        error = APIError("Server error")
        assert error.status_code == 500

    def test_api_error_custom_status(self):
        """Test APIError with custom status."""
        error = APIError("Not found", status_code=404)
        assert error.status_code == 404

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid token")
        assert error.status_code == 401

    def test_authorization_error(self):
        """Test AuthorizationError."""
        error = AuthorizationError("/admin/users")
        assert error.status_code == 403
        assert error.resource == "/admin/users"

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError(100, "hour", retry_after=3600)
        assert error.status_code == 429
        assert error.limit == 100
        assert error.window == "hour"
        assert error.retry_after == 3600

    def test_validation_error(self):
        """Test ValidationError."""
        errors = [
            {"field": "name", "error": "required"},
            {"field": "email", "error": "invalid format"},
        ]
        error = ValidationError(errors=errors)
        assert error.status_code == 422
        assert len(error.errors) == 2


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_inherit_from_base(self):
        """Test all exceptions inherit from MemoryForgeError."""
        exception_classes = [
            StorageError,
            StorageConnectionError,
            MemoryError,
            MemoryCapacityError,
            RetrievalError,
            EmbeddingError,
            ConsolidationError,
            CompressionError,
            LLMError,
            LLMProviderError,
            ConfigurationError,
            BackupError,
            HookError,
            APIError,
        ]

        for exc_class in exception_classes:
            error = exc_class.__new__(exc_class)
            assert isinstance(error, MemoryForgeError)

    def test_storage_exceptions_inherit_from_storage_error(self):
        """Test storage exceptions inherit from StorageError."""
        storage_classes = [
            StorageConnectionError,
            StorageWriteError,
            StorageReadError,
            EntryNotFoundError,
            SessionNotFoundError,
        ]

        for exc_class in storage_classes:
            assert issubclass(exc_class, StorageError)

    def test_api_exceptions_have_status_codes(self):
        """Test API exceptions have appropriate status codes."""
        assert AuthenticationError("").status_code == 401
        assert AuthorizationError("").status_code == 403
        assert RateLimitError(0, "").status_code == 429
        assert ValidationError().status_code == 422


class TestExceptionUsage:
    """Tests for real-world exception usage patterns."""

    def test_raising_and_catching(self):
        """Test raising and catching exceptions."""
        with pytest.raises(EntryNotFoundError) as exc_info:
            raise EntryNotFoundError("missing-id")

        assert exc_info.value.entry_id == "missing-id"

    def test_catching_base_exception(self):
        """Test catching by base exception."""
        with pytest.raises(MemoryForgeError):
            raise StorageConnectionError("redis")

    def test_exception_chaining(self):
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise StorageWriteError("insert") from e
        except StorageWriteError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)

    def test_to_dict_for_api_response(self):
        """Test converting exception to API response."""
        error = ValidationError(
            errors=[{"field": "name", "error": "required"}],
            message="Validation failed",
        )

        response = error.to_dict()

        assert response["error"] == "ValidationError"
        assert response["message"] == "Validation failed"
        assert "validation_errors" in response["details"]
