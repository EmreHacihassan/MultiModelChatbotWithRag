"""
Enterprise AI Assistant - Unit Tests for Exceptions
====================================================

Custom exception sınıfları için kapsamlı birim testleri.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Import exceptions
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.exceptions import (
        BaseAppException,
        ErrorSeverity,
        LLMException,
        LLMConnectionError,
        LLMTimeoutError,
        LLMModelNotFoundError,
        LLMRateLimitError,
        RAGException,
        DocumentNotFoundError,
        EmbeddingError,
        VectorStoreError,
        ChunkingError,
        AgentException,
        AgentTimeoutError,
        AgentMaxIterationsError,
        ToolExecutionError,
        APIException,
        ValidationError,
        AuthenticationError,
        AuthorizationError,
        RateLimitError,
        SessionException,
        SessionNotFoundError,
        SessionExpiredError,
        ConfigurationError,
        ExternalServiceError,
        CircuitBreakerOpenError,
        handle_exception,
    )
except ImportError as e:
    pytest.skip(f"Could not import exceptions: {e}", allow_module_level=True)


class TestBaseAppException:
    """BaseAppException testleri."""
    
    def test_basic_creation(self):
        """Temel exception oluşturma."""
        exc = BaseAppException("Test error")
        
        assert str(exc) == "Test error"
        assert exc.error_id is not None
        assert len(exc.error_id) == 8
    
    def test_with_severity(self):
        """Severity ile oluşturma."""
        exc = BaseAppException("Test", severity=ErrorSeverity.CRITICAL)
        
        assert exc.severity == ErrorSeverity.CRITICAL
    
    def test_with_user_message(self):
        """User message ile oluşturma."""
        exc = BaseAppException(
            "Internal error",
            user_message="Something went wrong"
        )
        
        assert exc.user_message == "Something went wrong"
    
    def test_retryable_flag(self):
        """Retryable flag."""
        exc1 = BaseAppException("Test", retryable=True)
        exc2 = BaseAppException("Test", retryable=False)
        
        assert exc1.retryable is True
        assert exc2.retryable is False
    
    def test_to_dict(self):
        """Dict'e dönüştürme."""
        exc = BaseAppException(
            "Test error",
            severity=ErrorSeverity.HIGH,
            retryable=True,
            user_message="User friendly message"
        )
        
        result = exc.to_dict()
        
        assert result["error_id"] is not None
        assert result["error_type"] == "BaseAppException"
        assert result["message"] == "Test error"
        assert result["severity"] == "high"
        assert result["retryable"] is True
        assert result["user_message"] == "User friendly message"
    
    def test_with_details(self):
        """Details ile oluşturma."""
        exc = BaseAppException(
            "Test",
            details={"key": "value", "count": 42}
        )
        
        result = exc.to_dict()
        assert result["details"]["key"] == "value"
        assert result["details"]["count"] == 42


class TestLLMExceptions:
    """LLM exception testleri."""
    
    def test_llm_exception(self):
        """Genel LLM exception."""
        exc = LLMException("LLM error", model="qwen2.5:7b")
        
        assert exc.model == "qwen2.5:7b"
        assert "LLM" in str(type(exc).__name__)
    
    def test_llm_connection_error(self):
        """LLM bağlantı hatası."""
        exc = LLMConnectionError(
            "Connection failed",
            model="qwen2.5:7b"
        )
        
        assert exc.retryable is True
        assert exc.severity == ErrorSeverity.HIGH
    
    def test_llm_timeout_error(self):
        """LLM timeout hatası."""
        exc = LLMTimeoutError(
            "Request timed out",
            model="qwen2.5:7b",
            timeout_seconds=120
        )
        
        assert exc.timeout_seconds == 120
        assert exc.retryable is True
    
    def test_llm_model_not_found(self):
        """Model bulunamadı hatası."""
        exc = LLMModelNotFoundError(
            "Model not found",
            model="nonexistent:model"
        )
        
        assert exc.retryable is False
    
    def test_llm_rate_limit(self):
        """LLM rate limit hatası."""
        exc = LLMRateLimitError(
            "Rate limited",
            model="qwen2.5:7b",
            retry_after=60
        )
        
        assert exc.retry_after == 60
        assert exc.retryable is True


class TestRAGExceptions:
    """RAG exception testleri."""
    
    def test_rag_exception(self):
        """Genel RAG exception."""
        exc = RAGException("RAG error")
        
        assert isinstance(exc, BaseAppException)
    
    def test_document_not_found(self):
        """Döküman bulunamadı hatası."""
        exc = DocumentNotFoundError(
            "Document not found",
            document_id="doc123"
        )
        
        assert exc.document_id == "doc123"
    
    def test_embedding_error(self):
        """Embedding hatası."""
        exc = EmbeddingError("Embedding failed")
        
        assert exc.retryable is True
    
    def test_vector_store_error(self):
        """Vector store hatası."""
        exc = VectorStoreError("ChromaDB error")
        
        assert isinstance(exc, RAGException)
    
    def test_chunking_error(self):
        """Chunking hatası."""
        exc = ChunkingError("Chunking failed")
        
        assert isinstance(exc, RAGException)


class TestAgentExceptions:
    """Agent exception testleri."""
    
    def test_agent_exception(self):
        """Genel agent exception."""
        exc = AgentException(
            "Agent error",
            agent_type="research"
        )
        
        assert exc.agent_type == "research"
    
    def test_agent_timeout(self):
        """Agent timeout hatası."""
        exc = AgentTimeoutError(
            "Agent timed out",
            agent_type="analyzer",
            timeout_seconds=300
        )
        
        assert exc.timeout_seconds == 300
    
    def test_max_iterations(self):
        """Max iterations hatası."""
        exc = AgentMaxIterationsError(
            "Max iterations reached",
            agent_type="react",
            max_iterations=10,
            iterations_completed=10
        )
        
        assert exc.max_iterations == 10
        assert exc.iterations_completed == 10
    
    def test_tool_execution_error(self):
        """Tool execution hatası."""
        exc = ToolExecutionError(
            "Tool failed",
            tool_name="web_search"
        )
        
        assert exc.tool_name == "web_search"


class TestAPIExceptions:
    """API exception testleri."""
    
    def test_api_exception(self):
        """Genel API exception."""
        exc = APIException(
            "API error",
            status_code=500
        )
        
        assert exc.status_code == 500
    
    def test_validation_error(self):
        """Validation hatası."""
        exc = ValidationError(
            "Validation failed",
            field="email",
            value="invalid-email"
        )
        
        assert exc.field == "email"
        assert exc.status_code == 422
    
    def test_authentication_error(self):
        """Authentication hatası."""
        exc = AuthenticationError("Invalid credentials")
        
        assert exc.status_code == 401
    
    def test_authorization_error(self):
        """Authorization hatası."""
        exc = AuthorizationError("Access denied")
        
        assert exc.status_code == 403
    
    def test_rate_limit_error(self):
        """Rate limit hatası."""
        exc = RateLimitError(
            "Rate limited",
            retry_after=60
        )
        
        assert exc.status_code == 429
        assert exc.retry_after == 60


class TestSessionExceptions:
    """Session exception testleri."""
    
    def test_session_exception(self):
        """Genel session exception."""
        exc = SessionException(
            "Session error",
            session_id="sess123"
        )
        
        assert exc.session_id == "sess123"
    
    def test_session_not_found(self):
        """Session bulunamadı hatası."""
        exc = SessionNotFoundError(
            "Session not found",
            session_id="invalid_session"
        )
        
        assert exc.session_id == "invalid_session"
    
    def test_session_expired(self):
        """Session expired hatası."""
        exc = SessionExpiredError(
            "Session expired",
            session_id="old_session"
        )
        
        assert exc.session_id == "old_session"


class TestOtherExceptions:
    """Diğer exception testleri."""
    
    def test_configuration_error(self):
        """Konfigürasyon hatası."""
        exc = ConfigurationError(
            "Invalid config",
            config_key="OLLAMA_URL"
        )
        
        assert exc.config_key == "OLLAMA_URL"
    
    def test_external_service_error(self):
        """External service hatası."""
        exc = ExternalServiceError(
            "Service unavailable",
            service_name="ChromaDB"
        )
        
        assert exc.service_name == "ChromaDB"
        assert exc.retryable is True
    
    def test_circuit_breaker_open_error(self):
        """Circuit breaker open hatası."""
        exc = CircuitBreakerOpenError(
            "Circuit open",
            circuit_name="ollama"
        )
        
        assert exc.circuit_name == "ollama"
        assert exc.retryable is True


class TestExceptionInheritance:
    """Exception inheritance testleri."""
    
    def test_all_inherit_from_base(self):
        """Tüm exceptionlar BaseAppException'dan türemeli."""
        exceptions = [
            LLMException("test"),
            RAGException("test"),
            AgentException("test"),
            APIException("test"),
            SessionException("test"),
            ConfigurationError("test"),
            ExternalServiceError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, BaseAppException)
    
    def test_all_inherit_from_exception(self):
        """Tüm exceptionlar Exception'dan türemeli."""
        exceptions = [
            BaseAppException("test"),
            LLMException("test"),
            RAGException("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, Exception)


class TestHandleException:
    """handle_exception utility testleri."""
    
    def test_handle_known_exception(self):
        """Bilinen exception işleme."""
        exc = LLMConnectionError("Connection failed", model="test")
        
        with patch("logging.Logger.error") as mock_log:
            result = handle_exception(exc)
        
        assert result == exc.to_dict()
    
    def test_handle_unknown_exception(self):
        """Bilinmeyen exception işleme."""
        exc = RuntimeError("Unknown error")
        
        result = handle_exception(exc)
        
        assert "error_type" in result
        assert result["message"] == "Unknown error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
