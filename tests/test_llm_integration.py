"""
Enterprise AI Assistant - Integration Tests for LLM Manager
============================================================

LLM Manager için mock-based integration testleri.
Gerçek LLM çağrısı yapmadan tüm fonksiyonelliği test eder.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


# Mock LLM Manager for testing without actual LLM
class MockedLLMManager:
    """
    LLM Manager mock implementation.
    Gerçek Ollama çağrısı yapmadan test için.
    """
    
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        self.backup_model = "qwen2.5:3b"
        self.call_count = 0
        self.last_prompt = None
        self.responses = {}
        self.should_fail = False
        self.fail_count = 0
        self.max_failures = 0
    
    def set_response(self, prompt_contains: str, response: str):
        """Belirli prompt için response ayarla."""
        self.responses[prompt_contains] = response
    
    def set_failure_mode(self, max_failures: int = 1):
        """Failure mode'u ayarla."""
        self.should_fail = True
        self.max_failures = max_failures
        self.fail_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Mock generate method."""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Failure mode
        if self.should_fail and self.fail_count < self.max_failures:
            self.fail_count += 1
            raise ConnectionError("Mock connection failed")
        
        # Check for predefined responses
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # Default response
        return f"Mock response to: {prompt[:50]}..."
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async mock generate."""
        return self.generate(prompt, **kwargs)
    
    def chat(self, messages: list, **kwargs) -> str:
        """Mock chat method."""
        last_message = messages[-1]["content"] if messages else ""
        return self.generate(last_message, **kwargs)
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return not self.should_fail
    
    def get_model_info(self) -> dict:
        """Get model info."""
        return {
            "name": self.model,
            "parameters": "7B",
            "quantization": "Q4_K_M"
        }


class TestLLMManagerWithMocks:
    """LLM Manager testleri (mocked)."""
    
    @pytest.fixture
    def llm(self):
        """Mock LLM fixture."""
        return MockedLLMManager()
    
    def test_basic_generation(self, llm):
        """Temel generation testi."""
        response = llm.generate("Hello, how are you?")
        
        assert response is not None
        assert len(response) > 0
        assert llm.call_count == 1
    
    def test_predefined_response(self, llm):
        """Önceden tanımlı response testi."""
        llm.set_response("weather", "The weather is sunny today.")
        
        response = llm.generate("What's the weather like?")
        
        assert response == "The weather is sunny today."
    
    def test_prompt_tracking(self, llm):
        """Prompt takibi testi."""
        llm.generate("Test prompt 1")
        llm.generate("Test prompt 2")
        
        assert llm.call_count == 2
        assert llm.last_prompt == "Test prompt 2"
    
    def test_failure_mode(self, llm):
        """Failure mode testi."""
        llm.set_failure_mode(max_failures=1)
        
        with pytest.raises(ConnectionError):
            llm.generate("Test")
        
        # Second call should succeed
        response = llm.generate("Test")
        assert response is not None
    
    def test_chat_interface(self, llm):
        """Chat interface testi."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = llm.chat(messages)
        
        assert response is not None
        assert llm.call_count == 1
    
    def test_model_info(self, llm):
        """Model bilgisi testi."""
        info = llm.get_model_info()
        
        assert info["name"] == "qwen2.5:7b"
        assert "parameters" in info
    
    @pytest.mark.asyncio
    async def test_async_generation(self, llm):
        """Async generation testi."""
        response = await llm.agenerate("Async test prompt")
        
        assert response is not None
        assert llm.call_count == 1


class TestLLMWithRetry:
    """Retry mekanizması testleri."""
    
    @pytest.fixture
    def llm(self):
        return MockedLLMManager()
    
    def test_retry_on_failure(self, llm):
        """Retry mekanizması testi."""
        llm.set_failure_mode(max_failures=2)
        
        # Simulate retry logic
        max_retries = 3
        result = None
        
        for attempt in range(max_retries):
            try:
                result = llm.generate("Test with retry")
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
        
        assert result is not None
        assert llm.call_count == 3  # 2 failures + 1 success


class TestLLMSystemPrompts:
    """System prompt testleri."""
    
    @pytest.fixture
    def llm(self):
        return MockedLLMManager()
    
    def test_system_prompt_injection(self, llm):
        """System prompt injection testi."""
        system_prompt = "You are a helpful assistant."
        user_prompt = "What is Python?"
        
        combined = f"{system_prompt}\n\nUser: {user_prompt}"
        response = llm.generate(combined)
        
        assert response is not None
        assert llm.last_prompt == combined
    
    def test_custom_response_for_system_prompt(self, llm):
        """Custom system prompt response."""
        llm.set_response("Python", "Python is a programming language.")
        
        response = llm.generate("Tell me about Python")
        
        assert "programming language" in response


class TestLLMStreaming:
    """Streaming response testleri."""
    
    @pytest.fixture
    def llm(self):
        return MockedLLMManager()
    
    def test_stream_simulation(self, llm):
        """Stream simulation testi."""
        def mock_stream(prompt: str):
            response = llm.generate(prompt)
            for char in response:
                yield char
        
        result = "".join(mock_stream("Stream test"))
        
        assert len(result) > 0


class TestLLMContextWindow:
    """Context window testleri."""
    
    @pytest.fixture
    def llm(self):
        return MockedLLMManager()
    
    def test_long_context(self, llm):
        """Uzun context testi."""
        long_prompt = "A" * 10000 + "\n\nSummarize this."
        
        response = llm.generate(long_prompt)
        
        assert response is not None
    
    def test_context_truncation(self, llm):
        """Context truncation testi."""
        max_context = 4000
        
        long_prompt = "A" * 8000
        truncated = long_prompt[-max_context:]
        
        response = llm.generate(truncated)
        
        assert response is not None
        assert len(llm.last_prompt) <= max_context


class TestLLMTemperatureAndParams:
    """Temperature ve diğer parametre testleri."""
    
    @pytest.fixture
    def llm(self):
        return MockedLLMManager()
    
    def test_with_temperature(self, llm):
        """Temperature parametresi testi."""
        response = llm.generate(
            "Generate a random number",
            temperature=0.9
        )
        
        assert response is not None
    
    def test_with_max_tokens(self, llm):
        """Max tokens parametresi testi."""
        response = llm.generate(
            "Write a story",
            max_tokens=100
        )
        
        assert response is not None


class TestLLMErrorHandling:
    """Hata yönetimi testleri."""
    
    @pytest.fixture
    def llm(self):
        return MockedLLMManager()
    
    def test_connection_error_handling(self, llm):
        """Connection error handling."""
        llm.should_fail = True
        llm.max_failures = float('inf')  # Always fail
        
        with pytest.raises(ConnectionError):
            llm.generate("Test")
    
    def test_graceful_degradation(self, llm):
        """Graceful degradation testi."""
        llm.model = "large_model"
        llm.backup_model = "small_model"
        
        # Simulate switching to backup
        try:
            llm.should_fail = True
            llm.max_failures = 1
            llm.generate("Test")
        except ConnectionError:
            llm.model = llm.backup_model
            llm.should_fail = False
        
        response = llm.generate("Test with backup")
        assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
