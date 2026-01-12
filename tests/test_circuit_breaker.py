"""
Enterprise AI Assistant - Unit Tests for Circuit Breaker
=========================================================

Circuit breaker pattern için kapsamlı birim testleri.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

# Import the circuit breaker
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path

# Test imports
try:
    from core.circuit_breaker import (
        CircuitBreaker,
        CircuitState,
        CircuitStats,
        CircuitBreakerRegistry,
        CircuitBreakerOpenError,
    )
except ImportError:
    # Create mock classes if import fails
    pass


class TestCircuitBreaker:
    """CircuitBreaker sınıfı testleri."""
    
    def test_initial_state_is_closed(self):
        """Başlangıç durumu CLOSED olmalı."""
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
    
    def test_records_success(self):
        """Başarılı çağrılar kaydedilmeli."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        @cb
        def success_func():
            return "success"
        
        result = success_func()
        
        assert result == "success"
        assert cb._success_count == 1
        assert cb._failure_count == 0
    
    def test_records_failure(self):
        """Başarısız çağrılar kaydedilmeli."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        @cb
        def fail_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            fail_func()
        
        assert cb._failure_count == 1
        assert cb.state == CircuitState.CLOSED
    
    def test_opens_after_threshold(self):
        """Eşik aşıldığında OPEN olmalı."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        @cb
        def fail_func():
            raise ValueError("Test error")
        
        # Trigger failures
        for _ in range(3):
            try:
                fail_func()
            except ValueError:
                pass
        
        assert cb.state == CircuitState.OPEN
    
    def test_rejects_when_open(self):
        """OPEN durumunda çağrılar reddedilmeli."""
        cb = CircuitBreaker("test", failure_threshold=1)
        
        @cb
        def fail_func():
            raise ValueError("Test error")
        
        @cb
        def success_func():
            return "success"
        
        # Open the circuit
        try:
            fail_func()
        except ValueError:
            pass
        
        assert cb.state == CircuitState.OPEN
        
        # Should raise CircuitBreakerOpenError
        with pytest.raises(Exception) as exc_info:
            success_func()
        
        assert "open" in str(exc_info.value).lower() or "circuit" in str(exc_info.value).lower()
    
    def test_half_open_after_timeout(self):
        """Timeout sonrası HALF_OPEN olmalı."""
        cb = CircuitBreaker("test", failure_threshold=1, timeout_seconds=0.1)
        
        @cb
        def fail_func():
            raise ValueError("Test error")
        
        # Open the circuit
        try:
            fail_func()
        except ValueError:
            pass
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_closes_on_success_in_half_open(self):
        """HALF_OPEN'da başarı sonrası CLOSED olmalı."""
        cb = CircuitBreaker(
            "test", 
            failure_threshold=1, 
            success_threshold=1,
            timeout_seconds=0.1
        )
        
        call_count = 0
        
        @cb
        def sometimes_fail():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return "success"
        
        # First call fails, opens circuit
        try:
            sometimes_fail()
        except ValueError:
            pass
        
        # Wait for half-open
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second call succeeds
        result = sometimes_fail()
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
    
    def test_reopens_on_failure_in_half_open(self):
        """HALF_OPEN'da hata sonrası tekrar OPEN olmalı."""
        cb = CircuitBreaker("test", failure_threshold=1, timeout_seconds=0.1)
        
        @cb
        def always_fail():
            raise ValueError("Always fails")
        
        # First failure opens circuit
        try:
            always_fail()
        except ValueError:
            pass
        
        # Wait for half-open
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Failure in half-open reopens
        try:
            always_fail()
        except ValueError:
            pass
        
        assert cb.state == CircuitState.OPEN
    
    def test_manual_reset(self):
        """Manuel reset çalışmalı."""
        cb = CircuitBreaker("test", failure_threshold=1)
        
        @cb
        def fail_func():
            raise ValueError("Test error")
        
        try:
            fail_func()
        except ValueError:
            pass
        
        assert cb.state == CircuitState.OPEN
        
        cb.reset()
        
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
    
    def test_get_stats(self):
        """İstatistikler doğru döndürülmeli."""
        cb = CircuitBreaker("test_stats", failure_threshold=5)
        
        @cb
        def mixed_func(should_fail: bool):
            if should_fail:
                raise ValueError("Error")
            return "success"
        
        # Some successes
        for _ in range(3):
            mixed_func(False)
        
        # Some failures
        for _ in range(2):
            try:
                mixed_func(True)
            except ValueError:
                pass
        
        stats = cb.get_stats()
        
        assert stats.name == "test_stats"
        assert stats.state == CircuitState.CLOSED
        assert stats.success_count == 3
        assert stats.failure_count == 2
        assert stats.total_calls == 5


class TestCircuitBreakerAsync:
    """Async circuit breaker testleri."""
    
    @pytest.mark.asyncio
    async def test_async_success(self):
        """Async başarılı çağrı."""
        cb = CircuitBreaker("async_test", failure_threshold=3)
        
        @cb
        async def async_success():
            return "async success"
        
        result = await async_success()
        assert result == "async success"
    
    @pytest.mark.asyncio
    async def test_async_failure(self):
        """Async başarısız çağrı."""
        cb = CircuitBreaker("async_test", failure_threshold=3)
        
        @cb
        async def async_fail():
            raise ValueError("Async error")
        
        with pytest.raises(ValueError):
            await async_fail()
        
        assert cb._failure_count == 1
    
    @pytest.mark.asyncio
    async def test_async_opens_circuit(self):
        """Async çağrılar circuit'ı açabilmeli."""
        cb = CircuitBreaker("async_test", failure_threshold=2)
        
        @cb
        async def async_fail():
            raise ValueError("Async error")
        
        for _ in range(2):
            try:
                await async_fail()
            except ValueError:
                pass
        
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    """Registry testleri."""
    
    def test_register_and_get(self):
        """Circuit kayıt ve alma."""
        registry = CircuitBreakerRegistry()
        cb = CircuitBreaker("registry_test", failure_threshold=3)
        
        registry.register(cb)
        retrieved = registry.get("registry_test")
        
        assert retrieved is cb
    
    def test_get_or_create(self):
        """Var olan veya yeni oluşturma."""
        registry = CircuitBreakerRegistry()
        
        cb1 = registry.get_or_create("new_circuit", failure_threshold=5)
        cb2 = registry.get_or_create("new_circuit", failure_threshold=10)
        
        assert cb1 is cb2
        assert cb1.failure_threshold == 5
    
    def test_get_all_stats(self):
        """Tüm istatistikleri alma."""
        registry = CircuitBreakerRegistry()
        
        cb1 = registry.get_or_create("circuit1", failure_threshold=3)
        cb2 = registry.get_or_create("circuit2", failure_threshold=3)
        
        stats = registry.get_all_stats()
        
        assert len(stats) >= 2
        names = [s.name for s in stats]
        assert "circuit1" in names
        assert "circuit2" in names
    
    def test_reset_all(self):
        """Tüm circuit'ları sıfırlama."""
        registry = CircuitBreakerRegistry()
        
        cb1 = registry.get_or_create("reset1", failure_threshold=1)
        cb2 = registry.get_or_create("reset2", failure_threshold=1)
        
        # Open both circuits
        @cb1
        def fail1():
            raise ValueError()
        
        @cb2
        def fail2():
            raise ValueError()
        
        try:
            fail1()
        except:
            pass
        
        try:
            fail2()
        except:
            pass
        
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN
        
        registry.reset_all()
        
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED


class TestCircuitBreakerThreadSafety:
    """Thread safety testleri."""
    
    def test_concurrent_calls(self):
        """Eş zamanlı çağrılar thread-safe olmalı."""
        cb = CircuitBreaker("concurrent_test", failure_threshold=100)
        
        call_count = 0
        lock = __import__("threading").Lock()
        
        @cb
        def increment():
            nonlocal call_count
            with lock:
                call_count += 1
            return call_count
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(increment) for _ in range(50)]
            for future in futures:
                future.result()
        
        assert call_count == 50
        assert cb._success_count == 50
    
    def test_concurrent_failures(self):
        """Eş zamanlı hatalar doğru sayılmalı."""
        cb = CircuitBreaker("concurrent_fail_test", failure_threshold=100)
        
        @cb
        def fail():
            raise ValueError("Error")
        
        def call_and_catch():
            try:
                fail()
            except ValueError:
                pass
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(call_and_catch) for _ in range(30)]
            for future in futures:
                future.result()
        
        assert cb._failure_count == 30


class TestExcludedExceptions:
    """Excluded exceptions testleri."""
    
    def test_excluded_exception_not_counted(self):
        """Excluded exception sayılmamalı."""
        cb = CircuitBreaker(
            "excluded_test",
            failure_threshold=2,
            excluded_exceptions=(ValueError,)
        )
        
        @cb
        def raise_value_error():
            raise ValueError("Excluded error")
        
        for _ in range(5):
            try:
                raise_value_error()
            except ValueError:
                pass
        
        # Should not open because ValueError is excluded
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
    
    def test_non_excluded_exception_counted(self):
        """Excluded olmayan exception sayılmalı."""
        cb = CircuitBreaker(
            "non_excluded_test",
            failure_threshold=2,
            excluded_exceptions=(ValueError,)
        )
        
        @cb
        def raise_type_error():
            raise TypeError("Not excluded error")
        
        for _ in range(2):
            try:
                raise_type_error()
            except TypeError:
                pass
        
        assert cb.state == CircuitState.OPEN
        assert cb._failure_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
