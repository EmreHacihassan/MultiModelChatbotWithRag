"""
Enterprise AI Assistant - Structured Logging System
====================================================

Production-grade JSON logging with:
- Correlation ID tracking
- Request context propagation
- Log levels and formatting
- Log rotation
- Performance metrics
"""

import logging
import logging.handlers
import sys
import json
import traceback
import threading
import contextvars
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import wraps
import time


# Context variables for request tracking
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)
user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "user_id", default=""
)
session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default=""
)
request_path: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_path", default=""
)


def generate_correlation_id() -> str:
    """Benzersiz correlation ID oluştur."""
    return str(uuid.uuid4())[:8]


def set_correlation_id(cid: str) -> None:
    """Correlation ID ayarla."""
    correlation_id.set(cid)


def get_correlation_id() -> str:
    """Mevcut correlation ID'yi al."""
    cid = correlation_id.get()
    if not cid:
        cid = generate_correlation_id()
        correlation_id.set(cid)
    return cid


class JSONFormatter(logging.Formatter):
    """
    JSON format logging formatter.
    
    Her log entry'si yapılandırılmış JSON olarak yazılır.
    """
    
    RESERVED_ATTRS = frozenset([
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName"
    ])
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_name: bool = True,
        include_path: bool = False,
        include_thread: bool = False,
        include_process: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_name = include_name
        self.include_path = include_path
        self.include_thread = include_thread
        self.include_process = include_process
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {}
        
        # Timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Level
        if self.include_level:
            log_data["level"] = record.levelname
        
        # Logger name
        if self.include_name:
            log_data["logger"] = record.name
        
        # Message
        log_data["message"] = record.getMessage()
        
        # Correlation ID
        cid = correlation_id.get()
        if cid:
            log_data["correlation_id"] = cid
        
        # User and session context
        uid = user_id.get()
        if uid:
            log_data["user_id"] = uid
        
        sid = session_id.get()
        if sid:
            log_data["session_id"] = sid
        
        # Request path
        rpath = request_path.get()
        if rpath:
            log_data["request_path"] = rpath
        
        # File path and line
        if self.include_path:
            log_data["path"] = record.pathname
            log_data["line"] = record.lineno
            log_data["function"] = record.funcName
        
        # Thread info
        if self.include_thread:
            log_data["thread"] = record.threadName
            log_data["thread_id"] = record.thread
        
        # Process info
        if self.include_process:
            log_data["process"] = record.processName
            log_data["process_id"] = record.process
        
        # Exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None
            }
        
        # Extra fields from record
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        # Static extra fields
        log_data.update(self.extra_fields)
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """
    Terminal için renkli log formatter.
    Development ortamı için.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"
    
    def __init__(self, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    
    def format(self, record: logging.LogRecord) -> str:
        # Add correlation ID to message if present
        cid = correlation_id.get()
        if cid:
            record.msg = f"[{cid}] {record.msg}"
        
        # Add color
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


class StructuredLogger:
    """
    Yapılandırılmış logger wrapper.
    
    Extra context ile loglama için kolaylık sağlar.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def _log(
        self,
        level: int,
        message: str,
        **kwargs
    ) -> None:
        """Internal log method with extra fields."""
        extra = kwargs.pop("extra", {})
        extra.update(kwargs)
        self._logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """Debug level log."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Info level log."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Warning level log."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Error level log."""
        self._logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Critical level log."""
        self._logger.critical(message, exc_info=exc_info, extra=kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Exception log with traceback."""
        self._logger.exception(message, extra=kwargs)
    
    # Convenience methods
    def api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ) -> None:
        """API request logging."""
        self.info(
            f"{method} {path} - {status_code}",
            event_type="api_request",
            http_method=method,
            http_path=path,
            http_status=status_code,
            duration_ms=round(duration_ms, 2),
            **kwargs
        )
    
    def llm_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
        success: bool = True,
        **kwargs
    ) -> None:
        """LLM request logging."""
        self.info(
            f"LLM request to {model}",
            event_type="llm_request",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            duration_ms=round(duration_ms, 2),
            success=success,
            **kwargs
        )
    
    def rag_query(
        self,
        query: str,
        results_count: int,
        duration_ms: float,
        **kwargs
    ) -> None:
        """RAG query logging."""
        self.info(
            f"RAG query executed",
            event_type="rag_query",
            query_length=len(query),
            results_count=results_count,
            duration_ms=round(duration_ms, 2),
            **kwargs
        )
    
    def agent_action(
        self,
        agent_type: str,
        action: str,
        success: bool = True,
        **kwargs
    ) -> None:
        """Agent action logging."""
        level = logging.INFO if success else logging.WARNING
        self._log(
            level,
            f"Agent {agent_type}: {action}",
            event_type="agent_action",
            agent_type=agent_type,
            action=action,
            success=success,
            **kwargs
        )
    
    def performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        **kwargs
    ) -> None:
        """Performance metric logging."""
        self.debug(
            f"Performance: {metric_name}={value}{unit}",
            event_type="performance_metric",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            **kwargs
        )
    
    def security_event(
        self,
        event_type: str,
        message: str,
        severity: str = "medium",
        **kwargs
    ) -> None:
        """Security event logging."""
        level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self._log(
            level,
            f"Security: {message}",
            event_type=f"security_{event_type}",
            security_severity=severity,
            **kwargs
        )


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[Union[str, Path]] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    include_console: bool = True,
    app_name: str = "enterprise-ai-assistant"
) -> None:
    """
    Logging sistemini yapılandır.
    
    Args:
        log_level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format tipi ("json" veya "text")
        log_file: Log dosyası yolu (opsiyonel)
        max_bytes: Log rotation için max dosya boyutu
        backup_count: Tutulacak backup dosya sayısı
        include_console: Console'a da yazsın mı
        app_name: Uygulama adı (JSON loglarında görünür)
    """
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Mevcut handler'ları temizle
    root_logger.handlers.clear()
    
    # Formatter seç
    if log_format == "json":
        formatter = JSONFormatter(
            include_path=True,
            include_thread=True,
            extra_fields={"app": app_name}
        )
    else:
        formatter = ColoredFormatter()
    
    # Console handler
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Console için her zaman colored formatter (development'ta)
        if log_format != "json":
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(formatter)
        
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Diğer loggerları sustur
    for logger_name in ["urllib3", "httpx", "chromadb", "sentence_transformers"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """
    Yapılandırılmış logger al.
    
    Args:
        name: Logger adı (genellikle __name__)
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(logging.getLogger(name))


def log_execution_time(
    logger: Optional[StructuredLogger] = None,
    operation: str = "operation",
    log_level: str = "debug"
):
    """
    Decorator: Fonksiyon çalışma süresini logla.
    
    Args:
        logger: Logger instance (None ise yeni oluşturulur)
        operation: İşlem adı
        log_level: Log seviyesi
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                
                log_method = getattr(logger, log_level, logger.debug)
                log_method(
                    f"{operation} completed",
                    function=func.__name__,
                    duration_ms=round(elapsed, 2),
                    success=True
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{operation} failed",
                    function=func.__name__,
                    duration_ms=round(elapsed, 2),
                    success=False,
                    error=str(e)
                )
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                
                log_method = getattr(logger, log_level, logger.debug)
                log_method(
                    f"{operation} completed",
                    function=func.__name__,
                    duration_ms=round(elapsed, 2),
                    success=True
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{operation} failed",
                    function=func.__name__,
                    duration_ms=round(elapsed, 2),
                    success=False,
                    error=str(e)
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


class LogContext:
    """
    Context manager for temporary log context.
    
    with LogContext(user_id="user123", session_id="sess456"):
        logger.info("This log will have user_id and session_id")
    """
    
    def __init__(
        self,
        correlation_id_val: Optional[str] = None,
        user_id_val: Optional[str] = None,
        session_id_val: Optional[str] = None,
        request_path_val: Optional[str] = None
    ):
        self.correlation_id_val = correlation_id_val
        self.user_id_val = user_id_val
        self.session_id_val = session_id_val
        self.request_path_val = request_path_val
        
        self._old_correlation_id = None
        self._old_user_id = None
        self._old_session_id = None
        self._old_request_path = None
    
    def __enter__(self):
        if self.correlation_id_val:
            self._old_correlation_id = correlation_id.get()
            correlation_id.set(self.correlation_id_val)
        
        if self.user_id_val:
            self._old_user_id = user_id.get()
            user_id.set(self.user_id_val)
        
        if self.session_id_val:
            self._old_session_id = session_id.get()
            session_id.set(self.session_id_val)
        
        if self.request_path_val:
            self._old_request_path = request_path.get()
            request_path.set(self.request_path_val)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old_correlation_id is not None:
            correlation_id.set(self._old_correlation_id)
        
        if self._old_user_id is not None:
            user_id.set(self._old_user_id)
        
        if self._old_session_id is not None:
            session_id.set(self._old_session_id)
        
        if self._old_request_path is not None:
            request_path.set(self._old_request_path)
        
        return False


__all__ = [
    "setup_logging",
    "get_logger",
    "StructuredLogger",
    "JSONFormatter",
    "ColoredFormatter",
    "log_execution_time",
    "LogContext",
    "generate_correlation_id",
    "set_correlation_id",
    "get_correlation_id",
    "correlation_id",
    "user_id",
    "session_id",
    "request_path",
]
