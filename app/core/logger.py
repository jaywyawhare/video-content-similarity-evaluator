import logging
import json
from datetime import datetime
from typing import Any, Dict
import time
import traceback


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with extra fields"""
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }

        # Add execution time if available
        if hasattr(record, "duration"):
            log_obj["duration_ms"] = round(record.duration * 1000, 2)

        # Add request context
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "endpoint"):
            log_obj["endpoint"] = record.endpoint
        if hasattr(record, "method"):
            log_obj["method"] = record.method

        # Add exception info with stack trace
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            log_obj["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value),
                "stacktrace": traceback.format_exception(*record.exc_info),
            }

        # Add any extra attributes
        if hasattr(record, "extras"):
            log_obj.update(record.extras)

        return json.dumps(log_obj)


class RequestIdFilter(logging.Filter):
    """Filter that adds request_id to log records"""

    def __init__(self, request_id: str):
        self.request_id = request_id

    def filter(self, record):
        record.request_id = self.request_id
        return True


class RequestLogging:
    """Context manager for request logging"""

    def __init__(self, logger, request_id: str):
        self.logger = logger
        self.request_id = request_id
        self.filter = RequestIdFilter(request_id)

    def __enter__(self):
        self.logger.addFilter(self.filter)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeFilter(self.filter)


class PerformanceLogging:
    """Context manager for tracking execution time"""

    def __init__(self, logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.logger.info(
            f"{self.operation} completed",
            extra={
                "operation": self.operation,
                "duration": duration,
                "success": exc_type is None,
            },
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with JSON formatting"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Add console handler with JSON formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)

        # Set base logging level
        logger.setLevel(logging.INFO)

        def log_with_context(
            self, level: int, msg: str, request_id: str = None, **kwargs
        ):
            """Log with request context"""
            extras = kwargs.pop("extra", {})
            if request_id:
                with RequestLogging(self, request_id):
                    if "duration" in kwargs:
                        extras["duration"] = kwargs.pop("duration")
                    self.log(level, msg, extra=extras, **kwargs)
            else:
                self.log(level, msg, extra=extras, **kwargs)

        logger.log_with_context = lambda *args, **kwargs: log_with_context(
            logger, *args, **kwargs
        )
        logger.perf_track = lambda op: PerformanceLogging(logger, op)

    return logger
