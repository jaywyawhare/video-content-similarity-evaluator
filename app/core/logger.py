import logging
import json
from datetime import datetime
from typing import Any, Dict


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
        }

        # Add extra fields from record
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add any extra attributes
        if record.__dict__.get("extras"):
            log_obj.update(record.__dict__["extras"])

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

        # Add a method to log with extra fields
        def log_with_context(
            self, level: int, msg: str, request_id: str = None, **kwargs
        ):
            """Log with request context"""
            if request_id:
                with RequestLogging(self, request_id):
                    self.log(level, msg, **kwargs)
            else:
                self.log(level, msg, **kwargs)

        logger.log_with_context = lambda *args, **kwargs: log_with_context(
            logger, *args, **kwargs
        )

    return logger
