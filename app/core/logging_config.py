import logging.config
import os
from typing import Dict, Any

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Define log file paths
APP_LOG = os.path.join(LOG_DIR, "app.log")
ERROR_LOG = os.path.join(LOG_DIR, "error.log")
DEBUG_LOG = os.path.join(LOG_DIR, "debug.log")
ACCESS_LOG = os.path.join(LOG_DIR, "access.log")

# Define log files to create
LOG_FILES = [APP_LOG, ERROR_LOG, DEBUG_LOG, ACCESS_LOG]

LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "app.core.logger.JsonFormatter",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "json",
            "filename": APP_LOG,
            "when": "midnight",
            "interval": 1,
            "backupCount": 30,
            "encoding": "utf-8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": ERROR_LOG,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf-8",
            "level": "ERROR",
        },
        "debug_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": DEBUG_LOG,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
            "level": "DEBUG",
        },
        "access_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "json",
            "filename": ACCESS_LOG,
            "when": "midnight",
            "interval": 1,
            "backupCount": 30,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "video_rating_service": {
            "handlers": ["console", "file", "error_file", "debug_file", "access_file"],
            "level": "INFO",
            "propagate": False,
        },
        "s3_service": {
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": False,
        },
        "audio_service": {
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": False,
        },
        "transcription_service": {
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": False,
        },
        "rating_service": {
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def setup_logging():
    """Initialize logging configuration"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        # Set log directory permissions
        os.chmod(LOG_DIR, 0o755)  # rwxr-xr-x

        # Create log files with proper permissions
        for log_file in LOG_FILES:
            if not os.path.exists(log_file):
                with open(log_file, "a") as f:
                    os.chmod(log_file, 0o644)  # rw-r--r--

        logging.config.dictConfig(LOGGING_CONFIG)

    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise
