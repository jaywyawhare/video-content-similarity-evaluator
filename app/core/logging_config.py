import logging.config
import os
import sys
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
            "delay": True,  # Delay file creation until first log
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": ERROR_LOG,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf-8",
            "level": "ERROR",
            "delay": True,
        },
        "performance_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "json",
            "filename": os.path.join(LOG_DIR, "performance.log"),
            "when": "H",  # Rotate hourly
            "interval": 1,
            "backupCount": 48,  # Keep 2 days worth
            "encoding": "utf-8",
            "delay": True,
        },
    },
    "loggers": {
        "video_rating_service": {
            "handlers": ["console", "file", "error_file", "performance_file"],
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
        os.chmod(LOG_DIR, 0o755)

        # Create log files with proper permissions
        for log_file in LOG_FILES + [os.path.join(LOG_DIR, "performance.log")]:
            if not os.path.exists(log_file):
                with open(log_file, "a") as f:
                    os.chmod(log_file, 0o644)

        logging.config.dictConfig(LOGGING_CONFIG)

        # Get the main service logger
        logger = logging.getLogger("video_rating_service")

        # Write initial messages to all log files to verify they're working
        logger.info("Application logging initialized - Info level test message")
        logger.error("Application logging initialized - Error level test message")
        logger.debug("Application logging initialized - Debug level test message")

        # Write an initial performance log
        perf_logger = logging.getLogger("video_rating_service")
        perf_logger.info(
            "Performance logging initialized",
            extra={"operation": "logging_setup", "duration": 0.0, "success": True},
        )

        # Verify all files were created
        missing_files = [f for f in LOG_FILES if not os.path.exists(f)]
        if missing_files:
            raise RuntimeError(f"Failed to create log files: {missing_files}")

        logger.info(
            "Logging system initialized successfully",
            extra={
                "log_dir": LOG_DIR,
                "log_files": [f for f in LOG_FILES if os.path.exists(f)],
                "config": {
                    k: v for k, v in LOGGING_CONFIG.items() if k != "formatters"
                },
            },
        )

    except Exception as e:
        print(f"Error setting up logging: {e}", file=sys.stderr)
        raise
