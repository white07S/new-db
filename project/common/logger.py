"""Common logging utilities for the project.

Provides JSON-formatted structured logging with support for extra properties.
"""

import json
import logging
import sys
import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        # If there are extra fields in the record (passed via extra={...}), add them
        if hasattr(record, "props"):
            log_entry.update(record.props)

        return json.dumps(log_entry)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup and return a logger with JSON formatting.
    
    Args:
        name: The name of the logger
        level: The logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplication
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name.
    
    Args:
        name: The name of the logger to retrieve
        
    Returns:
        The logger instance
    """
    return logging.getLogger(name)
