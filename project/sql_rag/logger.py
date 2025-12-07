
import json
import logging
import sys
import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
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

def setup_logger(name: str = "sql_rag", level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplication
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def get_logger(name: str):
    return logging.getLogger(name)
