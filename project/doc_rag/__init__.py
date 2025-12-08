# doc_rag package
from common.logger import setup_logger, get_logger

# Initialize root logger for the package
logger = setup_logger("doc_rag")

__all__ = ["logger", "setup_logger", "get_logger"]
