import logging
import os
import sys
from typing import Optional


DEFAULT_LOGGER_NAME = "agentkit"


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    logger_name: str = DEFAULT_LOGGER_NAME,
) -> logging.Logger:
    """Configure a shared logger for the agent framework."""
    resolved_level = (level or os.getenv("AGENTKIT_LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file or os.getenv("AGENTKIT_LOG_FILE"):
        file_path = log_file or os.getenv("AGENTKIT_LOG_FILE")
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-scoped logger configured for the agent framework."""
    configure_logging()
    if not name:
        return logging.getLogger(DEFAULT_LOGGER_NAME)
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")
