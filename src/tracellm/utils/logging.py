"""Structured logging for TraceLLM."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)
_initialized = False


def setup_logging(level: str = "info", log_file: str | None = None) -> logging.Logger:
    """Configure the root tracellm logger with rich console + optional file output."""
    global _initialized
    if _initialized:
        return logging.getLogger("tracellm")

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("tracellm")
    logger.setLevel(log_level)
    logger.propagate = False

    # Rich console handler
    rich_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(log_level)
    logger.addHandler(rich_handler)

    # File handler
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
        )
        logger.addHandler(file_handler)

    _initialized = True
    return logger


def get_logger(name: str = "tracellm") -> logging.Logger:
    """Get a child logger under the tracellm namespace."""
    return logging.getLogger(name)
