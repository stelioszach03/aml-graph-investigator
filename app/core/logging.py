from __future__ import annotations

import os
import sys
from typing import Optional

from loguru import logger as _logger


def configure_json_logger() -> None:
    """Configure Loguru to emit structured JSON logs to stdout.

    Log level is taken from environment variables in order of precedence:
    - LOG_LEVEL
    - APP_LOG_LEVEL
    Defaults to INFO if not set.
    """
    level: str = (os.getenv("LOG_LEVEL") or os.getenv("APP_LOG_LEVEL") or "INFO").upper()

    # Reset existing handlers and add JSON sink
    _logger.remove()
    _logger.add(
        sys.stdout,
        level=level,
        serialize=True,  # JSON output
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )


def get_logger(name: str):
    """Return a contextual logger bound with a component name.

    Example usage:
        log = get_logger("scripts.ingest")
        log.info("message with {}", "args")
    """
    return _logger.bind(component=name)
