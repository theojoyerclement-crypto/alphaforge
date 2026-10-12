"""Logging configuration.

All library modules log through children of the ``alphaforge`` logger, so a single call to
:func:`configure_logging` controls verbosity and (optionally) mirrors output to a run log file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = "alphaforge"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str | int = "INFO", log_file: str | Path | None = None
) -> logging.Logger:
    """Configure the ``alphaforge`` logger hierarchy.

    Args:
        level: Logging level name or number.
        log_file: Optional path; when given, logs are also written there (UTF-8).

    Returns:
        The configured root ``alphaforge`` logger.
    """
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, _DATEFMT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``alphaforge`` (pass ``__name__``)."""
    if name != _ROOT and not name.startswith(f"{_ROOT}."):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)
