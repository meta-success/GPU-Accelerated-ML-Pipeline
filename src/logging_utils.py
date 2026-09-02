"""Structured logging used by every pipeline stage."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure a single root logger with timestamps and module names."""
    global _CONFIGURED
    numeric = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("gpu_pipeline")
    logger.setLevel(numeric)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    else:
        for handler in logger.handlers:
            handler.setLevel(numeric)
        logger.setLevel(numeric)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    base = logging.getLogger("gpu_pipeline")
    if name:
        return base.getChild(name)
    return base
