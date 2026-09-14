# -*- coding: utf-8 -*-
"""Structured logging utility."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "aquavision",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up a structured JSON-style logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.
    log_file : str or None
        Optional path to a log file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "aquavision") -> logging.Logger:
    """Return an existing logger or create a new one."""
    return logging.getLogger(name)
