# -*- coding: utf-8 -*-
"""Reproducibility utilities."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value shared across all generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """Return a NumPy RandomState for reproducible sampling.

    Parameters
    ----------
    seed : int or None
        Seed for the RandomState. If None, a random seed is generated.

    Returns
    -------
    np.random.RandomState
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    return np.random.RandomState(seed)
