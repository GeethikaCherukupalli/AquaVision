# -*- coding: utf-8 -*-
"""Input validation utilities."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def assert_shapes(
    image: np.ndarray,
    expected_shape: Tuple[int, ...],
    name: str = "array",
) -> None:
    """Assert that an array has an expected shape.

    Parameters
    ----------
    image : np.ndarray
        Array to validate.
    expected_shape : tuple
        Expected shape. Use -1 as a wildcard for any dimension.
    name : str
        Name used in error messages.

    Raises
    ------
    ValueError
        If shapes do not match.
    """
    if len(image.shape) != len(expected_shape):
        raise ValueError(
            f"{name} has shape {image.shape}, expected {expected_shape}"
        )
    for actual, expected in zip(image.shape, expected_shape):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} has shape {image.shape}, expected {expected_shape}"
            )


def validate_image(
    image: np.ndarray,
    expected_channels: int = 2,
    name: str = "image",
) -> None:
    """Validate a SAR image array.

    Parameters
    ----------
    image : np.ndarray
        Image array to validate.
    expected_channels : int
        Expected number of channels.
    name : str
        Name for error messages.
    """
    if image.ndim != 3:
        raise ValueError(
            f"{name} must be 3D (C, H, W), got shape {image.shape}"
        )
    if image.shape[0] != expected_channels:
        raise ValueError(
            f"{name} has {image.shape[0]} channels, expected {expected_channels}"
        )
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(
            f"{name} must be float dtype, got {image.dtype}"
        )
    has_nan = np.isnan(image).any()
    has_inf = np.isinf(image).any()
    if has_nan:
        raise ValueError(f"{name} contains NaN values")
    if has_inf:
        raise ValueError(f"{name} contains Inf values")


def validate_mask(
    mask: np.ndarray,
    expected_size: int = 2048,
    name: str = "mask",
) -> None:
    """Validate a binary segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Mask array to validate.
    expected_size : int
        Expected spatial dimension (height or width).
    name : str
        Name for error messages.
    """
    if mask.ndim != 2:
        raise ValueError(
            f"{name} must be 2D (H, W), got shape {mask.shape}"
        )
    if mask.shape[0] != expected_size or mask.shape[1] != expected_size:
        raise ValueError(
            f"{name} expected {expected_size}×{expected_size}, got {mask.shape}"
        )
    unique = np.unique(mask)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"{name} must be binary [0, 1], got unique values {unique}"
        )
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(
            f"{name} must be integer dtype, got {mask.dtype}"
        )
