# -*- coding: utf-8 -*-
"""Channel-wise SAR normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class ChannelNormalizer:
    """Channel-wise mean/std normalizer for VV/VH SAR data.

    Parameters
    ----------
    vv_mean : float or None
        Mean of VV channel in dB.
    vv_std : float or None
        Std of VV channel in dB.
    vh_mean : float or None
        Mean of VH channel in dB.
    vh_std : float or None
        Std of VH channel in dB.
    """

    def __init__(
        self,
        vv_mean: Optional[float] = None,
        vv_std: Optional[float] = None,
        vh_mean: Optional[float] = None,
        vh_std: Optional[float] = None,
    ) -> None:
        self.vv_mean = vv_mean
        self.vv_std = vv_std
        self.vh_mean = vh_mean
        self.vh_std = vh_std

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply channel-wise normalization.

        Parameters
        ----------
        image : np.ndarray
            Array of shape (2, H, W) in dB.

        Returns
        -------
        np.ndarray
            Normalized array of shape (2, H, W).
        """
        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError(
                f"Expected (2, H, W), got shape {image.shape}"
            )

        vv_mean = self.vv_mean if self.vv_mean is not None else float(image[0].mean())
        vv_std = self.vv_std if self.vv_std is not None else float(image[0].std())
        vh_mean = self.vh_mean if self.vh_mean is not None else float(image[1].mean())
        vh_std = self.vh_std if self.vh_std is not None else float(image[1].std())

        if vv_std == 0:
            vv_std = 1.0
        if vh_std == 0:
            vh_std = 1.0

        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[0] = (image[0] - vv_mean) / vv_std
        normalized[1] = (image[1] - vh_mean) / vh_std
        return normalized

    def to_dict(self) -> Dict[str, float]:
        """Return normalization statistics as a dictionary."""
        return {
            "vv_mean": float(self.vv_mean) if self.vv_mean is not None else None,
            "vv_std": float(self.vv_std) if self.vv_std is not None else None,
            "vh_mean": float(self.vh_mean) if self.vh_mean is not None else None,
            "vh_std": float(self.vh_std) if self.vh_std is not None else None,
        }

    def save(self, path: str | Path) -> None:
        """Save normalization statistics to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ChannelNormalizer":
        """Load normalization statistics from a JSON file."""
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            vv_mean=data.get("vv_mean"),
            vv_std=data.get("vv_std"),
            vh_mean=data.get("vh_mean"),
            vh_std=data.get("vh_std"),
        )


def compute_normalization_stats(
    image_paths,
    chunk_size: int = 50,
) -> Tuple[float, float, float, float]:
    """Compute training-only channel-wise mean/std.

    Parameters
    ----------
    image_paths : iterable of str or Path
        Paths to training SAR images.
    chunk_size : int
        Number of images to process before updating running statistics.

    Returns
    -------
    tuple of float
        (vv_mean, vv_std, vh_mean, vh_std)
    """
    from .io import read_tiff

    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("No image paths provided")

    vv_sum = 0.0
    vv_sq_sum = 0.0
    vh_sum = 0.0
    vh_sq_sum = 0.0
    count = 0

    for path in image_paths:
        image = read_tiff(path)
        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError(
                f"Expected (2, H, W), got shape {image.shape} for {path}"
            )
        vv = image[0].astype(np.float64)
        vh = image[1].astype(np.float64)
        vv_sum += float(vv.sum())
        vv_sq_sum += float((vv * vv).sum())
        vh_sum += float(vh.sum())
        vh_sq_sum += float((vh * vh).sum())
        count += vv.size

    if count == 0:
        raise ValueError("No valid pixels found")

    vv_mean = vv_sum / count
    vh_mean = vh_sum / count
    vv_var = max(vv_sq_sum / count - vv_mean * vv_mean, 0.0)
    vh_var = max(vh_sq_sum / count - vh_mean * vh_mean, 0.0)
    vv_std = float(np.sqrt(vv_var))
    vh_std = float(np.sqrt(vh_var))

    if vv_std == 0:
        vv_std = 1.0
    if vh_std == 0:
        vh_std = 1.0

    return vv_mean, vv_std, vh_mean, vh_std