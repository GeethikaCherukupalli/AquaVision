# -*- coding: utf-8 -*-
"""Channel-wise SAR normalization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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


def _read_tiff_with_retry(path, max_retries: int = 3):
    """Read a TIFF, retrying up to ``max_retries`` times.

    Each attempt calls ``read_tiff(path)`` fresh, which reopens the file
    from scratch (no stale/partial rasterio handle is reused across
    attempts). This exists because Google Drive-backed TIFF reads can
    fail transiently (rclone/FUSE hiccups, timeouts) even though the
    file itself is fine.

    If every attempt fails, the path and the final exception are logged
    clearly and the exception is re-raised — a file that can't be read
    is never silently skipped or excluded from the statistics.
    """
    from .io import read_tiff

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return read_tiff(path)
        except Exception as exc:  # noqa: BLE001 - we re-raise below
            last_exc = exc
            logger.warning(
                "Read attempt %d/%d failed for %s: %s",
                attempt, max_retries, path, exc,
            )

    logger.error(
        "Giving up on %s after %d attempts. Final error: %s",
        path, max_retries, last_exc,
    )
    raise last_exc


def _load_normalization_cache(
    cache_path: Path,
    sorted_paths: list,
    random_seed: Optional[int],
) -> Optional[Tuple[float, float, float, float]]:
    """Return cached (vv_mean, vv_std, vh_mean, vh_std) if the cache is
    valid for this exact set of training paths + seed, else None."""
    if not cache_path.exists():
        return None

    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - cache is best-effort
        logger.warning(
            "Normalization cache at %s is unreadable (%s); recomputing.",
            cache_path, exc,
        )
        return None

    if (
        cached.get("image_paths") == sorted_paths
        and cached.get("count") == len(sorted_paths)
        and cached.get("random_seed") == random_seed
    ):
        logger.info(
            "Normalization cache hit at %s (%d training paths, "
            "random_seed=%s) — skipping rescan of TIFFs.",
            cache_path, len(sorted_paths), random_seed,
        )
        return (
            cached["vv_mean"], cached["vv_std"],
            cached["vh_mean"], cached["vh_std"],
        )

    logger.info(
        "Normalization cache at %s is stale (training path list or "
        "random_seed changed); recomputing.",
        cache_path,
    )
    return None


def _write_normalization_cache(
    cache_path: Path,
    sorted_paths: list,
    random_seed: Optional[int],
    stats: Tuple[float, float, float, float],
) -> None:
    vv_mean, vv_std, vh_mean, vh_std = stats
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "image_paths": sorted_paths,
                    "count": len(sorted_paths),
                    "random_seed": random_seed,
                    "vv_mean": vv_mean,
                    "vv_std": vv_std,
                    "vh_mean": vh_mean,
                    "vh_std": vh_std,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote normalization cache to %s", cache_path)
    except Exception as exc:  # noqa: BLE001 - cache is best-effort
        logger.warning(
            "Could not write normalization cache to %s (%s); "
            "continuing without it.",
            cache_path, exc,
        )


def compute_normalization_stats(
    image_paths,
    chunk_size: int = 50,
    dataset_root: "str | Path | None" = None,
    random_seed: Optional[int] = None,
    max_retries: int = 3,
) -> Tuple[float, float, float, float]:
    """Compute training-only channel-wise mean/std.

    Parameters
    ----------
    image_paths : iterable of str or Path
        Paths to training SAR images. Must be training scenes only —
        this function has no knowledge of splits and will happily
        compute statistics over whatever is passed in, so callers must
        continue to pass ``split.train`` paths only.
    chunk_size : int
        Number of images to process before updating running statistics.
    dataset_root : str, Path, or None
        If given, statistics are cached at
        ``<dataset_root>/artifacts/normalization_cache.json`` and
        reused on later calls with the exact same sorted path list and
        ``random_seed``. If None, no caching is performed (same
        behavior as before this change).
    random_seed : int or None
        Recorded in the cache and used as part of the cache-validity
        check, so a different split (different seed) never reuses
        another split's cached statistics.
    max_retries : int
        Number of read attempts per TIFF before giving up (see
        ``_read_tiff_with_retry``). Google Drive-backed reads can fail
        transiently; this does not change which data contributes to
        the statistics, only how robustly it's read.

    Returns
    -------
    tuple of float
        (vv_mean, vv_std, vh_mean, vh_std)
    """
    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("No image paths provided")

    sorted_paths = sorted(str(p) for p in image_paths)

    cache_path = None
    if dataset_root is not None:
        cache_path = Path(dataset_root) / "artifacts" / "normalization_cache.json"
        cached_stats = _load_normalization_cache(cache_path, sorted_paths, random_seed)
        if cached_stats is not None:
            return cached_stats

    vv_sum = 0.0
    vv_sq_sum = 0.0
    vh_sum = 0.0
    vh_sq_sum = 0.0
    count = 0

    for path in image_paths:
        image = _read_tiff_with_retry(path, max_retries=max_retries)
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

    stats = (vv_mean, vv_std, vh_mean, vh_std)

    if cache_path is not None:
        _write_normalization_cache(cache_path, sorted_paths, random_seed, stats)

    return stats