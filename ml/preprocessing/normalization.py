# -*- coding: utf-8 -*-
"""Channel-wise SAR normalization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Bump this if the cache file's schema changes. A cache written by an
# older/newer version is treated as stale and recomputed rather than
# trusted blindly.
NORMALIZATION_CACHE_VERSION = 1


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
) -> Optional[Dict]:
    """Return the cached dict if the cache exists, matches the current
    schema version, and its sorted training-image path list exactly
    matches ``sorted_paths``. Otherwise return None (cache miss)."""
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

    if cached.get("version") != NORMALIZATION_CACHE_VERSION:
        logger.info(
            "Normalization cache at %s has version %s (expected %s); "
            "recomputing.",
            cache_path, cached.get("version"), NORMALIZATION_CACHE_VERSION,
        )
        return None

    if (
        cached.get("training_image_paths") == sorted_paths
        and cached.get("num_training_images") == len(sorted_paths)
    ):
        return cached

    logger.info(
        "Normalization cache at %s is stale (training image path list "
        "does not match the current split); recomputing.",
        cache_path,
    )
    return None


def _write_normalization_cache_atomic(
    cache_path: Path,
    sorted_paths: list,
    random_seed: Optional[int],
    stats: Tuple[float, float, float, float],
) -> None:
    """Write the cache atomically: write to a temp file in the same
    directory, then os.replace() it into place, so an interrupted
    write (e.g. a Colab runtime restart mid-save) never leaves a
    corrupt/partial cache file behind."""
    import os
    import tempfile

    vv_mean, vv_std, vh_mean, vh_std = stats
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": NORMALIZATION_CACHE_VERSION,
        "training_image_paths": sorted_paths,
        "num_training_images": len(sorted_paths),
        "vv_mean": vv_mean,
        "vv_std": vv_std,
        "vh_mean": vh_mean,
        "vh_std": vh_std,
        "random_seed": random_seed,
    }

    tmp_name = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(cache_path.parent), prefix=".normalization_cache_", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_name, cache_path)  # atomic on POSIX and Windows
        tmp_name = None
        logger.info("Wrote normalization cache to %s", cache_path)
    except Exception as exc:  # noqa: BLE001 - cache is best-effort
        logger.warning(
            "Could not write normalization cache to %s (%s); "
            "continuing without it.",
            cache_path, exc,
        )
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.remove(tmp_name)


def compute_normalization_stats(
    image_paths,
    chunk_size: int = 50,
    dataset_root: "str | Path | None" = None,
    random_seed: Optional[int] = None,
    max_retries: int = 3,
) -> Tuple[float, float, float, float]:
    """Compute (or load cached) training-only channel-wise mean/std.

    Parameters
    ----------
    image_paths : iterable of str or Path
        Paths to training SAR images ONLY. This function has no
        knowledge of splits and will happily compute statistics over
        whatever is passed in, so callers must continue to pass
        ``split.train`` paths only — never validation or Part3/test.
    chunk_size : int
        Number of images to process before updating running statistics.
    dataset_root : str, Path, or None
        If given, statistics are cached at
        ``<dataset_root>/artifacts/normalization_cache.json`` and
        reused on later calls whose sorted training-image path list
        matches exactly. If None, no caching is performed.
    random_seed : int or None
        Recorded in the cache for provenance/reproducibility. Not part
        of the cache-validity check itself (a changed seed changes
        which scenes land in split.train, which changes the sorted
        path list, which already invalidates the cache on its own).
    max_retries : int
        Number of read attempts per TIFF before giving up.

    Returns
    -------
    tuple of float
        (vv_mean, vv_std, vh_mean, vh_std)
    """
    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("No image paths provided")

    sorted_paths = sorted(str(p) for p in image_paths)

    cache_path = (
        Path(dataset_root) / "artifacts" / "normalization_cache.json"
        if dataset_root is not None
        else None
    )

    if cache_path is not None:
        cached = _load_normalization_cache(cache_path, sorted_paths)
        if cached is not None:
            print("Loading normalization statistics from cache...")
            vv_mean, vv_std = cached["vv_mean"], cached["vv_std"]
            vh_mean, vh_std = cached["vh_mean"], cached["vh_std"]
            _print_normalization_summary(
                vv_mean, vv_std, vh_mean, vh_std, len(sorted_paths), cache_path,
            )
            return vv_mean, vv_std, vh_mean, vh_std

    print("Computing normalization statistics from training images...")

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
        _write_normalization_cache_atomic(cache_path, sorted_paths, random_seed, stats)

    _print_normalization_summary(
        vv_mean, vv_std, vh_mean, vh_std, len(sorted_paths), cache_path,
    )
    return stats


def _print_normalization_summary(
    vv_mean: float, vv_std: float, vh_mean: float, vh_std: float,
    num_training_images: int, cache_path: "Path | None",
) -> None:
    print(f"VV mean/std: {vv_mean:.6f} / {vv_std:.6f}")
    print(f"VH mean/std: {vh_mean:.6f} / {vh_std:.6f}")
    print(f"Number of training images: {num_training_images}")
    print(f"Cache path: {cache_path if cache_path is not None else '(caching disabled)'}")