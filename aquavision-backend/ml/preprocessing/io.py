# -*- coding: utf-8 -*-
"""Raster I/O utilities for SAR images and masks."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _suppress_georef_warnings() -> None:
    """Suppress expected rasterio georeferencing warnings for masks."""
    from rasterio.errors import NotGeoreferencedWarning

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def read_tiff(path: str | Path) -> np.ndarray:
    """Read a TIFF file into a NumPy array.

    Parameters
    ----------
    path : str or Path
        Path to the TIFF file.

    Returns
    -------
    np.ndarray
        Array of shape (C, H, W) as float32.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for reading TIFF files. "
            "Install with: pip install rasterio"
        ) from exc

    _suppress_georef_warnings()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(str(path)) as dataset:
        data = dataset.read()
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        return data.astype(np.float32)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a binary segmentation mask from a TIFF file.

    Parameters
    ----------
    path : str or Path
        Path to the mask file.

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W) as uint8.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for reading masks. "
            "Install with: pip install rasterio"
        ) from exc

    _suppress_georef_warnings()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(str(path)) as dataset:
        data = dataset.read(1)
        return (data > 0).astype(np.uint8)


def read_sar_product(path: str | Path) -> Tuple[np.ndarray, dict]:
    """Read a Sentinel-1-compatible SAR product.

    Parameters
    ----------
    path : str or Path
        Path to the SAR product TIFF.

    Returns
    -------
    tuple of (np.ndarray, dict)
        Image array of shape (2, H, W) and metadata dictionary.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for reading SAR products. "
            "Install with: pip install rasterio"
        ) from exc

    _suppress_georef_warnings()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    metadata: dict = {
        "path": str(path),
        "file_name": path.name,
    }

    with rasterio.open(str(path)) as dataset:
        data = dataset.read()
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        metadata["shape"] = data.shape
        metadata["dtype"] = str(data.dtype)
        metadata["crs"] = str(dataset.crs) if dataset.crs else None
        metadata["transform"] = (
            str(dataset.transform) if dataset.transform else None
        )
        metadata["width"] = dataset.width
        metadata["height"] = dataset.height
        metadata["count"] = dataset.count
        metadata["nodata"] = dataset.nodata

    return data.astype(np.float32), metadata


def write_tiff(
    path: str | Path,
    data: np.ndarray,
    crs: Optional[str] = None,
    transform=None,
) -> None:
    """Write a NumPy array to a GeoTIFF file.

    Parameters
    ----------
    path : str or Path
        Output path.
    data : np.ndarray
        Array to write. Shape (C, H, W) or (H, W).
    crs : str or None
        Coordinate reference system.
    transform : affine transform or None
        Geospatial transform.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for writing TIFF files. "
            "Install with: pip install rasterio"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": data.dtype,
    }
    if crs is not None:
        profile["crs"] = crs
    if transform is not None:
        profile["transform"] = transform

    with rasterio.open(str(path), "w", **profile) as dataset:
        dataset.write(data)