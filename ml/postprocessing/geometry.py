# -*- coding: utf-8 -*-
"""Geospatial spill geometry characterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SpillGeometry:
    """Geometric and geospatial properties of a spill mask."""

    area_km2: float
    perimeter_km: float
    centroid_lat: float
    centroid_lon: float
    bounding_box: Dict[str, float]
    length_km: float
    width_km: float
    orientation_deg: float
    compactness: float
    component_count: int
    pixel_count: int


class SpillGeometryExtractor:
    """Extract geometric and geospatial properties from a spill mask.

    Parameters
    ----------
    crs : str or None
        Coordinate reference system of the mask.
    transform : affine transform or None
        Geospatial transform of the mask.
    """

    def __init__(
        self,
        crs: Optional[str] = None,
        transform=None,
    ) -> None:
        self.crs = crs
        self.transform = transform

    def extract(
        self,
        mask: np.ndarray,
        crs: Optional[str] = None,
        transform=None,
    ) -> SpillGeometry:
        """Extract geometry from a binary mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask of shape (H, W).
        crs : str or None
            Coordinate reference system.
        transform : affine transform or None
            Geospatial transform.

        Returns
        -------
        SpillGeometry
            Geometric and geospatial properties.
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        binary = (mask > 0).astype(np.uint8)
        component_count, _ = self._connected_components(binary)
        pixel_count = int(binary.sum())

        if pixel_count == 0:
            return SpillGeometry(
                area_km2=0.0,
                perimeter_km=0.0,
                centroid_lat=None,
                centroid_lon=None,
                bounding_box={"min_lat": None, "max_lat": None, "min_lon": None, "max_lon": None},
                length_km=0.0,
                width_km=0.0,
                orientation_deg=0.0,
                compactness=0.0,
                component_count=component_count,
                pixel_count=0,
            )

        # Pixel-space geometry (for reference)
        rows, cols = np.where(binary == 1)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Geodesic/geographic calculations when transform is available
        area_km2 = 0.0
        perimeter_km = 0.0
        centroid_lat = None
        centroid_lon = None
        bounding_box = {"min_lat": None, "max_lat": None, "min_lon": None, "max_lon": None}

        if self._has_georeferencing(crs, transform):
            area_km2, perimeter_km, centroid_lat, centroid_lon, bounding_box = (
                self._geospatial_metrics(binary, crs, transform)
            )
        else:
            # Fallback: report pixel-space geometry with explicit caveat
            area_km2 = None
            perimeter_km = None
            centroid_lat = None
            centroid_lon = None
            bounding_box = {
                "min_row": int(min_row),
                "max_row": int(max_row),
                "min_col": int(min_col),
                "max_col": int(max_col),
            }

        # Pixel-space length/width/orientation (for reference)
        length_px = max_row - min_row + 1
        width_px = max_col - min_col + 1
        orientation_deg = self._orientation_degrees(rows, cols)

        return SpillGeometry(
            area_km2=float(area_km2) if area_km2 is not None else None,
            perimeter_km=float(perimeter_km) if perimeter_km is not None else None,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            bounding_box=bounding_box,
            length_km=(length_px / 1000.0) if transform is not None else None,
            width_km=(width_px / 1000.0) if transform is not None else None,
            orientation_deg=orientation_deg,
            compactness=self._compactness(pixel_count, perimeter_km),
            component_count=component_count,
            pixel_count=pixel_count,
        )

    def _has_georeferencing(
        self,
        crs: Optional[str],
        transform=None,
    ) -> bool:
        """Check if georeferencing information is available."""
        return crs is not None or transform is not None

    def _geospatial_metrics(
        self,
        mask: np.ndarray,
        crs: Optional[str],
        transform=None,
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        """Compute geospatial metrics using pyproj/geographiclib.

        This is a simplified implementation. For production-quality
        measurements, use a projected CRS or geodesic library.
        """
        try:
            from pyproj import Transformer
        except ImportError:
            return 0.0, 0.0, None, None, {
                "min_lat": None,
                "max_lat": None,
                "min_lon": None,
                "max_lon": None,
            }

        rows, cols = np.where(mask == 1)
        if len(rows) == 0:
            return 0.0, 0.0, None, None, {
                "min_lat": None,
                "max_lat": None,
                "min_lon": None,
                "max_lon": None,
            }

        if transform is not None:
            from affine import Affine

            transform = Affine(transform)
            xs = [transform.c + col * transform.a + col * transform.b for col in cols]
            ys = [transform.f + col * transform.d + col * transform.e for col in cols]
        else:
            # Fallback: assume CRS is EPSG:4326
            xs = cols.astype(float)
            ys = rows.astype(float)

        lat = [float(y) for y in ys]
        lon = [float(x) for x in xs]

        # Bounding box
        min_lat = min(lat)
        max_lat = max(lat)
        min_lon = min(lon)
        max_lon = max(lon)

        # Centroid (mean of all foreground pixels)
        centroid_lat = float(np.mean(lat))
        centroid_lon = float(np.mean(lon))

        # Area: approximate using geodesic distance between corners
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        area = abs(geod.polygon_area_perimeter(
            [min_lon, max_lon, max_lon, min_lon],
            [min_lat, min_lat, max_lat, max_lat],
        )[0]) / 1e6

        # Perimeter: approximate using geodesic distance
        perimeter = 0.0
        for i in range(len(lat) - 1):
            _, _, dist = geod.inv(lon[i], lat[i], lon[i + 1], lat[i + 1])
            perimeter += dist

        return area / 1e6, perimeter / 1000.0, centroid_lat, centroid_lon, {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        }

    def _connected_components(
        self,
        mask: np.ndarray,
    ) -> tuple:
        """Compute connected components (4-connectivity)."""
        height, width = mask.shape
        labeled = np.zeros(mask.shape, dtype=np.int32)
        components = []

        for row in range(height):
            for col in range(width):
                if mask[row, col] == 0 or labeled[row, col] != 0:
                    continue

                component_id = len(components) + 1
                stack = [(row, col)]
                labeled[row, col] = component_id
                size = 0

                while stack:
                    current_row, current_col = stack.pop()
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = current_row + dr, current_col + dc
                        if (
                            0 <= nr < height
                            and 0 <= nc < width
                            and mask[nr, nc] == 1
                            and labeled[nr, nc] == 0
                        ):
                            labeled[nr, nc] = component_id
                            stack.append((nr, nc))
                components.append(size)

        return labeled, len(components)

    def _orientation_degrees(self, rows: np.ndarray, cols: np.ndarray) -> float:
        """Compute the orientation of a spill in degrees."""
        if len(rows) < 2:
            return 0.0

        covariance = np.cov(rows, cols)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        max_index = np.argmax(eigenvalues)
        angle = np.degrees(np.arctan2(eigenvectors[0, max_index], eigenvectors[1, max_index]))
        return float(angle % 180.0)

    def _compactness(self, pixel_count: int, perimeter_km: Optional[float]) -> float:
        """Compute compactness (4πA/P²)."""
        if perimeter_km is None or perimeter_km <= 0 or pixel_count == 0:
            return 0.0
        # Approximate: use pixel count as area proxy (only valid for square pixels)
        return float(4.0 * np.pi * pixel_count / (perimeter_km * perimeter_km * 1e6))