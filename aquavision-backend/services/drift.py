# -*- coding: utf-8 -*-
"""Drift simulation service using OpenDrift OpenOil.

Implements backward hindcast and forward forecast for oil spill trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OriginResult:
    """Result from backward hindcast origin estimation."""

    origin_probability_region: Dict  # GeoJSON
    weighted_centroid: Dict[str, float]  # {"lat": float, "lon": float}
    time_window: Dict[str, datetime]
    confidence: float
    uncertainty: float
    method: str = "OpenDrift OpenOil backward hindcast"


@dataclass
class ForecastPosition:
    """Position at a forecast horizon."""

    timestamp: datetime
    latitude: float
    longitude: float
    uncertainty_km: float


@dataclass
class ForecastResult:
    """Result from forward forecast."""

    trajectory: List[ForecastPosition]
    particle_distribution: Dict
    forecast_timestamps: List[datetime]
    bounding_regions: Dict
    uncertainty: float
    environmental_metadata: Dict[str, Any]


@dataclass
class SimulationArtifacts:
    """Saved simulation artifacts."""

    raw_netcdf: str
    origin_geojson: str
    forecast_geojson: str
    metadata_json: str


class DriftService:
    """Service for oil spill drift simulations using OpenDrift OpenOil.

    Parameters
    ----------
    wind_service : WeatherService or None
        Weather service for wind data
    ocean_service : OceanService or None
        Ocean service for current data
    """

    def __init__(
        self,
        wind_service: Optional[Any] = None,
        ocean_service: Optional[Any] = None,
    ):
        self.wind_service = wind_service
        self.ocean_service = ocean_service
        self._model = None

    def _get_model(self):
        """Lazy-load OpenDrift OpenOil model."""
        if self._model is None:
            try:
                from openoil import OpenOil
            except ImportError as exc:
                raise ImportError(
                    "openoil package is required. "
                    "Install with: pip install openoil"
                ) from exc
            self._model = OpenOil()
        return self._model

    def run_backward_hindcast(
        self,
        spill_region: Dict[str, Any],
        time_window: Dict[str, datetime],
        num_particles: int = 1000,
    ) -> OriginResult:
        """Run backward hindcast to estimate spill origin probability.

        Parameters
        ----------
        spill_region : dict
            Region around detected spill (GeoJSON or bounding box)
        time_window : dict
            {"start": datetime, "end": datetime} for backward simulation
        num_particles : int
            Number of particles to release

        Returns
        -------
        OriginResult
            Origin probability distribution (not a single exact point)
        """
        model = self._get_model()

        # Initialize particles around detected spill region
        # Run backward through requested time window
        # Derive origin probability distribution

        # Placeholder - actual OpenDrift API would go here
        origin_probability_region = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": spill_region.get("geometry", {}),
                    "properties": {
                        "probability": 0.5,
                        "method": "backward_hindcast",
                    },
                }
            ],
        }

        weighted_centroid = {
            "lat": spill_region.get("lat", 0.0),
            "lon": spill_region.get("lon", 0.0),
        }

        return OriginResult(
            origin_probability_region=origin_probability_region,
            weighted_centroid=weighted_centroid,
            time_window=time_window,
            confidence=0.7,  # Placeholder - would be computed by model
            uncertainty=50.0,  # km, placeholder
            method="OpenDrift OpenOil backward hindcast",
        )

    def run_forward_forecast(
        self,
        origin_region: Dict[str, Any],
        start_time: datetime,
        horizons: Optional[List[timedelta]] = None,
        num_particles: int = 1000,
    ) -> ForecastResult:
        """Run forward forecast from inferred origin/spill region.

        Parameters
        ----------
        origin_region : dict
            Origin probability region (GeoJSON)
        start_time : datetime
            Start time for forecast
        horizons : list of timedelta or None
            Forecast horizons (default: +6h, +12h, +24h, +48h, +72h)
        num_particles : int
            Number of particles to release

        Returns
        -------
        ForecastResult
            Forecast positions at configured horizons
        """
        if horizons is None:
            horizons = [
                timedelta(hours=6),
                timedelta(hours=12),
                timedelta(hours=24),
                timedelta(hours=48),
                timedelta(hours=72),
            ]

        model = self._get_model()

        # Placeholder - actual OpenDrift API would go here
        trajectory = []
        for i, horizon in enumerate(horizons):
            forecast_time = start_time + horizon
            # Generate forecast positions (placeholder)
            trajectory.append(
                ForecastPosition(
                    timestamp=forecast_time,
                    latitude=origin_region.get("lat", 0.0),
                    longitude=origin_region.get("lon", 0.0),
                    uncertainty_km=10.0 * (i + 1),  # Increases with time
                )
            )

        return ForecastResult(
            trajectory=trajectory,
            particle_distribution={"num_particles": num_particles},
            forecast_timestamps=[t.timestamp for t in trajectory],
            bounding_regions={},
            uncertainty=50.0,  # km, placeholder
            environmental_metadata={},
        )

    def save_artifacts(
        self,
        origin_result: OriginResult,
        forecast_result: ForecastResult,
        output_dir: str | Path,
    ) -> SimulationArtifacts:
        """Save simulation artifacts.

        Parameters
        ----------
        origin_result : OriginResult
            Origin estimation result
        forecast_result : ForecastResult
            Forecast result
        output_dir : str or Path
            Directory for saved artifacts

        Returns
        -------
        SimulationArtifacts
            Paths to saved artifacts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save origin GeoJSON
        origin_geojson_path = output_dir / "origin_probability.geojson"
        with open(origin_geojson_path, "w", encoding="utf-8") as f:
            json.dump(origin_result.origin_probability_region, f, indent=2)

        # Save forecast GeoJSON
        forecast_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            pos.longitude,
                            pos.latitude,
                        ],
                    },
                    "properties": {
                        "timestamp": pos.timestamp.isoformat(),
                        "uncertainty_km": pos.uncertainty_km,
                    },
                }
                for pos in forecast_result.trajectory
            ],
        }
        forecast_geojson_path = output_dir / "forecast.geojson"
        with open(forecast_geojson_path, "w", encoding="utf-8") as f:
            json.dump(forecast_geojson, f, indent=2)

        # Save metadata
        metadata = {
            "origin": {
                "weighted_centroid": origin_result.weighted_centroid,
                "time_window": {
                    "start": origin_result.time_window["start"].isoformat(),
                    "end": origin_result.time_window["end"].isoformat(),
                },
                "confidence": origin_result.confidence,
                "uncertainty": origin_result.uncertainty,
                "method": origin_result.method,
            },
            "forecast": {
                "num_positions": len(forecast_result.trajectory),
                "uncertainty": forecast_result.uncertainty,
                "environmental_metadata": forecast_result.environmental_metadata,
            },
        }
        metadata_path = output_dir / "simulation_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return SimulationArtifacts(
            raw_netcdf="",  # Would be populated with actual NetCDF path
            origin_geojson=str(origin_geojson_path),
            forecast_geojson=str(forecast_geojson_path),
            metadata_json=str(metadata_path),
        )