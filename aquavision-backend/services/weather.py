# -*- coding: utf-8 -*-
"""Weather data service.

Uses Open-Meteo provider for wind data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from ..providers.weather import WeatherProvider, WeatherData


@dataclass
class WindData:
    u_wind: Any  # eastward wind component (m/s)
    v_wind: Any  # northward wind component (m/s)
    timestamp: datetime
    latitude: float
    longitude: float
    metadata: Dict[str, Any]


class WeatherService:
    """Service for retrieving weather data."""

    def __init__(self, provider: Optional[WeatherProvider] = None):
        self.provider = provider or WeatherProvider()

    def get_wind(
        self,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> WindData:
        """Get wind data for a point and time window.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees
        longitude : float
            Longitude in decimal degrees
        start_time : datetime
            Start of time window
        end_time : datetime
            End of time window

        Returns
        -------
        WindData
            Wind data normalized to eastward/northward components in m/s
        """
        weather_data = self.provider.get_wind(
            latitude, longitude, start_time, end_time
        )

        return WindData(
            u_wind=weather_data.u_wind,
            v_wind=weather_data.v_wind,
            timestamp=weather_data.timestamp,
            latitude=weather_data.latitude,
            longitude=weather_data.longitude,
            metadata=weather_data.metadata,
        )

    def get_grid_wind(
        self,
        bbox: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
    ) -> Any:
        """Get wind data on a grid within bounding box."""
        # Placeholder for grid-based wind fetch
        raise NotImplementedError("Grid wind fetch not implemented")