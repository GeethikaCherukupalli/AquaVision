# -*- coding: utf-8 -*-
"""Weather provider adapter using Open-Meteo.

Normalizes internally to:
* eastward wind u
* northward wind v
* m/s
* timestamp
* latitude
* longitude
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class WeatherData:
    u_wind: Any  # eastward wind component (m/s)
    v_wind: Any  # northward wind component (m/s)
    timestamp: datetime
    latitude: float
    longitude: float
    metadata: Dict[str, Any]


class WeatherProvider:
    """Adapter for weather data (initially Open-Meteo)."""

    def __init__(self):
        self.session = None

    def _get_session(self):
        """Get or create HTTP session."""
        if self.session is None:
            try:
                import requests  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "requests package is required. "
                    "Install with: pip install requests"
                ) from exc
            self.session = requests.Session()
        return self.session

    def get_wind(
        self,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> WeatherData:
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
        WeatherData
            Wind data normalized to eastward/northward components in m/s
        """
        session = self._get_session()
        return self._fetch_from_openmeteo(
            session, latitude, longitude, start_time, end_time
        )

    def _fetch_from_openmeteo(
        self,
        session: Any,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> WeatherData:
        """Fetch data from Open-Meteo API."""
        # Placeholder implementation
        raise NotImplementedError(
            "Open-Meteo fetch not implemented"
        )

    def _normalize_wind_components(
        self,
        wind_speed: Any,
        wind_direction: Any,
    ) -> tuple[Any, Any]:
        """Convert wind speed/direction to u/v components.

        Parameters
        ----------
        wind_speed : float or array
            Wind speed (m/s)
        wind_direction : float or array
            Wind direction (degrees from north, clockwise)

        Returns
        -------
        tuple
            (u_component, v_component) where:
            u = eastward (positive east)
            v = northward (positive north)
        """
        import numpy as np

        # Convert direction to radians (meteorological: degrees from north)
        dir_rad = np.radians(wind_direction)

        # u = -speed * sin(dir)  (eastward)
        # v = -speed * cos(dir)  (northward)
        # Negative because meteorological direction is where wind is FROM
        u = -wind_speed * np.sin(dir_rad)
        v = -wind_speed * np.cos(dir_rad)

        return u, v

    def get_metadata(self) -> Dict[str, Any]:
        """Get provider metadata."""
        return {
            "provider": "Open-Meteo",
            "base_url": "https://api.open-meteo.com/v1/forecast",
            "supports_forecast": True,
            "supports_historical": True,
        }