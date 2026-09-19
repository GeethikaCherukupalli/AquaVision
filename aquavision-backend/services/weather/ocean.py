# -*- coding: utf-8 -*-
"""Ocean data service.

Uses Copernicus Marine provider for ocean current data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from aquavision.providers.copernicus_marine import (
    CopernicusMarineProvider,
    OceanCurrentData,
)


@dataclass
class OceanData:
    u_component: Any
    v_component: Any
    dataset_info: Dict[str, Any]
    bbox: Dict[str, float]
    start_time: datetime
    end_time: datetime


class OceanService:
    """Service for retrieving ocean environmental data."""

    def __init__(self, provider: Optional[CopernicusMarineProvider] = None):
        self.provider = provider or CopernicusMarineProvider()

    def get_currents(
        self,
        bbox: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        dataset_id: Optional[str] = None,
    ) -> OceanData:
        """Get ocean current data for a region and time window.

        Parameters
        ----------
        bbox : dict
            Geographic bounding box with keys: west, south, east, north
        start_time : datetime
            Start of the time window
        end_time : datetime
            End of the time window
        dataset_id : str or None
            Specific dataset ID to use

        Returns
        -------
        OceanData
            Ocean current data with u/v components in m/s
        """
        current_data = self.provider.get_currents(
            bbox, start_time, end_time, dataset_id
        )

        return OceanData(
            u_component=current_data.u_component,
            v_component=current_data.v_component,
            dataset_info={
                "name": current_data.dataset.name,
                "version": current_data.dataset.version,
                "variables": current_data.dataset.variables,
            },
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
        )
