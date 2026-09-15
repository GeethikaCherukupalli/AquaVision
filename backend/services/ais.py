# -*- coding: utf-8 -*-
"""AIS candidate retrieval service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..providers.ais import AISTrajectory, AISPosition, AISProvider


@dataclass
class AISCandidate:
    """Candidate vessel with spatial-temporal filter applied."""

    vessel_id: str
    positions: List[AISPosition]
    minimum_distance_km: Optional[float] = None
    time_difference_minutes: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AISService:
    """Service for retrieving and filtering AIS candidates."""

    def __init__(self, provider: Optional[AISProvider] = None):
        self.provider = provider

    def get_candidates(
        self,
        origin_region: Dict[str, Any],
        origin_time_window: Dict[str, datetime],
        spatial_buffer_km: float = 50.0,
        temporal_buffer_hours: float = 24.0,
    ) -> List[AISCandidate]:
        """Retrieve candidate vessel tracks near spill origin.

        Parameters
        ----------
        origin_region : dict
            Origin probability region (GeoJSON or bounding box)
        origin_time_window : dict
            {"start": datetime, "end": datetime}
        spatial_buffer_km : float
            Spatial buffer around origin region in km
        temporal_buffer_hours : float
            Temporal buffer around origin time window in hours

        Returns
        -------
        List[AISCandidate]
            Candidate vessel tracks that passed filters
        """
        if self.provider is None:
            raise ValueError("No AIS provider configured")

        bbox = self._compute_bbox(origin_region, spatial_buffer_km)
        start_time = origin_time_window["start"]
        end_time = origin_time_window["end"]

        trajectories = self.provider.query_historical(
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
        )

        candidates = []
        for trajectory in trajectories:
            candidates.append(
                self._filter_trajectory(
                    trajectory, origin_region, origin_time_window
                )
            )

        return candidates

    def _compute_bbox(
        self,
        origin_region: Dict[str, Any],
        spatial_buffer_km: float,
    ) -> Dict[str, float]:
        """Compute expanded bounding box."""
        if "bbox" in origin_region:
            # Use existing bbox
            return origin_region["bbox"]

        # Compute from geometry
        return {"west": 0.0, "south": 0.0, "east": 1.0, "north": 1.0}

    def _filter_trajectory(
        self,
        trajectory: AISTrajectory,
        origin_region: Dict[str, Any],
        origin_time_window: Dict[str, datetime],
    ) -> AISCandidate:
        """Filter and annotate a single trajectory."""
        # Placeholder for actual filtering logic
        return AISCandidate(
            vessel_id=trajectory.vessel_id,
            positions=trajectory.positions,
            metadata=trajectory.metadata,
        )