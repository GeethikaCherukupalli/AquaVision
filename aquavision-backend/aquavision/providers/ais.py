# -*- coding: utf-8 -*-
"""AIS provider interface for historical queries and live streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class AISPosition:
    """Single AIS position report."""

    vessel_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed: Optional[float] = None  # knots
    course: Optional[float] = None  # degrees
    heading: Optional[float] = None  # degrees
    vessel_type: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AISTrajectory:
    """Reconstructed vessel trajectory."""

    vessel_id: str
    positions: List[AISPosition]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AISProvider(Protocol):
    """Provider interface for AIS data."""

    def query_historical(
        self,
        bbox: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        **kwargs: Any,
    ) -> List[AISTrajectory]:
        """Query historical AIS tracks in a region/time window."""
        ...

    def stream_live(self, **kwargs: Any) -> Any:
        """Return a live AIS stream adapter."""
        ...


class MockAISProvider:
    """In-memory AIS provider for tests and demos."""

    def __init__(self, trajectories: Optional[List[AISTrajectory]] = None):
        self.trajectories = trajectories or []

    def query_historical(
        self,
        bbox: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        **kwargs: Any,
    ) -> List[AISTrajectory]:
        """Filter trajectories to region and time window."""
        result = []
        for trajectory in self.trajectories:
            positions = [
                pos
                for pos in trajectory.positions
                if (
                    bbox["west"] <= pos.longitude <= bbox["east"]
                    and bbox["south"] <= pos.latitude <= bbox["north"]
                    and start_time <= pos.timestamp <= end_time
                )
            ]
            if positions:
                result.append(
                    AISTrajectory(
                        vessel_id=trajectory.vessel_id,
                        positions=positions,
                        metadata=trajectory.metadata,
                    )
                )
        return result

    def stream_live(self, **kwargs: Any) -> Any:
        """Return a simple iterable stream."""
        raise NotImplementedError("Live stream not implemented in mock provider")
