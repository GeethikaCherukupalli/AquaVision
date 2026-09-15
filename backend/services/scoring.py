# -*- coding: utf-8 -*-
"""AIS vessel scoring service for spill correlation analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ais import AISCandidate


@dataclass
class VesselScore:
    """Scored vessel candidate."""

    vessel_id: str
    score: float  # 0-100
    classification: str  # "low", "moderate", "high"
    evidence: Dict[str, Any]
    positions: List[Any] = field(default_factory=list)
    proximity: Optional[float] = None
    temporal_correlation: Optional[float] = None
    trajectory_consistency: Optional[float] = None
    behavioral_anomaly: Optional[float] = None


# Default scoring weights (as specified)
DEFAULT_WEIGHTS = {
    "proximity": 0.35,
    "temporal_correlation": 0.30,
    "trajectory_consistency": 0.20,
    "behavioral_anomaly": 0.15,
}

# Classification thresholds
CLASSIFICATION_THRESHOLDS = {
    "low": 33.3,
    "moderate": 66.6,
    "high": 100.0,
}


class AISScoringService:
    """Service for scoring and ranking vessel candidates.

    Parameters
    ----------
    weights : dict or None
        Custom scoring weights. Must sum to 1.0.
        Defaults to DEFAULT_WEIGHTS.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total}"
            )

    def score_candidates(
        self,
        candidates: List[AISCandidate],
        origin_region: Dict[str, Any],
        origin_time_window: Dict[str, Any],
    ) -> List[VesselScore]:
        """Score and rank vessel candidates.

        Parameters
        ----------
        candidates : list of AISCandidate
            Candidate vessel tracks
        origin_region : dict
            Origin probability region
        origin_time_window : dict
            Origin time window

        Returns
        -------
        List[VesselScore]
            Scored candidates, sorted by score descending
        """
        scored = []
        for candidate in candidates:
            evidence = self._compute_evidence(
                candidate, origin_region, origin_time_window
            )
            score = self._compute_weighted_score(evidence)
            classification = self._classify(score)

            scored.append(
                VesselScore(
                    vessel_id=candidate.vessel_id,
                    score=score,
                    classification=classification,
                    evidence=evidence,
                    positions=candidate.positions,
                    proximity=evidence.get("minimum_distance"),
                    temporal_correlation=evidence.get("time_difference"),
                    trajectory_consistency=evidence.get(
                        "trajectory_consistency"
                    ),
                    behavioral_anomaly=evidence.get("behavioral_anomaly"),
                )
            )

        # Sort by score descending
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def _compute_evidence(
        self,
        candidate: AISCandidate,
        origin_region: Dict[str, Any],
        origin_time_window: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute evidence metrics for a candidate."""
        # Placeholder for actual evidence computation
        # These would be computed from trajectory analysis
        return {
            "minimum_distance": None,  # km from vessel to spill
            "time_difference": None,  # minutes from spill time
            "trajectory_consistency": None,  # 0-1 score
            "behavioral_anomaly": None,  # 0-1 score
        }

    def _compute_weighted_score(self, evidence: Dict[str, Any]) -> float:
        """Compute weighted score from evidence."""
        # Normalize evidence to 0-100 scale
        proximity_score = self._normalize_proximity(
            evidence.get("minimum_distance")
        )
        temporal_score = self._normalize_temporal(
            evidence.get("time_difference")
        )
        trajectory_score = self._normalize_trajectory(
            evidence.get("trajectory_consistency")
        )
        behavioral_score = self._normalize_behavioral(
            evidence.get("behavioral_anomaly")
        )

        # Weighted sum
        score = (
            self.weights["proximity"] * proximity_score
            + self.weights["temporal_correlation"] * temporal_score
            + self.weights["trajectory_consistency"] * trajectory_score
            + self.weights["behavioral_anomaly"] * behavioral_score
        )

        return round(score, 2)

    def _normalize_proximity(self, distance_km: Optional[float]) -> float:
        """Normalize proximity to 0-100 score."""
        if distance_km is None:
            return 0.0
        # Closer = higher score
        # Assume max relevant distance is 100 km
        if distance_km > 100:
            return 0.0
        return max(0.0, 100.0 * (1.0 - distance_km / 100.0))

    def _normalize_temporal(self, minutes: Optional[float]) -> float:
        """Normalize temporal correlation to 0-100 score."""
        if minutes is None:
            return 0.0
        # Closer in time = higher score
        # Assume max relevant time is 48 hours (2880 minutes)
        if minutes > 2880:
            return 0.0
        return max(0.0, 100.0 * (1.0 - minutes / 2880.0))

    def _normalize_trajectory(self, consistency: Optional[float]) -> float:
        """Normalize trajectory consistency to 0-100 score."""
        if consistency is None:
            return 0.0
        # Already 0-1, scale to 0-100
        return max(0.0, min(100.0, consistency * 100.0))

    def _normalize_behavioral(self, anomaly: Optional[float]) -> float:
        """Normalize behavioral anomaly to 0-100 score."""
        if anomaly is None:
            return 0.0
        # Already 0-1, scale to 0-100
        return max(0.0, min(100.0, anomaly * 100.0))

    def _classify(self, score: float) -> str:
        """Classify vessel based on score."""
        if score >= CLASSIFICATION_THRESHOLDS["high"]:
            return "high"
        elif score >= CLASSIFICATION_THRESHOLDS["moderate"]:
            return "moderate"
        else:
            return "low"

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update scoring weights (must sum to 1.0)."""
        self.weights = new_weights
        self._validate_weights()
