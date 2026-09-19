# -*- coding: utf-8 -*-
"""Model artifact management for Stage 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


class ModelArtifactManager:
    """Manage model artifacts and metadata."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(
        self,
        model_name: str,
        metadata: Dict,
    ) -> Path:
        """Save model metadata to JSON."""
        metadata_path = self.artifact_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return metadata_path

    def load_metadata(
        self,
        model_name: str,
    ) -> Dict:
        """Load model metadata from JSON."""
        metadata_path = self.artifact_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_checkpoint(
        self,
        model_name: str,
        state_dict,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Save a model checkpoint."""
        import torch

        checkpoint_path = self.artifact_dir / f"{model_name}.pth"
        torch.save(
            {
                "model_state_dict": state_dict,
                "metadata": metadata or {},
            },
            str(checkpoint_path),
        )
        return checkpoint_path

    def load_checkpoint(
        self,
        model_name: str,
    ) -> Dict:
        """Load a model checkpoint."""
        import torch

        checkpoint_path = self.artifact_dir / f"{model_name}.pth"
        return torch.load(str(checkpoint_path), map_location="cpu")