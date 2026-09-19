from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


logger = logging.getLogger(__name__)


class ProductionAnalysisEngine:
    """Run the verified Stage-1 ResNet candidate path on a local SAR scene."""

    def __init__(self, artifact_root: str | Path | None = None) -> None:
        default_root = Path(__file__).resolve().parents[2] / "ml" / "artifacts"
        root = Path(artifact_root or os.getenv("AQUAVISION_ARTIFACT_ROOT", str(default_root)))
        self.artifact_dir = root / "resnet18" / "v1"
        self.checkpoint_path = self.artifact_dir / "resnet18_best_epoch06.pth"
        self.metadata_path = self.artifact_dir / "resnet18_metadata.json"
        self.normalization_path = self.artifact_dir / "sar_normalization.json"

    def _load_inference(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"ResNet checkpoint not found: {self.checkpoint_path}")
        if not self.metadata_path.exists() or not self.normalization_path.exists():
            raise FileNotFoundError("ResNet metadata or normalization artifact is missing")

        from ml.preprocessing.normalization import ChannelNormalizer
        from ml.resnet.inference import ResNetInference

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        normalizer = ChannelNormalizer.load(self.normalization_path)
        threshold = float(metadata["classification_threshold"])
        return ResNetInference(
            checkpoint_path=str(self.checkpoint_path),
            normalizer=normalizer,
            classifier_threshold=threshold,
            device="cpu",
        ), metadata

    def run(self, scene_path: str | Path, units: str = 'dB') -> Dict[str, Any]:
        path = Path(scene_path)
        if not path.exists():
            raise FileNotFoundError(f"SAR input not found: {path}")

        from ml.preprocessing.io import read_sar_product
        from ml.preprocessing.sar import prepare_vv_vh
        from ml.preprocessing.preview import vv_vh_preview_data_urls

        inference, metadata = self._load_inference()
        scene, scene_metadata = read_sar_product(path)
        scene = prepare_vv_vh(scene, units=units)
        previews = vv_vh_preview_data_urls(scene)
        if scene.ndim != 3 or scene.shape[0] != 2:
            raise ValueError(f"Expected a 2-channel VV/VH SAR scene, got {scene.shape}")
        if scene.shape[1:] != (512, 512):
            raise ValueError(f"Expected a 512x512 SAR scene, got {scene.shape[1:]}")
        valid = np.isfinite(scene)
        if not valid.all():
            raise ValueError(f"SAR scene contains {int((~valid).sum())} non-finite VV/VH values")
        logger.info(
            "SAR diagnostics source=%s shape=%s VV[min=%.3f max=%.3f mean=%.3f] "
            "VH[min=%.3f max=%.3f mean=%.3f] valid_pixels=%d",
            path,
            scene.shape,
            float(scene[0].min()), float(scene[0].max()), float(scene[0].mean()),
            float(scene[1].min()), float(scene[1].max()), float(scene[1].mean()),
            int(valid[0].sum() + valid[1].sum()),
        )

        patches = inference.predict_all(scene)
        probabilities = [float(p.probability) for p in patches]
        probability = max(probabilities, default=0.0)
        candidate = probability >= inference.classifier_threshold
        logger.info(
            "ResNet diagnostics input_shape=(1,2,512,512) patches=%d probability=%.6f threshold=%.6f",
            len(patches), probability, inference.classifier_threshold,
        )
        return {
            "analysis_id": f"production-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            "mode": "production",
            "status": "COMPLETED",
            "data_source": "cdse_processing_api",
            "source": "Sentinel-1",
            "candidate": bool(candidate),
            "classification_probability": probability,
            "classification_threshold": float(inference.classifier_threshold),
            "model": metadata["model"],
            "model_version": "v1",
            "input_channels": metadata["input_bands"],
            "patch_size": int(metadata["patch_size"]),
            "patch_count": len(patches),
            "patch_probabilities": probabilities,
            "scene": scene_metadata,
            "previews": previews,
            "stage_1": {
                "classifier": {
                    "model": metadata["model"],
                    "model_version": "v1",
                    "threshold": float(inference.classifier_threshold),
                    "probability": probability,
                    "candidate": bool(candidate),
                    "input_channels": metadata["input_bands"],
                    "patch_size": int(metadata["patch_size"]),
                },
                "quality": {"confidence": probability, "warnings": []},
            },
            "warnings": [],
        }