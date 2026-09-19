# -*- coding: utf-8 -*-
"""Complete Stage 1 inference pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models.resnet18.inference import ResNetInference
from ..models.unet.inference import UNetInference
from ..postprocessing.reconstruction import PatchSceneReconstructor
from ..postprocessing.postprocess import SpillPostprocessor
from ..postprocessing.geometry import SpillGeometryExtractor
from ..preprocessing.io import read_sar_product, write_tiff
from ..preprocessing.normalization import ChannelNormalizer
from ..preprocessing.patching import PatchExtractor


@dataclass
class Stage1Result:
    """Stage 1 inference result."""

    mask: np.ndarray
    probability_map: np.ndarray
    spill_geometry: Dict
    candidates: List[dict]
    metadata: Dict


class Stage1InferencePipeline:
    """End-to-end Stage 1 inference pipeline.

    Input:
        Sentinel-1-compatible 2-channel VV/VH scene
    Output:
        Scene-level spill mask, probability map, spill geometry, and metadata

    The pipeline is reusable by the future FastAPI backend.
    """

    def __init__(
        self,
        resnet_checkpoint: str,
        unet_checkpoint: str,
        classifier_threshold: float = 0.12,
        segmentation_threshold: float = 0.5,
        resnet_normalizer: Optional[ChannelNormalizer] = None,
        unet_normalizer: Optional[ChannelNormalizer] = None,
        device: Optional[str] = None,
    ) -> None:
        self.resnet = ResNetInference(
            checkpoint_path=resnet_checkpoint,
            normalizer=resnet_normalizer,
            classifier_threshold=classifier_threshold,
            device=device,
        )
        self.unet = UNetInference(
            checkpoint_path=unet_checkpoint,
            normalizer=unet_normalizer,
            segmentation_threshold=segmentation_threshold,
            device=device,
        )
        self.reconstructor = PatchSceneReconstructor()
        self.postprocessor = SpillPostprocessor(
            probability_threshold=segmentation_threshold,
        )
        self.geometry_extractor = SpillGeometryExtractor()

    def run(
        self,
        scene_path: str,
        output_dir: str | Path | None = None,
    ) -> Stage1Result:
        """Run the complete Stage 1 inference pipeline.

        Parameters
        ----------
        scene_path : str
            Path to the Sentinel-1-compatible SAR scene.
        output_dir : str or Path or None
            Directory for saving outputs.

        Returns
        -------
        Stage1Result
            Complete inference result.
        """
        # Read scene
        scene, metadata = read_sar_product(scene_path)
        if scene.ndim != 3 or scene.shape[0] != 2:
            raise ValueError(
                f"Expected 2-channel SAR scene, got shape {scene.shape}"
            )

        # Patch extraction
        patches = PatchExtractor(patch_size=512).extract(
            scene, scene_id=Path(scene_path).stem, split="inference"
        )

        # ResNet candidate gate
        candidates = self.resnet.predict_batch(patches)

        # U-Net segmentation on candidate patches
        self.unet.predict(candidates)

        # Reconstruct scene-level mask
        reconstruction = self.reconstructor.reconstruct(
            candidates, scene_shape=scene.shape[1:]
        )

        # Postprocess mask
        postprocessed = self.postprocessor.process(
            reconstruction.probability_map,
            reconstruction.mask,
        )

        # Geometry extraction
        geometry = self.geometry_extractor.extract(
            postprocessed.mask,
            crs=metadata.get("crs"),
            transform=metadata.get("transform"),
        )

        result = Stage1Result(
            mask=postprocessed.mask,
            probability_map=postprocessed.probability_map,
            spill_geometry=asdict(geometry),
            candidates=[
                {
                    "row": patch.row,
                    "col": patch.col,
                    "probability": patch.probability,
                    "mask": patch.mask.tolist() if patch.mask is not None else None,
                }
                for patch in candidates
            ],
            metadata={
                "scene_path": scene_path,
                "scene_metadata": metadata,
                "classifier_threshold": self.resnet.classifier_threshold,
                "segmentation_threshold": self.unet.segmentation_threshold,
                "patch_size": 512,
                "num_candidates": len(candidates),
                "num_components": postprocessed.component_count,
            },
        )

        if output_dir is not None:
            self.save_result(result, output_dir)

        return result

    def save_result(
        self,
        result: Stage1Result,
        output_dir: str | Path,
    ) -> None:
        """Save inference result to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        write_tiff(output_dir / "spill_mask.tif", result.mask.astype(np.uint8))
        write_tiff(
            output_dir / "spill_probability.tif",
            result.probability_map.astype(np.float32),
        )

        with open(output_dir / "spill_geometry.json", "w", encoding="utf-8") as handle:
            json.dump(result.spill_geometry, handle, indent=2)

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(result.metadata, handle, indent=2)