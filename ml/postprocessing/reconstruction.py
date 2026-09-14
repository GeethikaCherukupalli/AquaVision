# -*- coding: utf-8 -*-
"""Patch-to-scene reconstruction for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ReconstructionResult:
    """Result of patch-to-scene reconstruction."""

    mask: np.ndarray
    probability_map: np.ndarray
    candidates: List[dict]


class PatchSceneReconstructor:
    """Reconstruct a full-scene mask from patch-level predictions.

    The reconstruction preserves each patch's original row/column
    position in the scene. This is critical for maintaining spatial
    context across the full 2048×2048 scene.
    """

    def __init__(
        self,
        scene_size: int = 2048,
        patch_size: int = 512,
    ) -> None:
        if scene_size % patch_size != 0:
            raise ValueError("scene_size must be divisible by patch_size")
        self.scene_size = scene_size
        self.patch_size = patch_size
        self.num_patches_per_axis = scene_size // patch_size

    def reconstruct(
        self,
        patches: List[dict],
        scene_shape: Optional[Tuple[int, int]] = None,
    ) -> ReconstructionResult:
        """Reconstruct a scene-level mask from patch predictions.

        Parameters
        ----------
        patches : list of dict
            Patch predictions with row/col offsets and masks.
        scene_shape : tuple or None
            Expected scene shape (H, W). If None, inferred from patches.

        Returns
        -------
        ReconstructionResult
            Scene mask, probability map, and candidate metadata.
        """
        if not patches:
            raise ValueError("No patches provided for reconstruction")

        if scene_shape is None:
            max_row = max(p["row"] for p in patches) + self.patch_size
            max_col = max(p["col"] for p in patches) + self.patch_size
            scene_shape = (max_row, max_col)

        height, width = scene_shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Scene size {height}×{width} not divisible by patch_size "
                f"{self.patch_size}"
            )

        mask = np.zeros((height, width), dtype=np.uint8)
        probability_map = np.zeros((height, width), dtype=np.float32)
        candidates = []

        for patch in patches:
            row = patch["row"]
            col = patch["col"]
            patch_mask = patch.get("mask")
            patch_prob = patch.get("probability")

            if patch_mask is None:
                continue

            if patch_mask.shape != (self.patch_size, self.patch_size):
                raise ValueError(
                    f"Patch mask shape {patch_mask.shape} doesn't match "
                    f"expected {(self.patch_size, self.patch_size)}"
                )

            row_slice = slice(row, row + self.patch_size)
            col_slice = slice(col, col + self.patch_size)

            if patch_prob is not None:
                probability_map[row_slice, col_slice] = patch_prob
                mask[row_slice, col_slice] = (
                    patch_prob >= self._default_threshold
                ).astype(np.uint8)
            else:
                mask[row_slice, col_slice] = patch_mask

            candidates.append(
                {
                    "row": row,
                    "col": col,
                    "probability": patch_prob,
                    "mask": patch_mask,
                }
            )

        return ReconstructionResult(
            mask=mask,
            probability_map=probability_map,
            candidates=candidates,
        )

    def _default_threshold(self) -> float:
        """Default probability threshold for reconstruction."""
        return 0.5

    def save_results(
        self,
        result: ReconstructionResult,
        mask_path: str,
        probability_path: str,
    ) -> None:
        """Save reconstructed mask and probability map."""
        from ..preprocessing.io import write_tiff

        write_tiff(mask_path, result.mask.astype(np.uint8))
        write_tiff(probability_path, result.probability_map.astype(np.float32))