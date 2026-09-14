# -*- coding: utf-8 -*-
"""Configurable mask postprocessing for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class PostprocessResult:
    """Result of mask postprocessing."""

    mask: np.ndarray
    probability_map: np.ndarray
    removed_components: int
    component_count: int


class SpillPostprocessor:
    """Postprocess segmentation masks with configurable filters.

    Supported operations:
    - Probability thresholding
    - Tiny connected component removal
    - Connected component analysis
    - Optional morphological cleanup

    Parameters
    ----------
    probability_threshold : float
        Threshold for probability-based binarization.
    min_component_size : int
        Minimum size of connected components to keep.
    max_component_size : int or None
        Maximum size of connected components to keep.
    morphology : str
        Morphological operation: "none", "opening", "closing", "dilate", "erode".
    kernel_size : int
        Size of the morphological kernel.
    """

    def __init__(
        self,
        probability_threshold: float = 0.5,
        min_component_size: int = 32,
        max_component_size: Optional[int] = None,
        morphology: str = "none",
        kernel_size: int = 3,
    ) -> None:
        if not (0.0 <= probability_threshold <= 1.0):
            raise ValueError("probability_threshold must be in [0, 1]")
        if min_component_size < 0:
            raise ValueError("min_component_size cannot be negative")
        if max_component_size is not None and max_component_size < min_component_size:
            raise ValueError("max_component_size must be >= min_component_size")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.probability_threshold = probability_threshold
        self.min_component_size = min_component_size
        self.max_component_size = max_component_size
        self.morphology = morphology
        self.kernel_size = kernel_size

    def process(
        self,
        probability_map: np.ndarray,
        binary_mask: Optional[np.ndarray] = None,
    ) -> PostprocessResult:
        """Process a probability map or binary mask.

        Parameters
        ----------
        probability_map : np.ndarray
            Probability map of shape (H, W).
        binary_mask : np.ndarray or None
            Optional pre-computed binary mask.

        Returns
        -------
        PostprocessResult
            Processed mask, probability map, and component statistics.
        """
        if probability_map.ndim != 2:
            raise ValueError(f"Expected 2D probability map, got shape {probability_map.shape}")

        if binary_mask is None:
            mask = (probability_map >= self.probability_threshold).astype(np.uint8)
        else:
            mask = (binary_mask > 0).astype(np.uint8)

        # Connected component analysis
        labeled, num_components = self._connected_components(mask)
        component_sizes = np.bincount(labeled.ravel())[1:] if num_components > 0 else np.array([])

        # Apply component size filters
        keep_components = np.zeros(num_components + 1, dtype=bool)
        keep_components[0] = True
        if component_sizes.size > 0:
            keep_components[1:] = (
                (component_sizes >= self.min_component_size)
                & (
                    component_sizes <= self.max_component_size
                    if self.max_component_size is not None
                    else np.inf
                )
            )

        if np.any(~keep_components[1:]):
            mask = np.where(keep_components[labeled], mask, 0).astype(np.uint8)

        # Morphological cleanup
        if self.morphology != "none":
            mask = self._apply_morphology(mask)

        # Recompute component count after filtering
        final_labeled, final_num_components = self._connected_components(mask)
        final_sizes = np.bincount(final_labeled.ravel())[1:] if final_num_components > 0 else np.array([])

        return PostprocessResult(
            mask=mask,
            probability_map=probability_map,
            removed_components=int(np.sum(~keep_components[1:])),
            component_count=int(final_num_components),
        )

    def _connected_components(
        self,
        mask: np.ndarray,
    ) -> tuple:
        """Compute connected components (4-connectivity)."""
        height, width = mask.shape
        labeled = np.zeros(mask.shape, dtype=np.int32)
        components = []

        for row in range(height):
            for col in range(width):
                if mask[row, col] == 0 or labeled[row, col] != 0:
                    continue

                component_id = len(components) + 1
                stack = [(row, col)]
                labeled[row, col] = component_id
                size = 0

                while stack:
                    current_row, current_col = stack.pop()
                    size += 1

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = current_row + dr, current_col + dc
                        if (
                            0 <= nr < height
                            and 0 <= nc < width
                            and mask[nr, nc] == 1
                            and labeled[nr, nc] == 0
                        ):
                            labeled[nr, nc] = component_id
                            stack.append((nr, nc))

                components.append(size)

        return labeled, len(components)

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operation."""
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)

        if self.morphology == "opening":
            return self._opening(mask, kernel)
        if self.morphology == "closing":
            return self._closing(mask, kernel)
        if self.morphology == "dilate":
            return self._dilate(mask, kernel)
        if self.morphology == "erode":
            return self._erode(mask, kernel)

        return mask

    def _opening(self, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Morphological opening (erosion then dilation)."""
        return self._dilate(self._erode(mask, kernel), kernel)

    def _closing(self, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Morphological closing (dilation then erosion)."""
        return self._erode(self._dilate(mask, kernel), kernel)

    def _erode(self, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Morphological erosion."""
        height, width = mask.shape
        result = np.zeros_like(mask, dtype=np.uint8)
        kernel_height, kernel_width = kernel.shape
        pad_h, pad_w = kernel_height // 2, kernel_width // 2

        for row in range(pad_h, height - pad_h + 1):
            for col in range(pad_w, width - pad_w + 1):
                patch = mask[row - pad_h : row + pad_h + 1, col - pad_w : col + pad_w + 1]
                if np.all(patch == 1):
                    result[row, col] = 1
        return result

    def _dilate(self, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Morphological dilation."""
        height, width = mask.shape
        result = np.zeros_like(mask, dtype=np.uint8)
        kernel_height, kernel_width = kernel.shape
        pad_h, pad_w = kernel_height // 2, kernel_width // 2

        for row in range(height):
            for col in range(width):
                patch = mask[row - pad_h : row + pad_h + 1, col - pad_w : col + pad_w + 1]
                patch = patch[
                    max(0, pad_h - row) : min(height, row + pad_h + 1),
                    max(0, pad_w - col) : min(width, col + pad_w + 1),
                ]
                if np.any(patch == 1):
                    result[row, col] = 1
        return result