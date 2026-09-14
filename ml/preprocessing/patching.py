# -*- coding: utf-8 -*-
"""Patch extraction and generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class Patch:
    """A single 512×512 patch with spatial metadata."""

    image: np.ndarray
    mask: Optional[np.ndarray]
    row: int
    col: int
    scene_id: str
    split: str
    oil_fraction: Optional[float] = None
    label: Optional[int] = None

    @property
    def patch_size(self) -> int:
        """Return the patch spatial dimension."""
        return self.image.shape[-1]


class PatchExtractor:
    """Extract 512×512 patches from a 2048×2048 scene.

    Parameters
    ----------
    patch_size : int
        Size of each patch. Default is 512.
    """

    def __init__(self, patch_size: int = 512) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size

    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        scene_id: str = "",
        split: str = "",
    ) -> List[Patch]:
        """Extract all patches from a scene.

        Parameters
        ----------
        image : np.ndarray
            Array of shape (2, H, W).
        mask : np.ndarray or None
            Binary mask of shape (H, W).
        scene_id : str
            Identifier for the scene.
        split : str
            Train/validation/test split label.

        Returns
        -------
        list of Patch
        """
        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError(
                f"Expected image (2, H, W), got shape {image.shape}"
            )

        height, width = image.shape[1], image.shape[2]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Scene size {height}×{width} not divisible by patch_size "
                f"{self.patch_size}"
            )

        patches: List[Patch] = []
        for row in range(0, height, self.patch_size):
            for col in range(0, width, self.patch_size):
                patch_image = image[:, row : row + self.patch_size, col : col + self.patch_size].copy()
                patch_mask = None
                if mask is not None:
                    patch_mask = mask[row : row + self.patch_size, col : col + self.patch_size].copy()

                oil_fraction = None
                if patch_mask is not None:
                    oil_fraction = float(patch_mask.sum()) / float(patch_mask.size)

                label = None
                if oil_fraction is not None:
                    label = 1 if oil_fraction >= 0.01 else 0

                patches.append(
                    Patch(
                        image=patch_image,
                        mask=patch_mask,
                        row=row,
                        col=col,
                        scene_id=scene_id,
                        split=split,
                        oil_fraction=oil_fraction,
                        label=label,
                    )
                )
        return patches

    def iter_patches(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        scene_id: str = "",
        split: str = "",
    ) -> Iterator[Patch]:
        """Iterate over patches lazily."""
        yield from self.extract(image, mask, scene_id, split)


class PatchGenerator:
    """Generate patches from a dataset index.

    Parameters
    ----------
    patch_size : int
        Size of each patch. Default is 512.
    """

    def __init__(self, patch_size: int = 512) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size

    def generate_patches(
        self,
        scene_image: np.ndarray,
        scene_mask: Optional[np.ndarray] = None,
        scene_id: str = "",
        split: str = "",
    ) -> List[Patch]:
        """Generate patches from a single scene."""
        extractor = PatchExtractor(patch_size=self.patch_size)
        return extractor.extract(scene_image, scene_mask, scene_id, split)