# -*- coding: utf-8 -*-
"""Lazy dataset loading for SAR images and masks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .index import DatasetIndex, SceneIndexItem
from ..preprocessing.io import read_tiff, read_mask
from ..preprocessing.normalization import ChannelNormalizer


class LazyDataset:
    """Lazy dataset loading for SAR images and masks.

    The dataset class does not load the entire 90GB dataset into memory.
    It reads patches lazily from the original TIFF files using window-based
    access.

    Parameters
    ----------
    dataset_index : DatasetIndex
        Dataset index containing scene metadata.
    normalizer : ChannelNormalizer or None
        Normalization configuration. If None, per-image normalization is
        applied using the image's own statistics.
    """

    def __init__(
        self,
        dataset_index: DatasetIndex,
        normalizer: Optional[ChannelNormalizer] = None,
        patch_size: int = 512,
        split: Optional[str] = None,
    ) -> None:
        self.dataset_index = dataset_index
        self.normalizer = normalizer
        self.patch_size = patch_size
        self.split = split
        self.scenes = (
            dataset_index.scenes
            if split is None
            else [scene for scene in dataset_index.scenes if scene.split == split]
        )
        if not self.scenes:
            raise ValueError(f"No scenes found for split '{split}'")

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return sum(self._count_patches(scene) for scene in self.scenes)

    def _count_patches(self, scene: SceneIndexItem) -> int:
        """Count patches in a scene without loading data."""
        image = read_tiff(scene.image_path)
        height, width = image.shape[1], image.shape[2]
        return (height // self.patch_size) * (width // self.patch_size)

    def __getitem__(self, index: int) -> dict:
        """Load a single patch lazily.

        Parameters
        ----------
        index : int
            Index of the patch.

        Returns
        -------
        dict
            Patch data with image, mask, label, and metadata.
        """
        scene = self._get_scene(index)
        row, col = self._get_row_col(index)

        image = read_tiff(scene.image_path)
        image = image[
            :, row : row + self.patch_size, col : col + self.patch_size
        ].copy()

        mask = None
        label = None
        oil_fraction = None
        if scene.mask_path and Path(scene.mask_path).exists():
            mask = read_mask(scene.mask_path)
            mask = mask[row : row + self.patch_size, col : col + self.patch_size]
            oil_fraction = float(mask.sum()) / float(mask.size)
            label = 1 if oil_fraction >= self.dataset_index.oil_fraction_threshold else 0

        if self.normalizer is not None:
            image = self.normalizer.normalize(image)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "oil_fraction": oil_fraction,
            "scene_id": scene.scene_id,
            "scene_split": scene.split,
            "row": row,
            "col": col,
            "patch_size": self.patch_size,
        }

    def _get_scene(self, index: int) -> SceneIndexItem:
        """Get the scene for a patch index."""
        remaining = index
        for scene in self.scenes:
            patch_count = self._count_patches(scene)
            if remaining < patch_count:
                return scene
            remaining -= patch_count
        raise IndexError(f"Patch index {index} out of range")

    def _get_row_col(self, index: int) -> Tuple[int, int]:
        """Get row/col offsets for a patch index."""
        scene = self._get_scene(index)
        image = read_tiff(scene.image_path)
        height, width = image.shape[1], image.shape[2]
        patches_per_row = width // self.patch_size
        local_index = index - self._scene_index_offset(scene)
        row = (local_index // patches_per_row) * self.patch_size
        col = (local_index % patches_per_row) * self.patch_size
        return row, col

    def _scene_index_offset(self, scene: SceneIndexItem) -> int:
        """Return the starting patch index for a scene."""
        offset = 0
        for current in self.scenes:
            if current is scene:
                return offset
            offset += self._count_patches(current)
        raise ValueError(f"Scene not in dataset: {scene.scene_id}")


class PatchDataset:
    """Dataset for ResNet/U-Net training with batchable tensor conversion."""

    def __init__(
        self,
        dataset: LazyDataset,
        transform=None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of patches."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """Get a patch with tensor conversion."""
        item = self.dataset[index]
        if self.transform is not None:
            item = self.transform(item)
        return item