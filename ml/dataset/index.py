# -*- coding: utf-8 -*-
"""Dataset indexing for Sentinel-1 oil spill dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class SceneIndexItem:
    """Index entry for a single scene.

    Parameters
    ----------
    scene_id : str
        Unique identifier for the scene.
    image_path : str
        Path to the SAR image TIFF.
    mask_path : Optional[str]
        Path to the corresponding segmentation mask TIFF.
    split : str
        Split label: train, validation, or test.
    class_label : Optional[str]
        Class label: oil, no_oil, or lookalike.
    part : Optional[str]
        Dataset part: Part1, Part2, or Part3.
    """

    scene_id: str
    image_path: str
    mask_path: Optional[str]
    split: str
    class_label: Optional[str]
    part: Optional[str] = None
    source_path: Optional[str] = None


class DatasetIndex:
    """Index dataset scenes and patches.

    The index is built lazily from the dataset root. The 90GB dataset is
    never copied into the repository — only relative paths are stored.

    Parameters
    ----------
    dataset_root : str or Path
        Root directory of the dataset.
    patch_size : int
        Patch size in pixels.
    oil_fraction_threshold : float
        Threshold for labeling patches from oil masks.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        patch_size: int = 512,
        oil_fraction_threshold: float = 0.01,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.patch_size = patch_size
        self.oil_fraction_threshold = oil_fraction_threshold
        self.scenes: List[SceneIndexItem] = []
        self.patch_index: List[dict] = []

    def discover_scenes(self) -> List[SceneIndexItem]:
        """Discover all scenes in the dataset.

        The exact nested directory structure inside Part1/Part2/Part3
        must be inspected rather than guessed. This method performs a
        conservative scan looking for TIFF image/mask pairs.

        Returns
        -------
        list of SceneIndexItem
        """
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.dataset_root}. "
                "Set AQUAVISION_DATA_ROOT to the correct path."
            )

        scenes: List[SceneIndexItem] = []
        seen_paths: set = set()

        for image_path in self.dataset_root.rglob("*.tif"):
            if image_path in seen_paths or not image_path.exists():
                continue
            seen_paths.add(image_path)

            # Skip files that are obviously masks (contain "mask" in name)
            if "mask" in image_path.name.lower():
                continue

            # Try to find a corresponding mask
            mask_path = self._find_mask(image_path)
            scene_id = image_path.stem

            # Determine part and class from path
            relative = image_path.relative_to(self.dataset_root)
            parts = relative.parts
            part = parts[0] if len(parts) > 1 else None
            class_label = self._infer_class_label(relative)

            scenes.append(
                SceneIndexItem(
                    scene_id=scene_id,
                    image_path=str(image_path),
                    mask_path=str(mask_path) if mask_path else None,
                    split="train",
                    class_label=class_label,
                    part=part,
                    source_path=str(image_path),
                )
            )

        self.scenes = scenes
        return scenes

    def _find_mask(self, image_path: Path) -> Optional[Path]:
        """Find the corresponding mask file for an image.

        Search strategy (conservative, no guessing):
        1. Same directory, same stem, different extension
        2. Parent directory, mask subdir
        3. Sibling directory containing "mask" in the name

        Returns
        -------
        Path or None
        """
        candidates = []
        stem = image_path.stem
        parent = image_path.parent

        # Same directory, same stem, different extension
        candidates.append(parent / f"{stem}.tif")
        candidates.append(parent / f"{stem}.tiff")
        candidates.append(parent / f"{stem}.png")

        # Look for mask in sibling directories
        for subdir in parent.iterdir() if parent.exists() else []:
            if subdir.is_dir() and "mask" in subdir.name.lower():
                candidates.append(subdir / f"{stem}.tif")
                candidates.append(subdir / f"{stem}.tiff")
                candidates.append(subdir / f"{stem}.png")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _infer_class_label(self, relative_path: Path) -> Optional[str]:
        """Infer class label from path components.

        Parameters
        ----------
        relative_path : Path
            Path relative to dataset root.

        Returns
        -------
        Optional[str]
            "oil", "no_oil", "lookalike", or None.
        """
        path_text = str(relative_path).lower()
        if "oil" in path_text and "mask" not in path_text:
            return "oil"
        if "no_oil" in path_text or "no-oil" in path_text or "nooil" in path_text:
            return "no_oil"
        if "lookalike" in path_text or "look_alike" in path_text:
            return "lookalike"
        return None

    def build_patch_index(self, split: Optional[str] = None) -> List[dict]:
        """Build a patch-level index without loading image data.

        Parameters
        ----------
        split : str or None
            If provided, only include scenes from that split.

        Returns
        -------
        list of dict
            Patch index entries with row/col/label/oil_fraction metadata.
        """
        from ..preprocessing.io import read_tiff, read_mask

        patches: List[dict] = []
        for scene in self.scenes:
            if split is not None and scene.split != split:
                continue
            if scene.image_path is None:
                continue

            image = read_tiff(scene.image_path)
            if image.ndim != 3 or image.shape[0] != 2:
                raise ValueError(
                    f"Expected (2, H, W), got shape {image.shape} for {scene.image_path}"
                )

            height, width = image.shape[1], image.shape[2]
            if height % self.patch_size != 0 or width % self.patch_size != 0:
                raise ValueError(
                    f"Scene size {height}×{width} not divisible by patch_size "
                    f"{self.patch_size} for {scene.scene_id}"
                )

            mask = None
            if scene.mask_path and Path(scene.mask_path).exists():
                mask = read_mask(scene.mask_path)

            for row in range(0, height, self.patch_size):
                for col in range(0, width, self.patch_size):
                    patch_mask = None
                    oil_fraction = None
                    label = None
                    if mask is not None:
                        patch_mask = mask[
                            row : row + self.patch_size,
                            col : col + self.patch_size,
                        ]
                        oil_fraction = float(patch_mask.sum()) / float(patch_mask.size)
                        label = (
                            1
                            if oil_fraction >= self.oil_fraction_threshold
                            else 0
                        )

                    patches.append(
                        {
                            "scene_id": scene.scene_id,
                            "image_path": scene.image_path,
                            "mask_path": scene.mask_path,
                            "row": row,
                            "col": col,
                            "split": scene.split,
                            "class_label": scene.class_label,
                            "part": scene.part,
                            "label": label,
                            "oil_fraction": oil_fraction,
                        }
                    )

        self.patch_index = patches
        return patches

    def save_index(self, path: str | Path) -> None:
        """Save the dataset index to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                [asdict(scene) for scene in self.scenes],
                handle,
                indent=2,
                sort_keys=True,
            )

    def load_index(self, path: str | Path) -> None:
        """Load the dataset index from a JSON file."""
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.scenes = [SceneIndexItem(**item) for item in data]


class PatchIndexBuilder:
    """Build a patch index from a dataset root."""

    def __init__(
        self,
        dataset_root: str | Path,
        patch_size: int = 512,
        oil_fraction_threshold: float = 0.01,
    ) -> None:
        self.index = DatasetIndex(dataset_root, patch_size, oil_fraction_threshold)

    def build(self, split: Optional[str] = None) -> List[dict]:
        """Build and return the patch index."""
        scenes = self.index.discover_scenes()
        return self.index.build_patch_index(split)