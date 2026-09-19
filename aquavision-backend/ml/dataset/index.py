# -*- coding: utf-8 -*-
"""Dataset indexing for Sentinel-1 oil spill dataset.

The AquaVision dataset has three distinct parts:

Part1:
    Oil images + corresponding oil masks.
    Used for train/validation.

Part2:
    No-oil and lookalike images + corresponding masks.
    Used for train/validation.

Part3:
    Official held-out test set with ground-truth masks.
    NEVER used for training, validation, or threshold tuning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import rasterio


logger = logging.getLogger(__name__)


@dataclass
class SceneIndexItem:
    """Index entry for a single SAR scene."""

    scene_id: str
    image_path: str
    mask_path: Optional[str]
    split: str
    class_label: Optional[str]
    part: Optional[str] = None
    source_path: Optional[str] = None


class DatasetIndex:
    """Index the AquaVision Sentinel-1 oil-spill dataset.

    Dataset semantics
    -----------------
    Part1 + Part2:
        Training/validation candidates.

    Part3:
        Strictly held-out test data.

    Binary ResNet labels:
        1 = oil
        0 = not oil (no_oil + lookalike)

    Oil patch labels:
        Derived from mask oil fraction using
        ``oil_fraction_threshold``.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_scenes(self) -> List[SceneIndexItem]:
        """Discover scenes using the known Zenodo directory structure.

        Part1 and Part2 are marked as ``train`` initially. The existing
        scene-level splitter can subsequently divide these into train and
        validation.

        Part3 is ALWAYS marked ``test`` and must remain untouched.
        """

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.dataset_root}"
            )

        part1 = self.dataset_root / "Part1"
        part2 = self.dataset_root / "Part2"
        part3 = self.dataset_root / "Part3"

        scenes: List[SceneIndexItem] = []

        # --------------------------------------------------------------
        # Part 1: Oil
        # --------------------------------------------------------------

        part1_images = (
            part1
            / "01_Train_Val_Oil_Spill_images"
            / "Oil"
        )

        part1_masks = (
            part1
            / "01_Train_Val_Oil_Spill_mask"
            / "Mask_oil"
        )

        scenes.extend(
            self._build_paired_scenes(
                image_dir=part1_images,
                mask_dir=part1_masks,
                class_label="oil",
                part="Part1",
                split="train",
                mask_suffix="",
            )
        )

        # --------------------------------------------------------------
        # Part 2: No Oil
        # --------------------------------------------------------------

        part2_no_oil_images = (
            part2
            / "01_Train_Val_No_Oil_Images"
            / "No_oil"
        )

        part2_no_oil_masks = (
            part2
            / "01_Train_Val_No_Oil_mask"
            / "Mask_no_oil"
        )

        scenes.extend(
            self._build_paired_scenes(
                image_dir=part2_no_oil_images,
                mask_dir=part2_no_oil_masks,
                class_label="no_oil",
                part="Part2",
                split="train",
                mask_suffix="",
            )
        )

        # --------------------------------------------------------------
        # Part 2: Lookalike
        # --------------------------------------------------------------

        part2_lookalike_images = (
            part2
            / "01_Train_Val_Lookalike_images"
            / "Lookalike"
        )

        part2_lookalike_masks = (
            part2
            / "01_Train_Val_Lookalike_mask"
            / "Mask_lookalike"
        )

        scenes.extend(
            self._build_paired_scenes(
                image_dir=part2_lookalike_images,
                mask_dir=part2_lookalike_masks,
                class_label="lookalike",
                part="Part2",
                split="train",
                mask_suffix="",
            )
        )

        # --------------------------------------------------------------
        # Part 3: HELD-OUT TEST SET
        # --------------------------------------------------------------

        part3_root = (
            part3
            / "02_Test_images_and_ground_truth"
        )

        part3_images = part3_root / "Images"
        part3_masks = part3_root / "Mask"

        for class_dir_name, class_label in [
            ("Oil", "oil"),
            ("No oil", "no_oil"),
            ("Lookalike", "lookalike"),
        ]:
            image_dir = part3_images / class_dir_name
            mask_dir = part3_masks / class_dir_name

            scenes.extend(
                self._build_paired_scenes(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    class_label=class_label,
                    part="Part3",
                    split="test",
                    mask_suffix="_segmentation",
                )
            )

        # --------------------------------------------------------------
        # Final validation
        # --------------------------------------------------------------

        scene_ids = [scene.scene_id for scene in scenes]

        if len(scene_ids) != len(set(scene_ids)):
            duplicates = sorted(
                {
                    scene_id
                    for scene_id in scene_ids
                    if scene_ids.count(scene_id) > 1
                }
            )
            raise ValueError(
                f"Duplicate scene IDs detected: {duplicates}"
            )

        self.scenes = scenes

        logger.info(
            "Discovered %d scenes: Part1=%d, Part2=%d, Part3=%d",
            len(scenes),
            sum(s.part == "Part1" for s in scenes),
            sum(s.part == "Part2" for s in scenes),
            sum(s.part == "Part3" for s in scenes),
        )

        return scenes

    # ------------------------------------------------------------------
    # Deterministic directory pairing
    # ------------------------------------------------------------------

    def _build_paired_scenes(
        self,
        image_dir: Path,
        mask_dir: Path,
        class_label: str,
        part: str,
        split: str,
        mask_suffix: str = "",
    ) -> List[SceneIndexItem]:
        """Build scene entries from matching image/mask IDs.

        Only image-mask pairs are included.

        For Part2 No_oil, the known dataset contains 688 images but
        only 685 masks. The three unmatched images are explicitly
        reported and excluded from this paired index rather than being
        silently assigned an incorrect mask.
        """

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        if not mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {mask_dir}"
            )

        image_files = {
            path.stem: path
            for path in image_dir.glob("*.tif")
        }

        mask_files = {}

        for path in mask_dir.glob("*.tif"):
            stem = path.stem

            if mask_suffix and stem.endswith(mask_suffix):
                stem = stem[: -len(mask_suffix)]

            mask_files[stem] = path

        image_ids = set(image_files)
        mask_ids = set(mask_files)

        missing_masks = sorted(image_ids - mask_ids)
        orphan_masks = sorted(mask_ids - image_ids)

        if missing_masks:
            logger.warning(
                "%s %s: %d images have no matching mask. "
                "These images will NOT be included in the paired index. "
                "Missing IDs: %s",
                part,
                class_label,
                len(missing_masks),
                missing_masks[:20],
            )

        if orphan_masks:
            logger.warning(
                "%s %s: %d masks have no matching image. "
                "They will be ignored. IDs: %s",
                part,
                class_label,
                len(orphan_masks),
                orphan_masks[:20],
            )

        paired_ids = sorted(image_ids & mask_ids)

        scenes: List[SceneIndexItem] = []

        for sample_id in paired_ids:
            image_path = image_files[sample_id]
            mask_path = mask_files[sample_id]

            # Include dataset part + class to guarantee uniqueness.
            scene_id = (
                f"{part}_{class_label}_{sample_id}"
            )

            scenes.append(
                SceneIndexItem(
                    scene_id=scene_id,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    split=split,
                    class_label=class_label,
                    part=part,
                    source_path=str(image_path),
                )
            )

        logger.info(
            "%s %s: %d images, %d masks, %d matched pairs",
            part,
            class_label,
            len(image_files),
            len(mask_files),
            len(paired_ids),
        )

        return scenes

    # ------------------------------------------------------------------
    # Patch index
    # ------------------------------------------------------------------

    def build_patch_index(
        self,
        split: Optional[str] = None,
    ) -> List[dict]:
        """Build a patch-level index.

        Image dimensions are read from TIFF metadata without loading the
        entire SAR image into memory.

        Masks are loaded only when patch labels need to be calculated.
        """

        from ..preprocessing.io import read_mask

        patches: List[dict] = []

        for scene in self.scenes:

            if split is not None and scene.split != split:
                continue

            with rasterio.open(scene.image_path) as src:
                height = src.height
                width = src.width
                count = src.count

            if count != 2:
                raise ValueError(
                    f"Expected 2 SAR bands (VV/VH), got {count} "
                    f"for {scene.image_path}"
                )

            if (
                height % self.patch_size != 0
                or width % self.patch_size != 0
            ):
                raise ValueError(
                    f"Scene size {height}×{width} is not divisible "
                    f"by patch size {self.patch_size}: "
                    f"{scene.scene_id}"
                )

            mask = None

            if scene.mask_path:
                mask = read_mask(scene.mask_path)

                if mask.shape != (height, width):
                    raise ValueError(
                        f"Mask shape {mask.shape} does not match "
                        f"image shape {(height, width)} for "
                        f"{scene.scene_id}"
                    )

            for row in range(
                0,
                height,
                self.patch_size,
            ):
                for col in range(
                    0,
                    width,
                    self.patch_size,
                ):

                    label = None
                    oil_fraction = None

                    if mask is not None:
                        patch_mask = mask[
                            row : row + self.patch_size,
                            col : col + self.patch_size,
                        ]

                        oil_fraction = (
                            float(patch_mask.sum())
                            / float(patch_mask.size)
                        )

                        label = int(
                            oil_fraction
                            >= self.oil_fraction_threshold
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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        """Save the scene index to JSON."""

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                [asdict(scene) for scene in self.scenes],
                handle,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a compact summary of the discovered dataset."""

        return {
            "total_scenes": len(self.scenes),
            "parts": {
                "Part1": sum(
                    s.part == "Part1"
                    for s in self.scenes
                ),
                "Part2": sum(
                    s.part == "Part2"
                    for s in self.scenes
                ),
                "Part3": sum(
                    s.part == "Part3"
                    for s in self.scenes
                ),
            },
            "classes": {
                "oil": sum(
                    s.class_label == "oil"
                    for s in self.scenes
                ),
                "no_oil": sum(
                    s.class_label == "no_oil"
                    for s in self.scenes
                ),
                "lookalike": sum(
                    s.class_label == "lookalike"
                    for s in self.scenes
                ),
            },
            "splits": {
                "train": sum(
                    s.split == "train"
                    for s in self.scenes
                ),
                "validation": sum(
                    s.split == "validation"
                    for s in self.scenes
                ),
                "test": sum(
                    s.split == "test"
                    for s in self.scenes
                ),
            },
        }