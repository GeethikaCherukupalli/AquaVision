# -*- coding: utf-8 -*-
"""Tests for scene-level data splitting."""

import pytest
import numpy as np
from ml.dataset.index import SceneIndexItem
from ml.dataset.splitter import SceneSplitter


class TestSceneLevelSplitting:
    """Tests for scene-level splitting."""

    def test_no_patch_leakage(self):
        """Verify patches from same scene never cross splits."""
        # Create scenes with known patch counts
        scenes = [
            SceneIndexItem(
                scene_id="scene_A",
                image_path="/path/A.tif",
                mask_path="/mask/A.tif",
                split="train",
                class_label="oil",
            ),
            SceneIndexItem(
                scene_id="scene_B",
                image_path="/path/B.tif",
                mask_path="/mask/B.tif",
                split="train",
                class_label="oil",
            ),
            SceneIndexItem(
                scene_id="scene_C",
                image_path="/path/C.tif",
                mask_path="/mask/C.tif",
                split="train",
                class_label="no_oil",
            ),
        ]

        splitter = SceneSplitter(random_seed=42)
        result = splitter.split(scenes, train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25)

        # Each scene must be entirely in one split
        train_ids = {s.scene_id for s in result.train}
        val_ids = {s.scene_id for s in result.validation}
        test_ids = {s.scene_id for s in result.test}

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
        assert len(train_ids | val_ids | test_ids) == 3

    def test_stratified_by_class(self):
        """Test stratified splitting preserves class distribution."""
        scenes = [
            SceneIndexItem(f"oil_{i}", f"/path/oil_{i}.tif", f"/mask/oil_{i}.tif", "train", "oil")
            for i in range(8)
        ] + [
            SceneIndexItem(f"no_oil_{i}", f"/path/no_oil_{i}.tif", f"/mask/no_oil_{i}.tif", "train", "no_oil")
            for i in range(8)
        ] + [
            SceneIndexItem(f"lookalike_{i}", f"/path/lookalike_{i}.tif", f"/mask/lookalike_{i}.tif", "train", "lookalike")
            for i in range(8)
        ]

        splitter = SceneSplitter(random_seed=42)
        result = splitter.split(scenes, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15)

        # Check each class appears in each split
        for split_name, split_scenes in [
            ("train", result.train),
            ("validation", result.validation),
            ("test", result.test),
        ]:
            classes = {s.class_label for s in split_scenes}
            assert "oil" in classes
            assert "no_oil" in classes
            assert "lookalike" in classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])