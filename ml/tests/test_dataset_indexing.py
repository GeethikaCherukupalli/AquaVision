# -*- coding: utf-8 -*-
"""Tests for dataset indexing."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ml.dataset.index import DatasetIndex, SceneIndexItem
from ml.dataset.splitter import SceneSplitter, SceneSplit


class TestDatasetIndex:
    """Tests for DatasetIndex."""

    def test_scene_index_item_creation(self):
        """Test SceneIndexItem dataclass."""
        item = SceneIndexItem(
            scene_id="test_001",
            image_path="/path/to/image.tif",
            mask_path="/path/to/mask.tif",
            split="train",
            class_label="oil",
            part="Part1",
        )
        assert item.scene_id == "test_001"
        assert item.image_path == "/path/to/image.tif"
        assert item.mask_path == "/path/to/mask.tif"
        assert item.split == "train"
        assert item.class_label == "oil"
        assert item.part == "Part1"

    @patch("ml.dataset.index.Path.rglob")
    def test_discover_scenes_empty(self, mock_rglob):
        """Test discover_scenes with no scenes."""
        mock_rglob.return_value = []
        index = DatasetIndex("/fake/path")
        scenes = index.discover_scenes()
        assert scenes == []

    def test_class_label_inference(self):
        """Test class label inference from path."""
        index = DatasetIndex("/fake/path")
        path = Path("Part1/oil/scene_001.tif")
        assert index._infer_class_label(path) == "oil"

        path = Path("Part2/no_oil/scene_002.tif")
        assert index._infer_class_label(path) == "no_oil"

        path = Path("Part2/look_alike/scene_003.tif")
        assert index._infer_class_label(path) == "lookalike"


class TestSceneSplitter:
    """Tests for SceneSplitter."""

    def test_split_basic(self):
        """Test basic scene splitting."""
        scenes = [
            SceneIndexItem(f"scene_{i}", f"/path/{i}.tif", f"/mask/{i}.tif", "train", "oil")
            for i in range(10)
        ]
        splitter = SceneSplitter(random_seed=42)
        result = splitter.split(scenes, train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)

        assert len(result.train) == 7
        assert len(result.validation) == 2
        assert len(result.test) == 1
        assert len(result.all_scenes) == 10

    def test_split_per_class(self):
        """Test stratified scene splitting."""
        scenes = [
            SceneIndexItem(f"oil_{i}", f"/path/oil_{i}.tif", f"/mask/oil_{i}.tif", "train", "oil")
            for i in range(5)
        ] + [
            SceneIndexItem(f"no_oil_{i}", f"/path/no_oil_{i}.tif", f"/mask/no_oil_{i}.tif", "train", "no_oil")
            for i in range(5)
        ]
        splitter = SceneSplitter(random_seed=42)
        result = splitter.split(scenes, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)

        # Each class should have at least 1 in each split
        for split_scenes in [result.train, result.validation, result.test]:
            oil_count = sum(1 for s in split_scenes if s.class_label == "oil")
            no_oil_count = sum(1 for s in split_scenes if s.class_label == "no_oil")
            assert oil_count >= 1
            assert no_oil_count >= 1

    def test_split_validation(self):
        """Test split ratio validation."""
        scenes = [
            SceneIndexItem(f"scene_{i}", f"/path/{i}.tif", f"/mask/{i}.tif", "train", "oil")
            for i in range(10)
        ]
        splitter = SceneSplitter(random_seed=42)
        with pytest.raises(ValueError):
            splitter.split(scenes, train_ratio=0.5, validation_ratio=0.5, test_ratio=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])