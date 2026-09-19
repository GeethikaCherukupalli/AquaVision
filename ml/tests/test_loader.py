"""Tests for LazyDataset's split-filtering behavior.

These exercise LazyDataset.__init__ only (scene selection), which is
exactly the code path that raised:
    ValueError: No scenes found for split 'validation'

No TIFF reads happen in __init__, so no real dataset files are needed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aquavision.dataset.index import DatasetIndex, SceneIndexItem
from aquavision.dataset.loader import LazyDataset
from aquavision.dataset.splitter import SceneSplitter


def build_index_with_scenes():
    idx = DatasetIndex(dataset_root="/fake/oil_spill")
    scenes = []
    for i in range(8):
        scenes.append(SceneIndexItem(
            scene_id=f"Part1_oil_{i:04d}",
            image_path=f"/fake/part1/oil/{i}.tif",
            mask_path=f"/fake/part1/oil_mask/{i}.tif",
            split="train",
            class_label="oil",
            part="Part1",
        ))
    for i in range(6):
        scenes.append(SceneIndexItem(
            scene_id=f"Part2_no_oil_{i:04d}",
            image_path=f"/fake/part2/no_oil/{i}.tif",
            mask_path=f"/fake/part2/no_oil_mask/{i}.tif",
            split="train",
            class_label="no_oil",
            part="Part2",
        ))
    for i in range(4):
        scenes.append(SceneIndexItem(
            scene_id=f"Part3_oil_{i:04d}",
            image_path=f"/fake/part3/oil/{i}.tif",
            mask_path=f"/fake/part3/oil_mask/{i}.tif",
            split="test",
            class_label="oil",
            part="Part3",
        ))
    idx.scenes = scenes
    return idx


def test_lazydataset_train_split_nonempty():
    idx = build_index_with_scenes()
    split = SceneSplitter(random_seed=42).split(idx.scenes)
    idx.scenes = split.train + split.validation + split.test

    train_ds = LazyDataset(dataset_index=idx, split="train")
    assert len(train_ds.scenes) > 0
    assert all(s.split == "train" for s in train_ds.scenes)


def test_lazydataset_validation_split_nonempty():
    """This is the exact regression test for the reported crash:
    before the fix, this raised
    ValueError: No scenes found for split 'validation'.
    """
    idx = build_index_with_scenes()
    split = SceneSplitter(random_seed=42).split(idx.scenes)
    idx.scenes = split.train + split.validation + split.test

    val_ds = LazyDataset(dataset_index=idx, split="validation")
    assert len(val_ds.scenes) > 0
    assert all(s.split == "validation" for s in val_ds.scenes)


def test_lazydataset_test_split_nonempty():
    idx = build_index_with_scenes()
    split = SceneSplitter(random_seed=42).split(idx.scenes)
    idx.scenes = split.train + split.validation + split.test

    test_ds = LazyDataset(dataset_index=idx, split="test")
    assert len(test_ds.scenes) > 0
    assert all(s.split == "test" for s in test_ds.scenes)
    assert all(s.part == "Part3" for s in test_ds.scenes)


def test_lazydataset_raises_for_truly_empty_split():
    """Sanity check that the ValueError still fires for a genuinely
    empty split (e.g. a typo'd split name), so we haven't silenced a
    real error condition while fixing the false-positive one."""
    idx = build_index_with_scenes()
    idx.scenes[0].split = "train"  # ensure at least one real split exists
    with pytest.raises(ValueError, match="No scenes found for split 'bogus'"):
        LazyDataset(dataset_index=idx, split="bogus")
