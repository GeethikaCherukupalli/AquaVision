"""Tests for SceneSplitter.

These use small synthetic SceneIndexItem objects only. No TIFF files
and no part of the real 90 GB dataset are required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aquavision.dataset.index import SceneIndexItem
from aquavision.dataset.splitter import SceneSplitter


def make_scenes():
    scenes = []
    # Part1: 20 oil scenes
    for i in range(20):
        scenes.append(SceneIndexItem(
            scene_id=f"Part1_oil_{i:04d}",
            image_path=f"/fake/part1/oil/{i}.tif",
            mask_path=f"/fake/part1/oil_mask/{i}.tif",
            split="train",
            class_label="oil",
            part="Part1",
        ))
    # Part2: 15 no_oil + 10 lookalike scenes
    for i in range(15):
        scenes.append(SceneIndexItem(
            scene_id=f"Part2_no_oil_{i:04d}",
            image_path=f"/fake/part2/no_oil/{i}.tif",
            mask_path=f"/fake/part2/no_oil_mask/{i}.tif",
            split="train",
            class_label="no_oil",
            part="Part2",
        ))
    for i in range(10):
        scenes.append(SceneIndexItem(
            scene_id=f"Part2_lookalike_{i:04d}",
            image_path=f"/fake/part2/lookalike/{i}.tif",
            mask_path=f"/fake/part2/lookalike_mask/{i}.tif",
            split="train",
            class_label="lookalike",
            part="Part2",
        ))
    # Part3: 12 held-out test scenes (mixed classes), already split="test"
    for i, cls in enumerate(["oil", "no_oil", "lookalike"] * 4):
        scenes.append(SceneIndexItem(
            scene_id=f"Part3_{cls}_{i:04d}",
            image_path=f"/fake/part3/{cls}/{i}.tif",
            mask_path=f"/fake/part3/{cls}_mask/{i}.tif",
            split="test",
            class_label=cls,
            part="Part3",
        ))
    return scenes


def test_split_counts_are_sane():
    scenes = make_scenes()
    split = SceneSplitter(random_seed=42).split(scenes)
    assert len(split.train) + len(split.validation) == 45
    assert len(split.test) == 12
    assert len(split.train) > 0
    assert len(split.validation) > 0


def test_every_scene_has_correct_split_field():
    """Regression test for the exact bug reported:
    scenes placed in split.validation must carry scene.split == 'validation',
    not the original discovery-time value of 'train'.
    """
    scenes = make_scenes()
    split = SceneSplitter(random_seed=42).split(scenes)

    assert all(s.split == "train" for s in split.train)
    assert all(s.split == "validation" for s in split.validation)
    assert all(s.split == "test" for s in split.test)


def test_no_overlap_between_partitions():
    scenes = make_scenes()
    split = SceneSplitter(random_seed=42).split(scenes)

    train_ids = {s.scene_id for s in split.train}
    val_ids = {s.scene_id for s in split.validation}
    test_ids = {s.scene_id for s in split.test}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_part3_is_test_only():
    scenes = make_scenes()
    split = SceneSplitter(random_seed=42).split(scenes)

    assert all(s.part == "Part3" for s in split.test)
    assert not any(s.part == "Part3" for s in split.train)
    assert not any(s.part == "Part3" for s in split.validation)


def test_split_is_deterministic_given_seed():
    scenes_a = make_scenes()
    scenes_b = make_scenes()

    split_a = SceneSplitter(random_seed=42).split(scenes_a)
    split_b = SceneSplitter(random_seed=42).split(scenes_b)

    ids_a = sorted(s.scene_id for s in split_a.validation)
    ids_b = sorted(s.scene_id for s in split_b.validation)
    assert ids_a == ids_b


def test_dataset_scenes_concat_has_correct_split_after_reassignment():
    """Mirrors what train_resnet() does:
    dataset.scenes = split.train + split.validation + split.test
    """
    scenes = make_scenes()
    split = SceneSplitter(random_seed=42).split(scenes)
    combined = split.train + split.validation + split.test

    val_from_combined = [s for s in combined if s.split == "validation"]
    assert len(val_from_combined) == len(split.validation)
    assert len(val_from_combined) > 0
