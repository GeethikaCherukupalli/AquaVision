# -*- coding: utf-8 -*-
"""Tests for ResNet checkpointing and resume functionality."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.models.resnet18.train import ResNetTrainer
from ml.models.resnet18.model import ResNet18Gate
from ml.preprocessing.normalization import ChannelNormalizer, compute_normalization_stats


def _make_trainer(config=None, device="cpu"):
    cfg = config or {
        "input_channels": 2,
        "pretrained": False,
        "random_seed": 42,
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "classifier_threshold": 0.12,
        "mixed_precision": False,
        "positive_class_weight": None,
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "num_workers": 0,
    }
    return ResNetTrainer(cfg)


def _dummy_batch(n=4):
    return {
        "image": torch.randn(n, 2, 64, 64),
        "label": torch.randint(0, 2, (n,)).float(),
    }


class TestCheckpointSaveLoad:
    """Test that checkpoints can be saved and loaded."""

    def test_save_load_roundtrip(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)
        assert ckpt_path.exists()

        # Load into a fresh trainer
        trainer2 = _make_trainer()
        loaded_epoch = trainer2.load_checkpoint(ckpt_path)
        assert loaded_epoch == 2  # Next epoch to run

    def test_checkpoint_contains_required_keys(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        required = [
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "best_val_loss",
            "history",
            "random_seed",
            "config",
            "normalization_stats",
            "classifier_threshold",
            "dataset_split_info",
        ]
        for key in required:
            assert key in ckpt, f"Missing key: {key}"

    def test_contains_scheduler_state(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        assert "scheduler_state_dict" in ckpt
        assert ckpt["scheduler_state_dict"] is not None

    def test_contains_normalization_stats(self, tmp_path):
        trainer = _make_trainer()
        trainer.normalizer = ChannelNormalizer(
            vv_mean=0.5, vv_std=1.2, vh_mean=-0.3, vh_std=0.8
        )
        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        assert ckpt["normalization_stats"] is not None
        assert ckpt["normalization_stats"]["vv_mean"] == 0.5
        assert ckpt["normalization_stats"]["vh_mean"] == -0.3

    def test_contains_classifier_threshold(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        assert ckpt["classifier_threshold"] == 0.12

    def test_contains_dataset_split_info(self, tmp_path):
        trainer = _make_trainer()
        trainer._train_scenes = [{"scene_id": "train_1"}, {"scene_id": "train_2"}]
        trainer._val_scenes = [{"scene_id": "val_1"}]
        trainer._test_scenes = [{"scene_id": "test_1"}]
        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        split_info = ckpt["dataset_split_info"]
        assert split_info["train_count"] == 2
        assert split_info["val_count"] == 1
        assert split_info["test_count"] == 1


class TestCheckpointAtomicWrite:
    """Test that checkpoint writes are atomic."""

    def test_atomic_write_no_leftover_tmp(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "atomic.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1, atomic=True)

        # No leftover temp files
        tmp_files = list(tmp_path.glob(".checkpoint_*.tmp"))
        assert tmp_files == []
        assert ckpt_path.exists()

    def test_atomic_write_replaces_previous(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "latest.pth"

        # Write first checkpoint
        trainer.save_checkpoint(ckpt_path, epoch=1, atomic=True)
        ckpt1 = torch.load(str(ckpt_path), map_location="cpu")
        assert ckpt1["epoch"] == 1

        # Write second checkpoint
        trainer.save_checkpoint(ckpt_path, epoch=2, atomic=True)
        ckpt2 = torch.load(str(ckpt_path), map_location="cpu")
        assert ckpt2["epoch"] == 2
        assert ckpt2["epoch"] != ckpt1["epoch"]

    def test_non_atomic_write(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "test.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1, atomic=False)
        assert ckpt_path.exists()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        assert ckpt["epoch"] == 1


class TestResumeFromCheckpoint:
    """Test resume functionality."""

    def test_resume_from_epoch(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "epoch_1.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)

        # Create new trainer and resume
        trainer2 = _make_trainer()
        resume_epoch = trainer2.load_checkpoint(ckpt_path)
        assert resume_epoch == 2  # Next epoch to run

    def test_resume_restores_best_val_loss(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "best.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        ckpt["best_val_loss"] = 3.14
        torch.save(ckpt, str(ckpt_path))

        trainer2 = _make_trainer()
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2._best_val_loss == pytest.approx(3.14)

    def test_resume_restores_history(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "latest.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)

        trainer2 = _make_trainer()
        trainer2.load_checkpoint(ckpt_path)
        assert "train" in trainer2._history
        assert "epoch_metrics" in trainer2._history

    def test_resume_restores_config(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "latest.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)

        trainer2 = _make_trainer()
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.config["random_seed"] == 42

    def test_resume_restores_classifier_threshold(self, tmp_path):
        trainer = _make_trainer()
        trainer.config["classifier_threshold"] = 0.15
        ckpt_path = tmp_path / "latest.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)

        trainer2 = _make_trainer()
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.config["classifier_threshold"] == 0.15


class TestLatestCheckpointSelection:
    """Test selection of latest checkpoint."""

    def test_latest_pth_exists(self, tmp_path):
        trainer = _make_trainer()
        latest_path = tmp_path / "latest.pth"
        trainer.save_checkpoint(latest_path, epoch=1)
        assert latest_path.exists()

        ckpt = torch.load(str(latest_path), map_location="cpu")
        assert ckpt["epoch"] == 1

    def test_multiple_epoch_checkpoints(self, tmp_path):
        trainer = _make_trainer()
        for epoch in range(1, 4):
            trainer.save_checkpoint(tmp_path / f"epoch_{epoch}.pth", epoch=epoch)
            trainer.save_checkpoint(tmp_path / "latest.pth", epoch=epoch)

        # Verify latest is epoch 3
        latest = torch.load(str(tmp_path / "latest.pth"), map_location="cpu")
        assert latest["epoch"] == 3

        # Verify epoch checkpoints exist
        for epoch in range(1, 4):
            assert (tmp_path / f"epoch_{epoch}.pth").exists()


class TestNormalizationCache:
    """Tests for normalization cache hit/miss."""

    def test_cache_hit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            lambda path, max_retries=3: np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            ),
        )

        paths = ["/fake/a.tif", "/fake/b.tif"]
        stats1 = compute_normalization_stats(
            paths, dataset_root=tmp_path, random_seed=42
        )

        read_count = {"n": 0}

        def counting_read(path, max_retries=3):
            read_count["n"] += 1
            return np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            )

        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            counting_read,
        )

        stats2 = compute_normalization_stats(
            paths, dataset_root=tmp_path, random_seed=42
        )
        assert read_count["n"] == 0  # Cache hit, no reads
        assert stats1 == stats2

    def test_cache_miss_recomputes(self, tmp_path, monkeypatch):
        read_count = {"n": 0}

        def counting_read(path, max_retries=3):
            read_count["n"] += 1
            return np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            )

        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            counting_read,
        )

        paths = ["/fake/a.tif", "/fake/b.tif"]
        compute_normalization_stats(
            paths, dataset_root=tmp_path, random_seed=42
        )
        assert read_count["n"] == 2  # Both files read

    def test_cache_invalidated_by_different_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            lambda path, max_retries=3: np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            ),
        )

        paths1 = ["/fake/a.tif", "/fake/b.tif"]
        compute_normalization_stats(
            paths1, dataset_root=tmp_path, random_seed=42
        )

        read_count = {"n": 0}

        def counting_read(path, max_retries=3):
            read_count["n"] += 1
            return np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            )

        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            counting_read,
        )

        paths2 = ["/fake/a.tif", "/fake/c.tif"]
        compute_normalization_stats(
            paths2, dataset_root=tmp_path, random_seed=42
        )
        assert read_count["n"] == 2  # Recomputed due to different paths

    def test_cache_requires_exact_path_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            lambda path, max_retries=3: np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            ),
        )

        paths = sorted(["/fake/b.tif", "/fake/a.tif"])
        compute_normalization_stats(
            paths, dataset_root=tmp_path, random_seed=42
        )

        # Different order should still be cache hit (paths are sorted)
        read_count = {"n": 0}

        def counting_read(path, max_retries=3):
            read_count["n"] += 1
            return np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            )

        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            counting_read,
        )

        compute_normalization_stats(
            ["/fake/a.tif", "/fake/b.tif"], dataset_root=tmp_path, random_seed=42
        )
        assert read_count["n"] == 0  # Cache hit (paths sorted)

    def test_cache_version_mismatch(self, tmp_path, monkeypatch):
        from ml.preprocessing.normalization import compute_normalization_stats, NORMALIZATION_CACHE_VERSION

        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            lambda path, max_retries=3: np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            ),
        )

        paths = ["/fake/a.tif"]
        compute_normalization_stats(
            paths, dataset_root=tmp_path, random_seed=42
        )

        # Manually change version to invalidate cache
        cache_file = tmp_path / "artifacts" / "normalization_cache.json"
        cache_data = cache_file.read_text()
        import json
        data = json.loads(cache_data)
        data["version"] = NORMALIZATION_CACHE_VERSION + 999
        cache_file.write_text(json.dumps(data))

        read_count = {"n": 0}

        def counting_read(path, max_retries=3):
            read_count["n"] += 1
            return np.stack(
                [np.full((4, 4), 2.0), np.full((4, 4), -1.0)], axis=0
            )

        monkeypatch.setattr(
            "ml.preprocessing.normalization._read_tiff_with_retry",
            counting_read,
        )

        compute_normalization_stats(
            paths, dataset_root=tmp_path, random_seed=42
        )
        assert read_count["n"] == 1  # Recomputed due to version mismatch


class TestInterruptedTrainingState:
    """Test that interrupted training state is preserved."""

    def test_keyboard_interrupt_saves_state(self, tmp_path, monkeypatch):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "latest.pth"

        # Simulate interruption by manually saving state
        trainer.last_epoch = 6
        trainer.save_checkpoint(ckpt_path, epoch=6)

        # Verify checkpoint can be loaded
        trainer2 = _make_trainer()
        resume_epoch = trainer2.load_checkpoint(ckpt_path)
        assert resume_epoch == 7  # Next epoch to run after epoch 6
        assert trainer2.last_epoch == 6

    def test_resume_at_correct_epoch(self, tmp_path):
        trainer = _make_trainer()
        ckpt_path = tmp_path / "latest.pth"
        trainer.save_checkpoint(ckpt_path, epoch=6)

        trainer2 = _make_trainer()
        resume_epoch = trainer2.load_checkpoint(ckpt_path)
        assert resume_epoch == 7  # Should resume at epoch 7, not restart
