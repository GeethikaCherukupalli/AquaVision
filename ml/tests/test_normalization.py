"""Tests for compute_normalization_stats: retry-on-failure and caching.

Uses small synthetic in-memory arrays via a monkeypatched read_tiff, so
no real TIFF files or dataset are required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aquavision.preprocessing.io as io_module
from aquavision.preprocessing.normalization import compute_normalization_stats


def fake_image():
    # (2, H, W) VV/VH array with known, simple statistics.
    vv = np.full((4, 4), 2.0, dtype=np.float32)
    vh = np.full((4, 4), -1.0, dtype=np.float32)
    return np.stack([vv, vh], axis=0)


def test_computes_correct_stats_for_uniform_images(monkeypatch):
    monkeypatch.setattr(io_module, "read_tiff", lambda path: fake_image())
    vv_mean, vv_std, vh_mean, vh_std = compute_normalization_stats(
        ["/fake/a.tif", "/fake/b.tif"]
    )
    assert vv_mean == pytest.approx(2.0)
    assert vh_mean == pytest.approx(-1.0)
    # Uniform images -> zero variance -> std floored to 1.0
    assert vv_std == pytest.approx(1.0)
    assert vh_std == pytest.approx(1.0)


def test_retries_up_to_three_times_then_succeeds(monkeypatch):
    """Fails twice, succeeds on the 3rd attempt -> should NOT raise,
    and each attempt must call read_tiff again (i.e. reopen the file)."""
    calls = {"n": 0}

    def flaky_read(path):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError(f"simulated transient Drive read failure #{calls['n']}")
        return fake_image()

    monkeypatch.setattr(io_module, "read_tiff", flaky_read)

    vv_mean, vv_std, vh_mean, vh_std = compute_normalization_stats(["/fake/flaky.tif"])

    assert calls["n"] == 3  # 2 failures + 1 success = exactly 3 reopen attempts
    assert vv_mean == pytest.approx(2.0)


def test_raises_after_exhausting_retries_and_does_not_skip_the_file(monkeypatch):
    calls = {"n": 0}

    def always_fails(path):
        calls["n"] += 1
        raise OSError("permanent read failure")

    monkeypatch.setattr(io_module, "read_tiff", always_fails)

    with pytest.raises(OSError, match="permanent read failure"):
        compute_normalization_stats(["/fake/broken.tif"])

    assert calls["n"] == 3  # exactly max_retries attempts, no silent skip


def test_reopens_on_every_retry_not_just_first_attempt(monkeypatch):
    """Each retry must be a fresh call to read_tiff (a fresh open),
    not a reuse of a stale handle."""
    seen_paths = []

    def flaky_read(path):
        seen_paths.append(path)
        if len(seen_paths) < 3:
            raise OSError("transient")
        return fake_image()

    monkeypatch.setattr(io_module, "read_tiff", flaky_read)
    compute_normalization_stats(["/fake/x.tif"])

    assert seen_paths == ["/fake/x.tif", "/fake/x.tif", "/fake/x.tif"]


def test_cache_hit_avoids_rescanning_tiffs(tmp_path, monkeypatch):
    read_count = {"n": 0}

    def counting_read(path):
        read_count["n"] += 1
        return fake_image()

    monkeypatch.setattr(io_module, "read_tiff", counting_read)

    paths = ["/fake/a.tif", "/fake/b.tif", "/fake/c.tif"]

    # First call: cache miss, must read every file.
    stats1 = compute_normalization_stats(
        paths, dataset_root=tmp_path, random_seed=42
    )
    assert read_count["n"] == 3
    cache_file = tmp_path / "artifacts" / "normalization_cache.json"
    assert cache_file.exists()

    # Second call, identical paths + seed: cache hit, zero new reads.
    stats2 = compute_normalization_stats(
        paths, dataset_root=tmp_path, random_seed=42
    )
    assert read_count["n"] == 3  # unchanged - no rescanning
    assert stats1 == stats2


def test_cache_miss_when_training_paths_change(tmp_path, monkeypatch):
    monkeypatch.setattr(io_module, "read_tiff", lambda path: fake_image())

    compute_normalization_stats(
        ["/fake/a.tif", "/fake/b.tif"], dataset_root=tmp_path, random_seed=42
    )
    read_count = {"n": 0}

    def counting_read(path):
        read_count["n"] += 1
        return fake_image()

    monkeypatch.setattr(io_module, "read_tiff", counting_read)

    # Different path list -> cache must be treated as stale.
    compute_normalization_stats(
        ["/fake/a.tif", "/fake/c.tif"], dataset_root=tmp_path, random_seed=42
    )
    assert read_count["n"] == 2  # recomputed, did not reuse stale cache


def test_cache_miss_when_random_seed_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(io_module, "read_tiff", lambda path: fake_image())
    paths = ["/fake/a.tif", "/fake/b.tif"]

    compute_normalization_stats(paths, dataset_root=tmp_path, random_seed=42)

    read_count = {"n": 0}

    def counting_read(path):
        read_count["n"] += 1
        return fake_image()

    monkeypatch.setattr(io_module, "read_tiff", counting_read)
    compute_normalization_stats(paths, dataset_root=tmp_path, random_seed=1)

    assert read_count["n"] == 2  # different seed -> cache invalidated


def test_no_caching_when_dataset_root_not_given(tmp_path, monkeypatch):
    """Backward compatible: omitting dataset_root must behave exactly
    like before this change (no cache file, no caching)."""
    monkeypatch.setattr(io_module, "read_tiff", lambda path: fake_image())
    compute_normalization_stats(["/fake/a.tif"])
    assert not (tmp_path / "artifacts" / "normalization_cache.json").exists()
