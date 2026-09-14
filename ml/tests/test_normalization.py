# -*- coding: utf-8 -*-
"""Tests for SAR normalization."""

import pytest
import numpy as np
from ml.preprocessing.normalization import ChannelNormalizer, compute_normalization_stats


class TestChannelNormalizer:
    """Tests for ChannelNormalizer."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        image = np.ones((2, 512, 512), dtype=np.float32) * 10.0
        normalizer = ChannelNormalizer(vv_mean=5.0, vv_std=2.0, vh_mean=0.0, vh_std=1.0)
        normalized = normalizer.normalize(image)
        assert np.allclose(normalized[0], (10.0 - 5.0) / 2.0)
        assert np.allclose(normalized[1], (10.0 - 0.0) / 1.0)

    def test_normalize_uses_image_stats(self):
        """Test normalization falls back to image statistics."""
        image = np.array([
            np.full((512, 512), 5.0, dtype=np.float32),
            np.full((512, 512), -10.0, dtype=np.float32),
        ])
        normalizer = ChannelNormalizer()
        normalized = normalizer.normalize(image)
        assert np.allclose(normalized[0], 0.0)  # mean-subtracted
        assert np.allclose(normalized[1], 0.0)

    def test_save_load(self, tmp_path):
        """Test save/load roundtrip."""
        normalizer = ChannelNormalizer(vv_mean=5.0, vv_std=2.0, vh_mean=-3.0, vh_std=1.5)
        path = tmp_path / "normalizer.json"
        normalizer.save(path)
        loaded = ChannelNormalizer.load(path)
        assert loaded.vv_mean == 5.0
        assert loaded.vv_std == 2.0
        assert loaded.vh_mean == -3.0
        assert loaded.vh_std == 1.5

    def test_invalid_std(self):
        """Test validation of standard deviation."""
        with pytest.raises(ValueError):
            ChannelNormalizer(vv_std=0.0)


class TestComputeNormalizationStats:
    """Tests for compute_normalization_stats."""

    def test_compute_stats(self, tmp_path):
        """Test computing stats from image files."""
        import rasterio

        # Create test TIFFs
        paths = []
        for i in range(3):
            path = tmp_path / f"test_{i}.tif"
            data = np.array([
                np.full((512, 512), 5.0 + i, dtype=np.float32),
                np.full((512, 512), -10.0 - i, dtype=np.float32),
            ])
            with rasterio.open(
                str(path), "w",
                driver="GTiff",
                height=512, width=512,
                count=2, dtype="float32",
            ) as dst:
                dst.write(data)
            paths.append(str(path))

        vv_mean, vv_std, vh_mean, vh_std = compute_normalization_stats(paths)
        # Mean of 5, 6, 7 = 6.0
        assert abs(vv_mean - 6.0) < 0.01
        # Mean of -10, -11, -12 = -11.0
        assert abs(vh_mean + 11.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])