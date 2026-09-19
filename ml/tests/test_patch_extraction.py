# -*- coding: utf-8 -*-
"""Tests for patch extraction."""

import pytest
import numpy as np
from ml.preprocessing.patching import PatchExtractor, PatchGenerator, Patch


class TestPatchExtractor:
    """Tests for PatchExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PatchExtractor(patch_size=512)
        self.image = np.random.randn(2, 2048, 2048).astype(np.float32)
        self.mask = np.zeros((2048, 2048), dtype=np.uint8)
        self.mask[100:200, 100:200] = 1

    def test_extract_16_patches(self):
        """Test extracting 16 patches from 2048x2048."""
        patches = self.extractor.extract(self.image, self.mask, "test_scene", "train")
        assert len(patches) == 16
        assert all(p.image.shape == (2, 512, 512) for p in patches)
        assert all(p.row in [0, 512, 1024, 1536] for p in patches)
        assert all(p.col in [0, 512, 1024, 1536] for p in patches)

    def test_oil_fraction_calculation(self):
        """Test oil fraction is calculated correctly."""
        patches = self.extractor.extract(self.image, self.mask, "test_scene", "train")

        # The patch at row=0, col=0 contains the oil
        oil_patch = next(p for p in patches if p.row == 0 and p.col == 0)
        expected_fraction = (100 * 100) / (512 * 512)
        assert abs(oil_patch.oil_fraction - expected_fraction) < 1e-4

        # Label should be 1 (oil_fraction >= 0.01)
        assert oil_patch.label == 1

        # Other patches should have oil_fraction 0 and label 0
        non_oil_patches = [p for p in patches if p.row != 0 or p.col != 0]
        for p in non_oil_patches:
            assert p.oil_fraction == 0.0
            assert p.label == 0

    def test_patch_spatial_metadata(self):
        """Test patch preserves spatial metadata."""
        patches = self.extractor.extract(self.image, self.mask, "scene_001", "validation")
        for patch in patches:
            assert patch.scene_id == "scene_001"
            assert patch.split == "validation"
            assert patch.row >= 0
            assert patch.col >= 0
            assert patch.patch_size == 512


class TestPatchGenerator:
    """Tests for PatchGenerator."""

    def test_generate_patches(self):
        """Test PatchGenerator interface."""
        generator = PatchGenerator(patch_size=512)
        image = np.random.randn(2, 2048, 2048).astype(np.float32)
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        patches = generator.generate_patches(image, mask, "test_scene", "train")
        assert len(patches) == 16


class TestPatchDataclass:
    """Tests for Patch dataclass."""

    def test_patch_creation(self):
        """Test Patch dataclass creation."""
        image = np.random.randn(2, 512, 512).astype(np.float32)
        mask = np.zeros((512, 512), dtype=np.uint8)
        patch = Patch(
            image=image,
            mask=mask,
            row=512,
            col=1024,
            scene_id="test",
            split="train",
            oil_fraction=0.5,
            label=1,
        )
        assert patch.patch_size == 512
        assert patch.image.shape == (2, 512, 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])