# -*- coding: utf-8 -*-
"""Tests for patch-to-scene reconstruction."""

import pytest
import numpy as np
from ml.postprocessing.reconstruction import PatchSceneReconstructor, ReconstructionResult


class TestPatchSceneReconstructor:
    """Tests for PatchSceneReconstructor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reconstructor = PatchSceneReconstructor(
            scene_size=2048,
            patch_size=512,
        )

    def test_reconstruct_full_mask(self):
        """Test reconstruction from 16 patches."""
        patches = []
        for row in range(0, 2048, 512):
            for col in range(0, 2048, 512):
                mask = np.zeros((512, 512), dtype=np.uint8)
                # Put oil in top-left corner
                if row == 0 and col == 0:
                    mask[100:200, 100:200] = 1
                patches.append({
                    "row": row,
                    "col": col,
                    "mask": mask,
                    "probability": 0.5 if (row == 0 and col == 0) else 0.0,
                })

        result = self.reconstructor.reconstruct(patches)
        assert result.mask.shape == (2048, 2048)
        assert result.mask[0, 0] == 1  # Oil at top-left

    def test_patches_preserve_offset(self):
        """Test patches are placed at correct offsets."""
        patches = []
        # Only one patch with oil at bottom-right
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 1
        patches.append({
            "row": 1536,
            "col": 1536,
            "mask": mask,
            "probability": 0.9,
        })

        result = self.reconstructor.reconstruct(patches)
        # Check the oil is at the correct offset
        assert result.mask[1536 + 100, 1536 + 100] == 1

    def test_empty_patches(self):
        """Test reconstruction with empty patches."""
        patches = [{
            "row": 0,
            "col": 0,
            "mask": np.zeros((512, 512), dtype=np.uint8),
            "probability": 0.0,
        }]
        result = self.reconstructor.reconstruct(patches)
        assert result.mask.sum() == 0

    def test_raises_on_invalid_shape(self):
        """Test validation of patch shapes."""
        patches = [{
            "row": 0,
            "col": 0,
            "mask": np.zeros((256, 256), dtype=np.uint8),
            "probability": 0.0,
        }]
        with pytest.raises(ValueError):
            self.reconstructor.reconstruct(patches)

    def test_save_results(self, tmp_path):
        """Test saving reconstruction results."""
        patches = [{
            "row": 0,
            "col": 0,
            "mask": np.ones((512, 512), dtype=np.uint8),
            "probability": 0.5,
        }]
        result = self.reconstructor.reconstruct(patches)

        mask_path = tmp_path / "mask.tif"
        prob_path = tmp_path / "prob.tif"
        self.reconstructor.save_results(result, str(mask_path))
        assert mask_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])