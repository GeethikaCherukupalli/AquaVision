# -*- coding: utf-8 -*-
"""Tests for segmentation postprocessing."""

import pytest
import numpy as np
from ml.postprocessing.postprocess import SpillPostprocessor, PostprocessResult


class TestSpillPostprocessor:
    """Tests for SpillPostprocessor."""

    def test_probability_thresholding(self):
        """Test probability-based binarization."""
        processor = SpillPostprocessor(
            probability_threshold=0.5,
            min_component_size=1,
        )
        prob_map = np.array([[0.0, 0.6, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        result = processor.process(prob_map)
        assert result.mask[1, 1] == 1
        assert result.mask[0, 0] == 0

    def test_min_component_size(self):
        """Test tiny component removal."""
        processor = SpillPostprocessor(
            probability_threshold=0.5,
            min_component_size=10,
        )
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1  # Single pixel = tiny component
        result = processor.process(prob_map=mask.astype(np.float32), binary_mask=mask)
        # Single pixel should be removed
        assert result.component_count == 0 or result.removed_components >= 1

    def test_mask_shape_validation(self):
        """Test mask shape validation."""
        processor = SpillPostprocessor()
        with pytest.raises(ValueError):
            processor.process(np.ones((3, 3, 3), dtype=np.float32))

    def test_postprocess_result(self):
        """Test PostprocessResult structure."""
        mask = np.ones((10, 10), dtype=np.uint8)
        processor = SpillPostprocessor(probability_threshold=0.5, min_component_size=1)
        result = processor.process(mask.astype(np.float32), binary_mask=mask)
        assert isinstance(result.mask, np.ndarray)
        assert isinstance(result.probability_map, np.ndarray)
        assert isinstance(result.removed_components, int)
        assert isinstance(result.component_count, int)

    def test_morphology_options(self):
        """Test morphological operations."""
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        for morph in ["none", "opening", "closing", "dilate", "erode"]:
            processor = SpillPostprocessor(
                probability_threshold=0.5,
                min_component_size=1,
                morphology=morph,
                kernel_size=3,
            )
            result = processor.process(mask.astype(np.float32), binary_mask=mask)
            assert result.mask.shape == mask.shape

    def test_invalid_morphology(self):
        """Test invalid morphology raises error."""
        with pytest.raises(ValueError):
            SpillPostprocessor(morphology="invalid")

    def test_invalid_kernel_size(self):
        """Test invalid kernel size raises error."""
        with pytest.raises(ValueError):
            SpillPostprocessor(kernel_size=2)

    def test_save_qualitative_examples(self, tmp_path):
        """Test saving qualitative examples."""
        from ml.postprocessing.postprocess import save_qualitative_examples
        examples = [
            {
                "scene_id": "test_001",
                "row": 0,
                "col": 0,
                "image": np.ones((2, 512, 512)).tolist(),
                "mask": np.ones((512, 512)).tolist(),
                "prediction": np.ones((512, 512)).tolist(),
            }
        ]
        save_qualitative_examples(examples, str(tmp_path))
        assert (tmp_path / "qualitative_examples.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])