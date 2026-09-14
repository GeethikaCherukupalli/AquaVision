# -*- coding: utf-8 -*-
"""Tests for spill geometry extraction."""

import pytest
import numpy as np
from ml.postprocessing.geometry import SpillGeometryExtractor, SpillGeometry


class TestSpillGeometryExtractor:
    """Tests for SpillGeometryExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = SpillGeometryExtractor()

    def test_extract_empty_mask(self):
        """Test geometry extraction on empty mask."""
        mask = np.zeros((512, 512), dtype=np.uint8)
        geometry = self.extractor.extract(mask)
        assert geometry.pixel_count == 0
        assert geometry.area_km2 == 0.0
        assert geometry.centroid_lat is None
        assert geometry.centroid_lon is None

    def test_extract_nonempty_mask(self):
        """Test geometry extraction on non-empty mask."""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 1
        geometry = self.extractor.extract(mask)
        assert geometry.pixel_count == 10000
        assert geometry.component_count >= 1
        assert geometry.orientation_deg >= 0

    def test_geospatial_metrics_no_transform(self):
        """Test geospatial metrics without transform returns None values."""
        mask = np.ones((512, 512), dtype=np.uint8)
        geometry = self.extractor.extract(mask)
        # Without transform/CRS, area_km2 and centroid should be None
        assert geometry.area_km2 is None
        assert geometry.perimeter_km is None
        assert geometry.centroid_lat is None
        assert geometry.centroid_lon is None

    def test_bounding_box_keys(self):
        """Test bounding box dictionary structure."""
        mask = np.ones((512, 512), dtype=np.uint8)
        geometry = self.extractor.extract(mask)
        bbox = geometry.bounding_box
        if bbox:
            assert isinstance(bbox, dict)
            assert "min_lat" in bbox or "min_row" in bbox

    def test_orientation_range(self):
        """Test orientation is in valid range [0, 180)."""
        mask = np.random.randn(512, 512) > 0.5
        mask = mask.astype(np.uint8)
        geometry = self.extractor.extract(mask)
        assert 0 <= geometry.orientation_deg < 180

    def test_spill_geometry_dataclass(self):
        """Test SpillGeometry dataclass."""
        geometry = SpillGeometry(
            area_km2=5.5,
            perimeter_km=10.2,
            centroid_lat=35.5,
            centroid_lon=-75.3,
            bounding_box={
                "min_lat": 35.0,
                "max_lat": 36.0,
                "min_lon": -76.0,
                "max_lon": -74.0,
            },
            length_km=5.0,
            width_km=3.0,
            orientation_deg=45.0,
            compactness=0.7,
            component_count=1,
            pixel_count=10000,
        )
        assert geometry.area_km2 == 5.5
        assert geometry.centroid_lat == 35.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])