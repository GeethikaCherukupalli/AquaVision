# -*- coding: utf-8 -*-
"""Tests for U-Net input/output shapes."""

import pytest
import numpy as np
import torch
from ml.unet.model import UNet


class TestUNetIO:
    """Tests for U-Net input/output shapes."""

    def test_unet_input_shape(self):
        """Test U-Net accepts (2, 512, 512) input."""
        model = UNet(input_channels=2, output_channels=1, base_channels=32)
        x = torch.randn(1, 2, 512, 512)
        output = model(x)
        assert output.shape == (1, 1, 512, 512)

    def test_unet_batch_shape(self):
        """Test U-Net handles batch dimension."""
        model = UNet(input_channels=2, output_channels=1, base_channels=32)
        x = torch.randn(4, 2, 512, 512)
        output = model(x)
        assert output.shape == (4, 1, 512, 512)

    def test_unet_requires_2_channels(self):
        """Test U-Net requires 2 input channels."""
        with pytest.raises(ValueError):
            UNet(input_channels=3, output_channels=1, base_channels=32)

    def test_unet_output_single_channel(self):
        """Test U-Net produces single output channel."""
        model = UNet(input_channels=2, output_channels=1, base_channels=32)
        x = torch.randn(1, 2, 512, 512)
        output = model(x)
        assert output.shape[1] == 1

    def test_unet_encoder_decoder_symmetry(self):
        """Test encoder-decoder symmetry preserves spatial dimensions."""
        model = UNet(input_channels=2, output_channels=1, base_channels=32)
        x = torch.randn(1, 2, 512, 512)
        output = model(x)
        assert output.shape[-2:] == x.shape[-2:]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])