# -*- coding: utf-8 -*-
"""Tests for ResNet input shape and architecture."""

import pytest
import numpy as np
import torch
from ml.resnet.model import ResNet18Gate


class TestResNetInput:
    """Tests for ResNet18 input shape."""

    def test_resnet_input_shape(self):
        """Test ResNet18 accepts (2, 512, 512) input."""
        model = ResNet18Gate(input_channels=2, pretrained=False)
        x = torch.randn(1, 2, 512, 512)
        logits = model(x)
        assert logits.shape == (1, 1)

    def test_resnet_batch_shape(self):
        """Test ResNet18 handles batch dimension."""
        model = ResNet18Gate(input_channels=2, pretrained=False)
        x = torch.randn(4, 2, 512, 512)
        logits = model(x)
        assert logits.shape == (4, 1)

    def test_resnet_requires_2_channels(self):
        """Test ResNet18 requires 2 input channels."""
        model = ResNet18Gate(input_channels=2, pretrained=False)
        with pytest.raises(ValueError):
            ResNet18Gate(input_channels=3, pretrained=False)

    def test_resnet_first_conv(self):
        """Test first convolution has 2 input channels."""
        model = ResNet18Gate(input_channels=2, pretrained=False)
        assert model.conv1.in_channels == 2
        assert model.conv1.out_channels == 64

    def test_resnet_final_fc(self):
        """Test final fully connected layer."""
        model = ResNet18Gate(input_channels=2, pretrained=False)
        assert model.fc.in_features == 512
        assert model.fc.out_features == 1

    def test_resnet_pretrained_weight_init(self):
        """Test pretrained weight initialization."""
        # Just verify it doesn't crash
        model = ResNet18Gate(input_channels=2, pretrained=True)
        x = torch.randn(1, 2, 512, 512)
        logits = model(x)
        assert logits.shape == (1, 1)

    def test_resnet_output_logits(self):
        """Test model outputs logits, not probabilities."""
        model = ResNet18Gate(input_channels=2, pretrained=False)
        x = torch.randn(1, 2, 512, 512)
        logits = model(x)
        # Logits should be unconstrained (not sigmoid)
        assert not (logits >= 0).all() and not (logits <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])