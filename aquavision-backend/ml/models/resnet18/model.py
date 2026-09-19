# -*- coding: utf-8 -*-
"""ResNet18 binary candidate gate."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ResNet18Gate(nn.Module):
    """ResNet18 binary candidate gate for oil spill detection.

    The gate answers: "Should this spatial patch be passed to the
    more expensive U-Net segmentation stage?"

    Input:
        2-channel VV/VH SAR patch of shape (2, 512, 512)
    Output:
        Single logit per patch (before sigmoid)
    """

    def __init__(
        self,
        input_channels: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(
                "ResNet18Gate currently expects exactly 2 input channels"
            )

        import torchvision.models as models

        self._pretrained = pretrained
        backbone = models.resnet18(weights="DEFAULT" if pretrained else None)

        # Modify first convolution for 2 input channels
        original_conv1 = backbone.conv1
        # Average RGB weights across channel dimension and repeat for 2 channels
        rgb_weights = original_conv1.weight.data  # (64, 3, 7, 7)
        avg_weights = rgb_weights.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        new_weights = avg_weights.repeat(1, 2, 1, 1)  # (64, 2, 7, 7)

        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data.copy_(new_weights)

        # Copy batch norm from original
        self.bn1 = backbone.bn1

        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, 2, 512, 512).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.parameters()).device