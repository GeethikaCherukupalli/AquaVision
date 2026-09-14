# -*- coding: utf-8 -*-
"""U-Net segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net for pixel-level spill segmentation.

    Encoder:
        2 → 32 → 64 → 128
    Bottleneck:
        128 → 256
    Decoder:
        skip connections
    Output:
        1-channel segmentation probability mask
    """

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("U-Net currently expects exactly 2 input channels")
        if output_channels != 1:
            raise ValueError("U-Net currently expects a single output channel")

        self.encoder1 = DoubleConv(input_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 2, 512, 512).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, 1, 512, 512).
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)