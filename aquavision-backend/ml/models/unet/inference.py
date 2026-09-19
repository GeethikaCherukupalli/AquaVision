# -*- coding: utf-8 -*-
"""U-Net inference for Stage 1 segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .model import UNet
from ...preprocessing.normalization import ChannelNormalizer
from ...preprocessing.patching import Patch


class UNetInference:
    """U-Net inference for pixel-level spill segmentation."""

    def __init__(
        self,
        checkpoint_path: str,
        normalizer: Optional[ChannelNormalizer] = None,
        segmentation_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.segmentation_threshold = segmentation_threshold

        self.model = UNet().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.normalizer = normalizer or ChannelNormalizer()

    @torch.no_grad()
    def predict(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """Run U-Net inference on candidate patches.

        Parameters
        ----------
        patches : list of Patch
            Candidate patches from ResNet18.

        Returns
        -------
        list of Patch
            Patches with segmentation probability masks.
        """
        for patch in patches:
            image = patch.image.astype(np.float32)
            normalized = self.normalizer.normalize(image)
            tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)
            mask = (probabilities >= self.segmentation_threshold).float()

            patch.probability = probabilities.squeeze().cpu().numpy()
            patch.mask = mask.squeeze().cpu().numpy().astype(np.uint8)

        return patches