# -*- coding: utf-8 -*-
"""ResNet18 inference for Stage 1 candidate detection."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .model import ResNet18Gate
from ..preprocessing.normalization import ChannelNormalizer
from ..preprocessing.patching import PatchExtractor, Patch


class ResNetInference:
    """ResNet18 inference for candidate patch detection."""

    def __init__(
        self,
        checkpoint_path: str,
        normalizer: Optional[ChannelNormalizer] = None,
        classifier_threshold: float = 0.12,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classifier_threshold = classifier_threshold

        self.model = ResNet18Gate(pretrained=False).to(self.device)
        # The committed checkpoint contains trusted optimizer/history metadata
        # from training and requires the explicit legacy loader mode in PyTorch
        # 2.6+; the model architecture and weights are unchanged.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint["model_state_dict"]
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.normalizer = normalizer or ChannelNormalizer()
        self.patch_extractor = PatchExtractor(patch_size=512)

    def predict(
        self,
        scene: np.ndarray,
    ) -> List[Patch]:
        """Run ResNet18 inference on a full scene.

        Parameters
        ----------
        scene : np.ndarray
            SAR image of shape (2, H, W) in dB.

        Returns
        -------
        list of Patch
            Candidate patches that passed the gate.
        """
        candidate_patches = self.predict_all(scene)
        return [patch for patch in candidate_patches if patch.is_candidate]

    def predict_all(self, scene: np.ndarray) -> List[Patch]:
        """Run inference and return every patch with its probability."""
        patches = self.patch_extractor.extract(scene, scene_id="scene", split="inference")

        for patch in patches:
            image = patch.image.astype(np.float32)
            normalized = self.normalizer.normalize(image)
            tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logit = self.model(tensor)
                probability = torch.sigmoid(logit).item()

            patch.probability = probability
            patch.is_candidate = probability >= self.classifier_threshold
        return patches

    @torch.no_grad()
    def predict_batch(
        self,
        patches: List[Patch],
    ) -> List[Patch]:
        """Run inference on a batch of patches."""
        for patch in patches:
            image = patch.image.astype(np.float32)
            normalized = self.normalizer.normalize(image)
            tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

            logit = self.model(tensor)
            probability = torch.sigmoid(logit).item()
            patch.probability = probability
            patch.is_candidate = probability >= self.classifier_threshold

        return [p for p in patches if p.is_candidate]

    @staticmethod
    def _tensor_to_probability(logit: torch.Tensor) -> float:
        """Convert a logit to a probability."""
        return torch.sigmoid(logit).item()