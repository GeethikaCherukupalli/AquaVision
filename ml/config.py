# -*- coding: utf-8 -*-
"""Configuration management for AquaVision ML Stage 1.

All paths and hyperparameters are configurable. Nothing related to the
dataset location, model paths, or thresholds is hardcoded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DatasetConfig:
    """Dataset paths and patch configuration."""

    dataset_root: str = os.getenv("AQUAVISION_DATA_ROOT", "")
    patch_size: int = 512
    oil_fraction_threshold: float = 0.01
    scene_size: int = 2048
    image_channels: int = 2
    image_dtype: str = "float32"
    mask_dtype: str = "uint8"
    part1_dir: str = "Part1"
    part2_dir: str = "Part2"
    part3_dir: str = "Part3"
    train_split: float = 0.80
    validation_split: float = 0.10
    test_split: float = 0.10
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.scene_size % self.patch_size != 0:
            raise ValueError("scene_size must be divisible by patch_size")
        if not (0.0 <= self.oil_fraction_threshold <= 1.0):
            raise ValueError("oil_fraction_threshold must be in [0, 1]")
        if self.train_split + self.validation_split + self.test_split != 1.0:
            raise ValueError("train/validation/test splits must sum to 1.0")


@dataclass
class NormalizationConfig:
    """Channel-wise SAR normalization configuration."""

    vv_mean: Optional[float] = None
    vv_std: Optional[float] = None
    vh_mean: Optional[float] = None
    vh_std: Optional[float] = None
    compute_from_training: bool = True

    def __post_init__(self) -> None:
        for name in ("vv_mean", "vv_std", "vh_mean", "vh_std"):
            value = getattr(self, name)
            if value is not None and value <= 0 and name.endswith("std"):
                raise ValueError(f"{name} must be positive when provided")


@dataclass
class ResNetConfig:
    """ResNet18 candidate gate configuration."""

    input_channels: int = 2
    input_size: int = 512
    pretrained: bool = True
    classifier_threshold: float = 0.12
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    random_seed: int = 42
    num_workers: int = 4
    mixed_precision: bool = True
    positive_class_weight: Optional[float] = None
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    checkpoint_dir: str = "artifacts/resnet18"

    def __post_init__(self) -> None:
        if self.input_channels != 2:
            raise ValueError("ResNet18 currently expects exactly 2 input channels (VV/VH)")
        if self.input_size != 512:
            raise ValueError("ResNet18 currently expects 512×512 patches")
        if not (0.0 <= self.classifier_threshold <= 1.0):
            raise ValueError("classifier_threshold must be in [0, 1]")


@dataclass
class UNetConfig:
    """U-Net segmentation configuration."""

    input_channels: int = 2
    input_size: int = 512
    output_channels: int = 1
    base_channels: int = 32
    segmentation_threshold: float = 0.5
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    random_seed: int = 42
    num_workers: int = 4
    mixed_precision: bool = True
    checkpoint_dir: str = "artifacts/unet"

    def __post_init__(self) -> None:
        if self.input_channels != 2:
            raise ValueError("U-Net currently expects exactly 2 input channels (VV/VH)")
        if self.output_channels != 1:
            raise ValueError("U-Net currently expects a single segmentation output channel")
        if not (0.0 <= self.segmentation_threshold <= 1.0):
            raise ValueError("segmentation_threshold must be in [0, 1]")


@dataclass
class InferenceConfig:
    """Stage 1 inference configuration."""

    patch_size: int = 512
    classifier_threshold: float = 0.12
    segmentation_threshold: float = 0.5
    min_component_size: int = 32
    max_component_size: Optional[int] = None
    use_morphology: bool = True
    morphology_kernel_size: int = 3
    output_dir: str = "artifacts/inference"
    device: str = "auto"

    def __post_init__(self) -> None:
        if not (0.0 <= self.classifier_threshold <= 1.0):
            raise ValueError("classifier_threshold must be in [0, 1]")
        if not (0.0 <= self.segmentation_threshold <= 1.0):
            raise ValueError("segmentation_threshold must be in [0, 1]")
        if self.min_component_size < 0:
            raise ValueError("min_component_size cannot be negative")


@dataclass
class Stage1Config:
    """Complete Stage 1 configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    resnet: ResNetConfig = field(default_factory=ResNetConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    random_seed: int = 42

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stage1Config":
        """Build configuration from a nested dictionary."""
        config = cls()
        for section, values in data.items():
            if not hasattr(config, section) or not isinstance(values, dict):
                continue
            section_obj = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
        if "random_seed" in data:
            config.random_seed = int(data["random_seed"])
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a nested dictionary."""
        return {
            "dataset": self.dataset.__dict__,
            "normalization": self.normalization.__dict__,
            "resnet": self.resnet.__dict__,
            "unet": self.unet.__dict__,
            "inference": self.inference.__dict__,
            "random_seed": self.random_seed,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "Stage1Config":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def get_stage1_config() -> Stage1Config:
    """Load configuration from environment and optional config file."""
    config = Stage1Config()
    config_path = os.getenv("AQUAVISION_CONFIG")
    if config_path and Path(config_path).exists():
        config = Stage1Config.load(config_path)
    return config
