# -*- coding: utf-8 -*-
"""AquaVision ML Stage 1 — Sentinel-1 Oil Spill Detection.

Stage 1 pipeline:
    Sentinel-1 SAR (VV/VH)
        ↓
    Preprocessing
        ↓
    512×512 patches
        ↓
    ResNet18 binary candidate gate
        ↓
    U-Net segmentation
        ↓
    Patch-to-scene reconstruction
        ↓
    Spill geometry characterization
"""

__version__ = "0.1.0"
__stage__ = "stage1"
