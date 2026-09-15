# ML pipeline

## Stage 1

- Input: 2-channel SAR VV/VH tile
- Model gate: ResNet18 binary classifier
- Segmentation: U-Net
- Reconstruction: patch-to-scene join
- Geometry: connected components and geospatial geometry

## Training requirements

- scene-level split with Part3 held out
- 512x512 patch processing
- normalization statistics computed from training-only scenes
- checkpoint saving for latest, best, and final artifacts
