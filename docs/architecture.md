# AquaVision architecture

## Core flow

1. Sentinel-1 acquisition and preprocessing
2. ResNet18 candidate gate
3. U-Net segmentation
4. Geometry extraction and artifact saving
5. Environmental reconstruction
6. AIS correlation and vessel ranking
7. Frontend operational dashboard

## Layer boundaries

### Data access
- Sentinel-1 provider
- Copernicus Marine provider
- Open-Meteo provider
- AIS provider

### Service layer
- Satellite service
- Ocean service
- Weather service
- Drift simulation service
- AIS service
- NZ scoring service

### ML layer
- dataset discovery
- scene splitting
- normalization
- patch extraction
- ResNet training/inference
- U-Net training/inference
- postprocessing and geometry

### Application layer
- FastAPI API
- background job orchestration
- artifact persistence
