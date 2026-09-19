# AquaVision

AquaVision is an operational prototype for Sentinel-1 SAR oil-spill candidate detection. The current verified vertical slice is the ResNet18 v1 candidate gate; later segmentation, drift, and AIS modules remain separate prototype stages.

## Overview

The repository contains one authoritative backend under `aquavision-backend/` and one authoritative dashboard under `aquavision-frontend/`.

## Architecture summary

```text
Sentinel-1 VV/VH scene
  -> 512x512 SAR validation and training normalization
  -> ResNet18 v1 sigmoid probability
  -> threshold 0.84
  -> structured FastAPI result
  -> React dashboard status
```

## Backend

```bash
cd AquaVision/aquavision-backend
python -m venv .venv
. .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r ..\requirements.txt
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The backend is Python/FastAPI and has no `package.json`. Do not run `npm run dev`
from `aquavision-backend`; use the command above or `run.ps1`.

The backend exposes health, readiness, CDSE catalogue search, and Stage-1 analysis through FastAPI.

## Frontend

```bash
cd AquaVision/aquavision-frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## ML setup

The ML code lives under the `ml` package. The system uses:
- ResNet18 binary gate for candidate detection
- U-Net segmentation for pixel-level mask generation
- 2-channel SAR VV/VH input
- 512x512 patch processing
- scene-level train/validation/test split with Part3 held out

### Model artifact placement

The committed production candidate artifact is:

```text
ml/artifacts/resnet18/v1/resnet18_best_epoch06.pth
ml/artifacts/resnet18/v1/resnet18_metadata.json
ml/artifacts/resnet18/v1/sar_normalization.json
```

It expects a 2-band, 512x512 Sentinel-1-compatible scene in dB representation, ordered `VV`, `VH`. The trained normalization is loaded from metadata: VV mean/std `-32.184457 / 7.279350`, VH mean/std `-20.073348 / 5.797458`.

## Environment variables

Copy `.env.example` to `.env` and fill values as needed.

CDSE production search uses:
- `CDSE_CLIENT_ID`
- `CDSE_CLIENT_SECRET`
- `CDSE_TOKEN_URL`, `CDSE_CATALOG_URL`, `CDSE_PROCESS_URL`, `CDSE_COLLECTION`
- `CDSE_BACKSCATTER_COEFFICIENT` (required; must be confirmed from training provenance)
- `COPERNICUS_MARINE_USERNAME`
- `COPERNICUS_MARINE_PASSWORD`
- `AIS_USERNAME`
- `AIS_PASSWORD`

Analysis requires a real CDSE acquisition and a processed VV/VH scene. Missing credentials or data produce an explicit unavailable/error state.

## Sentinel-1 ingestion

The CDSE adapter performs client-credential authentication and STAC catalog search through `POST /api/v1/satellites/search`. It does not fabricate products when credentials or the provider are unavailable. Monitoring is acquisition/revisit based, not continuous.

Selected products are requested from `https://sh.dataspace.copernicus.eu/process/v1` as numerical two-band `FLOAT32` GeoTIFF output using the `sentinel-1-grd` collection, VV/VH bands, and dB units. The Processing API time range is a one-minute window around the CDSE-confirmed acquisition timestamp. The coefficient is not guessed because the repository does not record whether training used sigma0, beta0, gamma0, or gamma0 terrain.

## Stage 1

The production API validates the selected AOI, dates, and acquisition identifier against CDSE, requests the selected acquisition window from the Processing API, loads ResNet18 v1, applies the saved normalization, computes `sigmoid(logit)`, and returns `candidate = probability >= 0.84`. Missing input, model, or preprocessing errors return an analysis failure, not a no-candidate result.

## Future stages

Stage 2 OpenDrift reconstruction and Stage 3 AIS correlation are not connected to the active application and are not shown as completed analysis.

## API

```text
POST /api/v1/analysis          Run production Stage-1 analysis
GET  /api/v1/analysis/{job_id} Retrieve an analysis job
POST /api/v1/satellites/search Search Sentinel-1 products through CDSE
GET  /api/v1/readiness         Report artifact/provider readiness
```

Production example:

```json
{"mode":"historical","execution_mode":"production","scene_path":"data/fixtures/scene.tif","sensor":"sentinel-1"}
```

## Running an analysis

```bash
cd aquavision-backend
$env:PYTHONPATH=".."
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Use the frontend to draw an AOI, select dates, search CDSE acquisitions, select a returned product, and run Stage 1.

## Testing

```bash
python -m pytest -q
cd aquavision-frontend
npm run build
```

The ResNet tests require the project environment with `torch`, `torchvision`, and `pytest` installed. The committed checkpoint is not a fabricated fixture; it is the v1 candidate artifact.

## Implemented and not implemented

- Implemented: AOI rectangle selection, date range selection, CDSE catalogue search and revalidation, CDSE Processing API numerical VV/VH request, dB/unit validation, ResNet18 v1 inference, actual raster previews, and visible result/error states.
- Blocked pending provenance: the training coefficient (sigma0, beta0, gamma0, or gamma0 terrain) is not recorded, so `CDSE_BACKSCATTER_COEFFICIENT` is required and no default is guessed.
- The current environment used for this handoff did not contain `torch`/`pytest`, so checkpoint execution still needs to be run in the project environment.
- No authenticated CDSE smoke test was performed because credentials were unavailable.
- U-Net production integration, OpenDrift production integration, AIS attribution, and super-resolution are not part of this vertical slice.
- This is a prototype and requires domain validation before real operational use.
