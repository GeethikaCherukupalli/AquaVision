# AquaVision

AquaVision is a prototype oil-spill detection and environmental reconstruction system. It combines SAR scene analysis, candidate detection, segmentation, probabilistic origin reconstruction, and AIS-based vessel correlation in a single operational workflow.

## Overview

The system is designed to ingest Sentinel-1 SAR scenes, run a two-stage ML pipeline, estimate a probable origin region and time window, and then correlate candidate vessel tracks with the inferred spill origin. The current repository is a working prototype with a deterministic demo path and a production-oriented API boundary.

## Architecture summary

```text
Sentinel-1 scene
  -> Stage 1: ResNet gate + U-Net segmentation
  -> Spill geometry + artifact exports
  -> Stage 2: environmental reconstruction (OpenDrift/OpenOil, mock or real)
  -> Stage 3: AIS candidate scoring
  -> Operational dashboard
```

## Backend

```bash
cd AquaVision
python -m venv .venv
. .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The backend exposes the analysis API, satellite product search, simulation status, and AIS candidates through FastAPI.

## Frontend

```bash
cd AquaVision/frontend
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

Place trained model files in a directory such as:

```text
artifacts/resnet18/latest.pth
artifacts/resnet18/best.pth
artifacts/resnet18/final.pth
artifacts/unet/best.pth
artifacts/unet/final.pth
```

## Environment variables

Copy `.env.example` to `.env` and fill values as needed.

Required for production connectors:
- `CDSE_USERNAME`
- `CDSE_PASSWORD`
- `COPERNICUS_MARINE_USERNAME`
- `COPERNICUS_MARINE_PASSWORD`
- `AIS_USERNAME`
- `AIS_PASSWORD`

The demo path does not require external credentials.

## Sentinel-1 ingestion

The repository includes a clean provider boundary for Sentinel-1 acquisition through the CDSE adapter. Production credentials are required for live retrieval. The system intentionally does not claim continuous satellite coverage.

## Stage 1

The Stage 1 pipeline includes:
- SAR input preprocessing
- patch extraction
- ResNet candidate gate
- U-Net segmentation
- scene reconstruction
- geometry extraction
- output artifact writes

## Stage 2

Environmental reconstruction uses a provider/service boundary for:
- ocean currents from Copernicus Marine
- weather from Open-Meteo
- OpenDrift/OpenOil hindcast and forecast simulation

If external data are unavailable, the system falls back to clearly labeled demo/mock behavior.

## Stage 3

The AIS stage filters vessel trajectories by region and time, computes candidate correlations, and assigns a score from 0–100.

## Running demo mode

```bash
cd AquaVision
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Then call the API:

```bash
curl -X POST http://localhost:8000/api/v1/analysis -H 'Content-Type: application/json' -d '{"mode":"historical","description":"demo run","sensor":"sentinel-1"}'
```

## Running tests

```bash
cd AquaVision
python -m pytest -q
```

## Limitations

- Live provider execution requires configured credentials.
- OpenDrift/OpenOil requires the appropriate runtime dependencies and may not be installed in all dev environments.
- The demo analysis is intentionally deterministic and clearly marked as DEMO data.
- This is a prototype and requires domain validation before real operational use.
