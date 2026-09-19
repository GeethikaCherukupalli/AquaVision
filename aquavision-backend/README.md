# AquaVision backend

This directory is the backend project boundary for the AquaVision frontend. It owns the FastAPI application, ML pipeline, service integrations, checkpoints, runtime data, and backend tests.

Run from the repository root:

```powershell
uvicorn aquavision.app:app --reload --port 8000
```

The frontend uses these routes:

- `GET /api/v1/readiness`
- `POST /api/v1/analysis`
- `GET /api/v1/analysis/{job_id}`
- `POST /api/v1/satellites/search`
- `POST /api/v1/simulations`
- `POST /api/v1/ais/candidates`

The frontend labels the current response as simulation data whenever production providers or artifacts are unavailable. Scientific processing remains owned by FastAPI and its provider/service modules; React only requests and visualizes the structured response.