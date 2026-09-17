# AquaVision backend

This is the backend project boundary for the AquaVision frontend. The currently tested FastAPI implementation remains in the importable `backend/` package so existing Python tests and provider modules keep their stable imports.

Run from the repository root:

```powershell
uvicorn backend.app:app --reload --port 8000
```

The frontend uses these routes:

- `GET /api/v1/readiness`
- `POST /api/v1/analysis`
- `GET /api/v1/analysis/{job_id}`
- `POST /api/v1/satellites/search`
- `POST /api/v1/simulations`
- `POST /api/v1/ais/candidates`

The frontend labels the current response as simulation data whenever production providers or artifacts are unavailable. Scientific processing remains owned by FastAPI and its provider/service modules; React only requests and visualizes the structured response.