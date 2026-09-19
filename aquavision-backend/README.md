# AquaVision backend

This is the authoritative FastAPI application for AquaVision.

This is a Python/FastAPI service. Do not run `npm run dev` from this directory;
there is intentionally no `package.json` here.

Run from this directory. The root path is added so the shared `ml/` package and model artifacts remain available:

```powershell
$env:PYTHONPATH=".."
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Or, from this directory, run:

```powershell
.\run.ps1
```

Run the React frontend separately from `aquavision-frontend`:

```powershell
cd ..\aquavision-frontend
npm install
npm run dev
```

The frontend uses these routes:

- `GET /api/v1/readiness`
- `POST /api/v1/analysis`
- `GET /api/v1/analysis/{job_id}`
- `POST /api/v1/satellites/search`
- `GET /api/v1/health`

Processing requires `CDSE_BACKSCATTER_COEFFICIENT` to be explicitly set after training provenance confirms the coefficient. The service will not choose one automatically.
