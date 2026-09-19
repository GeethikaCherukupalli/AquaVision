# AquaVision API

The authoritative FastAPI application is `aquavision-backend/app.py`.

## Health and readiness

```text
GET /api/v1/health
GET /api/v1/readiness
```

## Sentinel-1 catalogue search

```text
POST /api/v1/satellites/search
```

Request fields:

```json
{
  "region": {"west": 68.0, "south": 8.0, "east": 78.0, "north": 23.0},
  "start_date": "2026-09-01",
  "end_date": "2026-09-08"
}
```

The endpoint uses CDSE credentials from environment variables and returns only catalogue products returned by CDSE. Missing credentials or provider failures are errors; no products are fabricated.

## Stage-1 analysis

```text
POST /api/v1/analysis
GET  /api/v1/analysis/{job_id}
```

The request must include the selected AOI, date range, CDSE acquisition identifier, and the acquisition timestamp returned by catalogue search:

```json
{
  "mode": "historical",
  "region": {"west": 68.0, "south": 8.0, "east": 78.0, "north": 23.0},
  "start_date": "2026-09-01",
  "end_date": "2026-09-08",
  "acquisition_id": "CDSE_PRODUCT_ID",
  "acquisition_time": "2026-09-04T04:12:00Z",
  "acquisition_time": "2026-09-04T04:12:00Z"
}
```

The response contains the computed ResNet18 v1 probability, threshold, candidate boolean, model metadata, selected acquisition timestamp, and AOI. Errors are never converted to `candidate: false`.
