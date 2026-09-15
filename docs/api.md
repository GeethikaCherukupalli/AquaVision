# AquaVision API

## Health

GET /api/v1/health

## Analysis

POST /api/v1/analysis

Request body:
```json
{
  "mode": "historical",
  "description": "demo historical run",
  "region": {"west": -9.0, "south": 42.0, "east": -7.0, "north": 43.5},
  "sensor": "sentinel-1"
}
```

## Satellite search

POST /api/v1/satellites/search

## Simulations

POST /api/v1/simulations
GET /api/v1/simulations/{job_id}

## AIS candidates

POST /api/v1/ais/candidates
