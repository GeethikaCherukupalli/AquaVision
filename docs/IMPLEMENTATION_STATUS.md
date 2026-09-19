# Implementation status

Updated: 2026-09-19

| Requirement | Status | Notes |
|---|---|---|
| Authoritative app roots | YES | `aquavision-backend/` and `aquavision-frontend/` are the only application roots. |
| AOI selection | YES | Leaflet map supports navigation and drag-to-draw geographic bbox selection. |
| Date selection | YES | Historical and monitoring workflows expose start/end date inputs. |
| CDSE catalogue search | IMPLEMENTED | Search uses configured OAuth client credentials and returns actual products only. |
| CDSE Processing API path | IMPLEMENTED | Selected acquisition window requests numerical VV/VH FLOAT32 GeoTIFF data. |
| ResNet18 v1 API path | IMPLEMENTED | CDSE-processed VV/VH data is validated and passed to the committed checkpoint. |
| Production frontend path | IMPLEMENTED | Demo execution was removed; UI requires AOI, acquisition, and real analysis inputs. |
| `npm run build` | PASS | Active Vite frontend builds successfully. |

## Remaining blockers

- The training backscatter coefficient is not recorded; `CDSE_BACKSCATTER_COEFFICIENT` must be confirmed before a real production request is allowed.
- No authenticated CDSE smoke test was run because credentials were unavailable.
- The active environment has no `pytest`, `torch`, or `torchvision`, so model execution tests were not run.
- U-Net, OpenDrift, AIS, and orbital satellite-position processing are not active production stages.
