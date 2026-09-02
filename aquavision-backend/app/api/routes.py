from fastapi import APIRouter, HTTPException, status, Query
from app.ml.sar.sentinel_service import fetch_sentinel_image
from app.ml.sar.processor import process_gee_image
# We will uncomment these as we integrate the real NetCDF and AIS
# from app.ml.metocean.hindcast_engine import run_lagrangian_simulation
# from app.ml.attribution.scorer import compute_vessel_attributions

router = APIRouter()

@router.get("/pipeline/full-investigation", status_code=status.HTTP_200_OK)
async def execute_full_investigation(
    lat: float = Query(28.932, description="Target Latitude (e.g., Gulf of Mexico)"),
    lon: float = Query(-88.974, description="Target Longitude"),
    date: str = Query(None, description="Optional ISO date string")
):
    try:
        # 1. Fetch live radar data & timestamp from Google Earth Engine
        print(f"Initiating Earth Engine fetch for Coordinates: {lat}, {lon}")
        gee_metadata = fetch_sentinel_image(lat=lat, lon=lon, target_date=date)

        # 2. Run PyTorch CNN Inference & Geometry Extraction
        print("Running SAR Deep Learning Inference...")
        sar_payload = process_gee_image(gee_metadata)

        # 3. Metocean & AIS (Temporarily returning empty lists while we verify SAR)
        # drift_payload = run_lagrangian_simulation(sar_payload)
        # suspects = compute_vessel_attributions(drift_payload["origin_estimate"], get_ais_telemetry())

        return {
            "status": "success",
            "detection": sar_payload,
            "metocean_hindcast": None, # drift_payload
            "suspect_ranking": [] # suspects
        }
        
    except Exception as exc:
        print(f"Error in pipeline: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "PipelineExecutionError", "message": str(exc)}
        )