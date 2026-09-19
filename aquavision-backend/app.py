from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Literal
import tempfile
import math

import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from services.satellite import SatelliteService


class AnalysisRequest(BaseModel):
    mode: Literal['historical', 'monitoring']
    region: Dict[str, float]
    sensor: Literal['sentinel-1'] = 'sentinel-1'
    start_date: str
    end_date: str
    acquisition_id: str
    acquisition_time: Optional[str] = None
    execution_mode: Literal['production'] = 'production'

    @model_validator(mode='after')
    def validate_request(self):
        required = ('west', 'south', 'east', 'north')
        if any(key not in self.region for key in required):
            raise ValueError('region must contain west, south, east, and north')
        if self.region['west'] >= self.region['east'] or self.region['south'] >= self.region['north']:
            raise ValueError('region bounds are invalid')
        start = datetime.fromisoformat(self.start_date)
        end = datetime.fromisoformat(self.end_date)
        if start > end:
            raise ValueError('start_date must not be after end_date')
        return self


class AnalysisJobResult(BaseModel):
    job_id: str
    status: str
    mode: str
    created_at: str
    updated_at: str
    result: Dict[str, Any]


app = FastAPI(title='AquaVision API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
satellite_service = SatelliteService()
analysis_jobs: Dict[str, AnalysisJobResult] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _intersection_bbox(region: Dict[str, float], product_bbox: Optional[List[float]]) -> Dict[str, float]:
    if not product_bbox or len(product_bbox) < 4:
        return region
    west = max(region['west'], product_bbox[0])
    south = max(region['south'], product_bbox[1])
    east = min(region['east'], product_bbox[2])
    north = min(region['north'], product_bbox[3])
    if west >= east or south >= north:
        raise ValueError('Selected AOI does not intersect the selected Sentinel-1 acquisition footprint')
    return {'west': west, 'south': south, 'east': east, 'north': north}


def _processing_tiles(bbox: Dict[str, float], max_tile_degrees: float = 0.45) -> List[Dict[str, float]]:
    """Split a catalog-filtered bbox into physical-scale 512px windows."""
    tiles = []
    latitude = bbox['south']
    while latitude < bbox['north']:
        tile_north = min(latitude + max_tile_degrees, bbox['north'])
        longitude = bbox['west']
        while longitude < bbox['east']:
            lon_step = max_tile_degrees / max(math.cos(math.radians((latitude + tile_north) / 2)), 0.25)
            tile_east = min(longitude + lon_step, bbox['east'])
            tiles.append({'west': longitude, 'south': latitude, 'east': tile_east, 'north': tile_north})
            longitude = tile_east
        latitude = tile_north
    return tiles


@app.get('/api/v1/health')
def health() -> Dict[str, Any]:
    return {'status': 'ok', 'service': 'AquaVision API', 'timestamp': datetime.now(timezone.utc).isoformat()}


@app.get('/api/v1/readiness')
def readiness() -> Dict[str, Any]:
    artifact_root = Path(os.getenv('AQUAVISION_ARTIFACT_ROOT', str(PROJECT_ROOT / 'ml' / 'artifacts')))
    checkpoint = artifact_root / 'resnet18' / 'v1' / 'resnet18_best_epoch06.pth'
    metadata = artifact_root / 'resnet18' / 'v1' / 'resnet18_metadata.json'
    normalization = artifact_root / 'resnet18' / 'v1' / 'sar_normalization.json'
    cdse_configured = bool(os.getenv('CDSE_CLIENT_ID') and os.getenv('CDSE_CLIENT_SECRET'))
    coefficient_configured = bool(os.getenv('CDSE_BACKSCATTER_COEFFICIENT'))
    missing = [name for name, path in {'resnet_checkpoint': checkpoint, 'resnet_metadata': metadata, 'sar_normalization': normalization}.items() if not path.exists()]
    if not cdse_configured:
        missing.append('cdse_credentials')
    if not coefficient_configured:
        missing.append('cdse_backscatter_coefficient')
    return {
        'status': 'ready' if not missing else 'not_ready',
        'production_analysis_available': not missing,
        'stage_1': {'artifacts': {'resnet_checkpoint': {'path': str(checkpoint), 'available': checkpoint.exists()}, 'metadata': {'path': str(metadata), 'available': metadata.exists()}, 'normalization': {'path': str(normalization), 'available': normalization.exists()}}},
        'providers': {'cdse': cdse_configured, 'cdse_processing': cdse_configured and coefficient_configured},
        'missing_configuration': missing,
    }


@app.post('/api/v1/satellites/search')
def search_satellites(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        region = payload['region']
        start = datetime.fromisoformat(payload['start_date'])
        end = datetime.fromisoformat(payload['end_date'])
        products = satellite_service.search_products(region, start, end, sensor='Sentinel-1', polarization='VV/VH')
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f'Invalid AOI/date search: {exc}') from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'CDSE authentication or catalogue search failed: {exc}') from exc
    return {'provider': 'CDSE', 'status': 'COMPLETED', 'results': products}


@app.post('/api/v1/analysis', response_model=AnalysisJobResult)
def create_analysis(payload: AnalysisRequest) -> AnalysisJobResult:
    if not payload.acquisition_time:
        raise HTTPException(status_code=422, detail='acquisition_time from the selected CDSE product is required for Processing API analysis.')
    try:
        from services.production_analysis import ProductionAnalysisEngine
        requested_start = datetime.fromisoformat(payload.start_date)
        requested_end = datetime.fromisoformat(payload.end_date)
        selected_product = satellite_service.find_product(
            payload.acquisition_id, payload.region, requested_start, requested_end
        )
        acquisition_value = selected_product.get('acquisition_time') or payload.acquisition_time
        acquisition_time = datetime.fromisoformat(acquisition_value.replace('Z', '+00:00'))
        processing_region = _intersection_bbox(payload.region, selected_product.get('bbox'))
        tiles = _processing_tiles(processing_region)
        process_start = acquisition_time - timedelta(minutes=1)
        process_end = acquisition_time + timedelta(minutes=1)
        with tempfile.TemporaryDirectory(prefix='aquavision-cdse-') as temporary_dir:
            tile_results = []
            for tile_index, tile in enumerate(tiles):
                scene_path = Path(temporary_dir) / f'sentinel1_vv_vh_{tile_index:04d}.tif'
                processing = satellite_service.process_sentinel1_vv_vh(
                    tile,
                    process_start,
                    process_end,
                    str(scene_path),
                    width=512,
                    height=512,
                )
                tile_result = ProductionAnalysisEngine().run(scene_path, units=processing['units'])
                tile_result['scene']['path'] = None
                tile_result['processing'] = processing
                tile_result['processing']['bbox'] = tile
                tile_results.append(tile_result)
            if not tile_results:
                raise ValueError('No valid processing tiles were generated for the selected AOI')
            result = tile_results[0]
            probabilities = [tile_result['classification_probability'] for tile_result in tile_results]
            result['candidate'] = any(tile_result['candidate'] for tile_result in tile_results)
            result['classification_probability'] = max(probabilities)
            result['patch_count'] = sum(tile_result['patch_count'] for tile_result in tile_results)
            result['tile_count'] = len(tile_results)
            result['tile_probabilities'] = probabilities
            result['processing_region'] = processing_region
            result['processing_tiles'] = [tile_result['processing']['bbox'] for tile_result in tile_results]
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f'CDSE processing or analysis failed: {exc}') from exc
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f'CDSE processing or analysis failed: {exc}') from exc
    result.update({'source': 'Sentinel-1', 'acquisition_id': selected_product['id'], 'acquisition_time': selected_product.get('acquisition_time') or payload.acquisition_time, 'selected_product': selected_product, 'aoi': payload.region, 'requested_start_date': payload.start_date, 'requested_end_date': payload.end_date})
    now = datetime.now(timezone.utc).isoformat()
    job = AnalysisJobResult(job_id=result['analysis_id'], status='COMPLETED', mode=payload.mode, created_at=now, updated_at=now, result=result)
    analysis_jobs[job.job_id] = job
    return job


@app.get('/api/v1/analysis/{job_id}')
def get_analysis(job_id: str) -> Dict[str, Any]:
    job = analysis_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Analysis job not found: {job_id}')
    return job.model_dump()
