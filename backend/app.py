from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.providers.ais import AISPosition, AISTrajectory, MockAISProvider
from backend.services.ais import AISService
from backend.services.drift import DriftService
from backend.services.scoring import AISScoringService


class AnalysisRequest(BaseModel):
    mode: str = Field(default='historical', description='historical or monitoring')
    description: Optional[str] = None
    region: Dict[str, float] = Field(default_factory=lambda: {"west": -10.0, "south": 35.0, "east": 10.0, "north": 60.0})
    sensor: str = 'sentinel-1'
    acquisition_time: Optional[str] = None


class AnalysisJobResult(BaseModel):
    job_id: str
    status: str
    mode: str
    description: Optional[str]
    created_at: str
    updated_at: str
    result: Dict[str, Any]


class DemoAnalysisEngine:
    def __init__(self) -> None:
        self.counter = 0
        self.ais_provider = MockAISProvider(self._demo_trajectories())
        self.ais_service = AISService(provider=self.ais_provider)
        self.scorer = AISScoringService()
        self.drift = DriftService()

    def _demo_trajectories(self) -> List[AISTrajectory]:
        base = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
        return [
            AISTrajectory(
                vessel_id='VESSEL-2049',
                positions=[
                    AISPosition('VESSEL-2049', base - timedelta(hours=4), 44.810, -8.970, 14.0, 120.0, None, 'cargo'),
                    AISPosition('VESSEL-2049', base - timedelta(hours=2), 44.830, -8.930, 13.6, 122.0, None, 'cargo'),
                    AISPosition('VESSEL-2049', base, 44.860, -8.900, 13.2, 119.0, None, 'cargo'),
                ],
                metadata={'demo': True},
            ),
            AISTrajectory(
                vessel_id='VESSEL-7731',
                positions=[
                    AISPosition('VESSEL-7731', base - timedelta(hours=5), 44.790, -8.860, 10.5, 130.0, None, 'tanker'),
                    AISPosition('VESSEL-7731', base - timedelta(hours=3), 44.820, -8.830, 10.9, 125.0, None, 'tanker'),
                    AISPosition('VESSEL-7731', base, 44.845, -8.810, 11.1, 124.0, None, 'tanker'),
                ],
                metadata={'demo': True},
            ),
        ]

    def run_demo_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        self.counter += 1
        job_id = f"demo-{self.counter:04d}"
        now = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
        origin = {
            'type': 'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [[[ -8.92, 44.83], [-8.87, 44.83], [-8.87, 44.87], [-8.92, 44.87], [-8.92, 44.83]]]},
                'properties': {'probability': 0.82, 'source': 'DEMO'},
            }],
            'bbox': {"west": -8.95, "south": 44.80, "east": -8.82, "north": 44.90},
        }
        candidate_tracks = self.ais_service.get_candidates(origin, {"start": now - timedelta(hours=6), "end": now})
        scored = self.scorer.score_candidates(candidate_tracks, origin, {"start": now - timedelta(hours=6), "end": now})
        result = {
            'analysis_id': job_id,
            'mode': request.mode,
            'status': 'COMPLETED',
            'data_source': 'DEMO',
            'sensor': request.sensor,
            'region': request.region,
            'satellite': {
                'sensor': request.sensor,
                'product': 'S1A_IW_GRDH_1SDV_DEMO',
                'acquisition_time': request.acquisition_time or now.isoformat(),
                'polarization': 'VV/VH',
                'provider': 'DEMO',
            },
            'stage_1': {
                'classifier': {
                    'model': 'ResNet18',
                    'threshold': 0.12,
                    'candidate_patch_count': 3,
                    'probabilities': [0.91, 0.83, 0.74],
                },
                'segmentation': {
                    'mask_path': 'artifacts/demo/spill_mask.tif',
                    'oil_pixel_count': 15420,
                    'oil_fraction': 0.034,
                    'connected_components': 1,
                    'largest_component': 1,
                },
                'geometry': {
                    'area_km2': 1.2,
                    'perimeter_km': 4.7,
                    'centroid': {'lat': 44.845, 'lon': -8.89},
                    'bbox': {'min_lat': 44.82, 'max_lat': 44.88, 'min_lon': -8.96, 'max_lon': -8.82},
                    'units': 'km',
                    'crs': 'EPSG:4326',
                },
                'quality': {'georeferencing_available': True, 'warnings': ['DEMO mode only'], 'confidence': 0.82},
            },
            'stage_2': {
                'origin_region': origin,
                'probable_origin_time_window': {'start': (now - timedelta(hours=12)).isoformat(), 'end': (now - timedelta(hours=2)).isoformat()},
                'uncertainty': {'radius_km': 18.0, 'confidence': 0.79},
                'forecast': {
                    'horizons': ['+6h', '+12h', '+24h', '+48h', '+72h'],
                    'points': [
                        {'timestamp': (now + timedelta(hours=6)).isoformat(), 'lat': 44.83, 'lon': -8.91},
                        {'timestamp': (now + timedelta(hours=12)).isoformat(), 'lat': 44.81, 'lon': -8.94},
                        {'timestamp': (now + timedelta(hours=24)).isoformat(), 'lat': 44.74, 'lon': -9.02},
                        {'timestamp': (now + timedelta(hours=48)).isoformat(), 'lat': 44.68, 'lon': -9.11},
                        {'timestamp': (now + timedelta(hours=72)).isoformat(), 'lat': 44.61, 'lon': -9.21},
                    ],
                },
            },
            'stage_3': {
                'vessel_candidates': [
                    {
                        'vessel_id': item.vessel_id,
                        'score': item.score,
                        'classification': item.classification,
                        'evidence': item.evidence,
                        'positions': [
                            {
                                'lat': position.latitude,
                                'lon': position.longitude,
                                'timestamp': position.timestamp.isoformat(),
                                'speed': position.speed,
                                'course': position.course,
                            }
                            for position in item.positions
                        ],
                        'minimum_distance_km': item.proximity,
                        'time_difference_minutes': item.temporal_correlation,
                        'trajectory_consistency': item.trajectory_consistency,
                        'behavioral_anomaly': item.behavioral_anomaly,
                    }
                    for item in scored
                ],
                'correlation_summary': 'High spatio-temporal correlation between candidate vessel tracks and probable origin region.'
            },
            'warnings': ['DEMO data only. Not production evidence.'],
            'demo_mode': True,
        }
        return {
            'job_id': job_id,
            'status': 'COMPLETED',
            'mode': request.mode,
            'description': request.description,
            'created_at': now.isoformat(),
            'updated_at': now.isoformat(),
            'result': result,
        }


app = FastAPI(title='AquaVision API', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

engine = DemoAnalysisEngine()
analysis_jobs: Dict[str, AnalysisJobResult] = {}
simulation_jobs: Dict[str, Dict[str, Any]] = {}


@app.get('/api/v1/health')
def health() -> Dict[str, Any]:
    return {'status': 'ok', 'service': 'AquaVision API', 'timestamp': datetime.utcnow().isoformat()}


@app.get('/api/v1/readiness')
def readiness() -> Dict[str, Any]:
    """Report whether real Stage 1 and provider integrations are configured."""
    data_root = os.getenv('AQUAVISION_DATA_ROOT', '')
    artifact_root = Path(os.getenv('AQUAVISION_ARTIFACT_ROOT', 'artifacts'))
    required_artifacts = {
        'resnet18': artifact_root / 'resnet18' / 'best.pth',
        'unet': artifact_root / 'unet' / 'best.pth',
    }
    providers = {
        'cdse': bool(os.getenv('CDSE_USERNAME') and os.getenv('CDSE_PASSWORD')),
        'copernicus_marine': bool(
            os.getenv('COPERNICUS_MARINE_USERNAME')
            and os.getenv('COPERNICUS_MARINE_PASSWORD')
        ),
        'openmeteo': True,
        'ais': bool(os.getenv('AIS_USERNAME') and os.getenv('AIS_PASSWORD')),
    }
    production_analysis_available = False
    missing = []
    if not data_root or not Path(data_root).exists():
        missing.append('AQUAVISION_DATA_ROOT')
    missing.extend(name for name, path in required_artifacts.items() if not path.exists())
    missing.extend(name for name, configured in providers.items() if not configured and name != 'openmeteo')
    if not production_analysis_available:
        missing.append('production_analysis_pipeline')
    return {
        'status': 'ready' if not missing else 'not_ready',
        'production_analysis_available': production_analysis_available,
        'stage_1': {
            'dataset_root': data_root or None,
            'artifacts': {name: {'path': str(path), 'available': path.exists()} for name, path in required_artifacts.items()},
        },
        'providers': providers,
        'missing_configuration': missing,
        'demo_mode_available': True,
    }


@app.post('/api/v1/analysis', response_model=AnalysisJobResult)
def create_analysis(payload: AnalysisRequest) -> AnalysisJobResult:
    output = engine.run_demo_analysis(payload)
    job = AnalysisJobResult(**output)
    analysis_jobs[job.job_id] = job
    return job


@app.get('/api/v1/analysis/{job_id}')
def get_analysis(job_id: str) -> Dict[str, Any]:
    job = analysis_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'Analysis job not found: {job_id}')
    return job.model_dump()


@app.post('/api/v1/satellites/search')
def search_satellites(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'provider': 'CDSE',
        'status': 'demo-only',
        'results': [{
            'product_id': 'S1A_IW_GRDH_1SDV_DEMO',
            'sensor': 'Sentinel-1',
            'acquisition_time': datetime.utcnow().isoformat(),
            'polarization': 'VV/VH',
            'aoi': payload.get('region', {}),
            'metadata': {'demo': True},
        }],
    }


@app.post('/api/v1/simulations')
def create_simulation(payload: Dict[str, Any]) -> Dict[str, Any]:
    drift = DriftService()
    result = drift.run_backward_hindcast(payload.get('spill_region', {'lat': 44.85, 'lon': -8.90}), {'start': datetime.utcnow() - timedelta(hours=12), 'end': datetime.utcnow()})
    response = {
        'job_id': 'sim-demo',
        'status': 'COMPLETED',
        'origin': result.origin_probability_region,
        'confidence': result.confidence,
        'uncertainty_km': result.uncertainty,
        'provider': 'MOCK',
    }
    simulation_jobs[response['job_id']] = response
    return response


@app.get('/api/v1/simulations/{job_id}')
def get_simulation(job_id: str) -> Dict[str, Any]:
    simulation = simulation_jobs.get(job_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail=f'Simulation job not found: {job_id}')
    return simulation


@app.post('/api/v1/ais/candidates')
def get_ais_candidates(payload: Dict[str, Any]) -> Dict[str, Any]:
    region = payload.get('region', {'west': -10.0, 'south': 35.0, 'east': 10.0, 'north': 60.0})
    time_window = payload.get('time_window', {'start': datetime.utcnow() - timedelta(hours=12), 'end': datetime.utcnow()})
    candidates = engine.ais_service.get_candidates({'bbox': region}, {'start': datetime.fromisoformat(time_window['start']), 'end': datetime.fromisoformat(time_window['end'])})
    return {'candidates': [
        {'vessel_id': candidate.vessel_id, 'positions': [{'lat': pos.latitude, 'lon': pos.longitude, 'timestamp': pos.timestamp.isoformat()} for pos in candidate.positions]}
        for candidate in candidates
    ]}
