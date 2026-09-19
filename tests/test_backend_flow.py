import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / 'aquavision-backend'))
from app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_analysis_requires_real_scene():
    payload = {
        'mode': 'historical',
        'region': {'west': -9.0, 'south': 42.0, 'east': -7.0, 'north': 43.5},
        'sensor': 'sentinel-1',
        'start_date': '2026-09-01',
        'end_date': '2026-09-02',
        'acquisition_id': 'sentinel-1-product',
        'acquisition_time': '2026-09-01T00:00:00Z',
    }
    response = client.post('/api/v1/analysis', json=payload)
    assert response.status_code == 422
    assert 'CDSE' in response.json()['detail']


def test_missing_analysis_returns_404():
    response = client.get('/api/v1/analysis/unknown-job')
    assert response.status_code == 404
