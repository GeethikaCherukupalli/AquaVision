from fastapi.testclient import TestClient

from aquavision.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_analysis_creation_demo_mode():
    payload = {
        'mode': 'historical',
        'description': 'demo historical run',
        'region': {'west': -9.0, 'south': 42.0, 'east': -7.0, 'north': 43.5},
        'sensor': 'sentinel-1',
    }
    response = client.post('/api/v1/analysis', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['status'] in {'QUEUED', 'RUNNING', 'COMPLETED'}
    assert 'job_id' in data
    assert 'result' in data

    lookup = client.get(f"/api/v1/analysis/{data['job_id']}")
    assert lookup.status_code == 200
    assert lookup.json()['job_id'] == data['job_id']

    missing = client.get('/api/v1/analysis/unknown-job')
    assert missing.status_code == 404
