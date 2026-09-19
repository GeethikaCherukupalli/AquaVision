import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'aquavision-backend'))
from providers.cdse import CDSEProvider
from ml.preprocessing.sar import linear_power_to_db, prepare_vv_vh


class FakeResponse:
    status_code = 200
    content = b'II*\x00' + b'fixture'
    headers = {'content-type': 'image/tiff'}

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.payload = None

    def post(self, url, **kwargs):
        if 'token' in url:
            return type('TokenResponse', (), {'raise_for_status': lambda self: None, 'json': lambda self: {'access_token': 'test-token'}})()
        self.payload = kwargs['json']
        return FakeResponse()


def test_processing_request_preserves_bbox_time_and_two_band_evalscript(monkeypatch, tmp_path):
    monkeypatch.setenv('CDSE_CLIENT_ID', 'id')
    monkeypatch.setenv('CDSE_CLIENT_SECRET', 'secret')
    monkeypatch.setenv('CDSE_BACKSCATTER_COEFFICIENT', 'SIGMA0_ELLIPSOID')
    session = FakeSession()
    provider = CDSEProvider(session=session)
    destination = tmp_path / 'scene.tif'
    metadata = provider.process_sentinel1_vv_vh(
        {'west': 68, 'south': 8, 'east': 78, 'north': 23},
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, 0, 1, tzinfo=timezone.utc),
        str(destination),
    )
    assert destination.exists()
    assert metadata['bands'] == ['VV', 'VH']
    assert session.payload['input']['bounds']['bbox'] == [68, 8, 78, 23]
    assert session.payload['input']['data'][0]['processing']['backCoeff'] == 'SIGMA0_ELLIPSOID'
    assert session.payload['input']['data'][0]['processing']['orthorectify'] is False
    assert 'LINEAR_POWER' in session.payload['evalscript']
    assert 'sample.VV' in session.payload['evalscript']
    assert 'sample.VH' in session.payload['evalscript']


def test_linear_power_to_db_and_fixed_channel_preparation():
    linear = np.array([[[1.0, 10.0]], [[0.1, 1.0]]], dtype=np.float32)
    prepared = prepare_vv_vh(linear, units='LINEAR_POWER')
    np.testing.assert_allclose(prepared, np.array([[[0.0, 10.0]], [[-10.0, 0.0]]], dtype=np.float32))
    np.testing.assert_allclose(linear_power_to_db(np.array([1.0, 10.0], dtype=np.float32)), [0.0, 10.0])


def test_missing_coefficient_fails_without_guessing(monkeypatch, tmp_path):
    monkeypatch.setenv('CDSE_CLIENT_ID', 'id')
    monkeypatch.setenv('CDSE_CLIENT_SECRET', 'secret')
    monkeypatch.delenv('CDSE_BACKSCATTER_COEFFICIENT', raising=False)
    provider = CDSEProvider(session=FakeSession())
    try:
        provider.process_sentinel1_vv_vh(
            {'west': 0, 'south': 0, 'east': 1, 'north': 1},
            datetime.now(timezone.utc), datetime.now(timezone.utc), str(tmp_path / 'scene.tif'),
        )
    except ValueError as exc:
        assert 'CDSE_BACKSCATTER_COEFFICIENT' in str(exc)
    else:
        raise AssertionError('Missing coefficient must not be guessed')
