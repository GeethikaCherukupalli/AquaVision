from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


@dataclass
class CDSEProduct:
    id: str
    title: str
    platform: str
    sensor: str
    acquisition_time: Optional[datetime] = None
    polarization: Optional[str] = None
    orbit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    download_url: Optional[str] = None
    bbox: Optional[List[float]] = None


class CDSEProvider:
    """Copernicus Data Space catalog and download adapter."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.client_id = os.getenv('CDSE_CLIENT_ID')
        self.client_secret = os.getenv('CDSE_CLIENT_SECRET')
        self.token_url = os.getenv(
            'CDSE_TOKEN_URL',
            'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        )
        self.catalog_url = os.getenv(
            'CDSE_CATALOG_URL',
            'https://sh.dataspace.copernicus.eu/catalog/v1/search',
        )
        self.process_url = os.getenv(
            'CDSE_PROCESS_URL',
            'https://sh.dataspace.copernicus.eu/process/v1',
        )
        self.collection = os.getenv('CDSE_COLLECTION', 'sentinel-1-grd')
        self.backscatter_coefficient = os.getenv('CDSE_BACKSCATTER_COEFFICIENT')
        self.session = session or requests.Session()

    def _token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise ValueError('CDSE credentials are missing. Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.')
        response = self.session.post(
            self.token_url,
            data={
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
            },
            timeout=30,
        )
        response.raise_for_status()
        token = response.json().get('access_token')
        if not token:
            raise RuntimeError('CDSE token response did not contain access_token')
        return token

    def search_products(
        self,
        aoi: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        product_type: str = 'GRD',
        sensor: str = 'Sentinel-1',
        polarization: Optional[str] = None,
    ) -> List[CDSEProduct]:
        west, south, east, north = (aoi[key] for key in ('west', 'south', 'east', 'north'))
        response = self.session.post(
            self.catalog_url,
            headers={'Authorization': f'Bearer {self._token()}'},
            json={
                'collections': [self.collection],
                'bbox': [west, south, east, north],
                'datetime': f'{start_time.isoformat()}/{end_time.isoformat()}',
                'limit': 100,
            },
            timeout=60,
        )
        response.raise_for_status()
        features = response.json().get('features', [])
        products = []
        for feature in features:
            properties = feature.get('properties', {})
            product_polarization = properties.get('polarisationChannels') or properties.get('polarization')
            if polarization and product_polarization and polarization not in str(product_polarization):
                continue
            acquisition = properties.get('datetime') or properties.get('start_datetime')
            products.append(CDSEProduct(
                id=feature.get('id', ''),
                title=properties.get('title', feature.get('id', '')),
                platform=properties.get('platform', 'Sentinel-1'),
                sensor=sensor,
                acquisition_time=datetime.fromisoformat(acquisition.replace('Z', '+00:00')) if acquisition else None,
                polarization=str(product_polarization) if product_polarization else polarization,
                orbit=properties.get('orbitDirection'),
                metadata={**properties, 'geometry': feature.get('geometry')},
                download_url=(feature.get('assets', {}).get('product', {}) or {}).get('href'),
                bbox=feature.get('bbox'),
            ))
        return products

    def get_product_metadata(self, product_id: str) -> Dict[str, Any]:
        response = self.session.get(
            f'{self.catalog_url}/{product_id}',
            headers={'Authorization': f'Bearer {self._token()}'},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def download_product(self, product_id: str, destination: str) -> str:
        metadata = self.get_product_metadata(product_id)
        href = (metadata.get('assets', {}).get('product', {}) or {}).get('href')
        if not href:
            raise RuntimeError(f'CDSE product has no downloadable product asset: {product_id}')
        response = self.session.get(
            href,
            headers={'Authorization': f'Bearer {self._token()}'},
            timeout=300,
            stream=True,
        )
        response.raise_for_status()
        with open(destination, 'wb') as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        return destination

    def process_sentinel1_vv_vh(
        self,
        aoi: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        destination: str,
        width: int = 512,
        height: int = 512,
    ) -> Dict[str, Any]:
        """Request numerical VV/VH dB GeoTIFF data from CDSE Processing API.

        CDSE returns linear Sigma0 power. The production preprocessing converts
        it to dB before applying the fixed training normalization.
        """
        if not self.backscatter_coefficient:
            raise ValueError(
                'CDSE_BACKSCATTER_COEFFICIENT is required; training provenance '
                'does not identify sigma0, beta0, gamma0, or terrain-corrected gamma0.'
            )
        if width % 512 or height % 512:
            raise ValueError('Processing output width and height must be multiples of 512')

        west, south, east, north = (aoi[key] for key in ('west', 'south', 'east', 'north'))
        evalscript = """//VERSION=3
function setup() {
    return {
        input: [{ bands: [\"VV\", \"VH\"], units: \"LINEAR_POWER\" }],
    output: { bands: 2, sampleType: \"FLOAT32\" }
  };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}"""
        payload = {
            'input': {
                'bounds': {
                    'bbox': [west, south, east, north],
                    'properties': {'crs': 'http://www.opengis.net/def/crs/EPSG/0/4326'},
                },
                'data': [{
                    'type': self.collection,
                    'dataFilter': {
                        'timeRange': {
                            'from': start_time.isoformat().replace('+00:00', 'Z'),
                            'to': end_time.isoformat().replace('+00:00', 'Z'),
                        },
                    },
                    'processing': {
                        'backCoeff': self.backscatter_coefficient,
                        'orthorectify': False,
                    },
                }],
            },
            'output': {
                'width': width,
                'height': height,
                'responses': [{'identifier': 'default', 'format': {'type': 'image/tiff'}}],
            },
            'evalscript': evalscript,
        }
        response = self.session.post(
            self.process_url,
            headers={
                'Authorization': f'Bearer {self._token()}',
                'Content-Type': 'application/json',
                'Accept': 'image/tiff',
            },
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'tif' not in content_type and not response.content.startswith((b'II*', b'MM\x00*')):
            raise RuntimeError(f'CDSE Processing API returned non-raster content: {content_type}')
        with open(destination, 'wb') as handle:
            handle.write(response.content)
        return {
            'process_url': self.process_url,
            'collection': self.collection,
            'backscatter_coefficient': self.backscatter_coefficient,
            'units': 'LINEAR_POWER',
            'bands': ['VV', 'VH'],
            'bbox': [west, south, east, north],
            'time_range': [start_time.isoformat(), end_time.isoformat()],
            'width': width,
            'height': height,
            'content_type': content_type,
        }
