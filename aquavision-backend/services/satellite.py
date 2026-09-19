from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from providers.cdse import CDSEProvider


class SatelliteService:
    def __init__(self, provider: Optional[CDSEProvider] = None) -> None:
        self.provider = provider or CDSEProvider()

    def search_products(
        self,
        aoi: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        product_type: str = 'GRD',
        sensor: str = 'Sentinel-1',
        polarization: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        products = self.provider.search_products(aoi, start_time, end_time, product_type, sensor, polarization)
        return [
            {
                'id': product.id,
                'title': product.title,
                'platform': product.platform,
                'sensor': product.sensor,
                'acquisition_time': product.acquisition_time.isoformat() if product.acquisition_time else None,
                'polarization': product.polarization,
                'orbit': product.orbit,
                'metadata': product.metadata,
                'download_url': product.download_url,
                'bbox': product.bbox,
            }
            for product in products
        ]

    def get_product_metadata(self, product_id: str) -> Dict[str, Any]:
        return self.provider.get_product_metadata(product_id)

    def find_product(
        self,
        product_id: str,
        aoi: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        for product in self.search_products(aoi, start_time, end_time, sensor='Sentinel-1', polarization='VV/VH'):
            if product['id'] == product_id:
                return product
        raise ValueError(f'Selected Sentinel-1 acquisition was not found in CDSE search: {product_id}')

    def download_product(self, product_id: str, destination: str) -> str:
        return self.provider.download_product(product_id, destination)

    def process_sentinel1_vv_vh(
        self,
        aoi: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        destination: str,
        width: int = 512,
        height: int = 512,
    ) -> Dict[str, Any]:
        return self.provider.process_sentinel1_vv_vh(
            aoi, start_time, end_time, destination, width, height
        )
