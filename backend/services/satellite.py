from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.providers.cdse import CDSEProvider


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
            }
            for product in products
        ]

    def get_product_metadata(self, product_id: str) -> Dict[str, Any]:
        return self.provider.get_product_metadata(product_id)

    def download_product(self, product_id: str, destination: str) -> str:
        return self.provider.download_product(product_id, destination)
