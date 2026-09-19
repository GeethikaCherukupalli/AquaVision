from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


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


class CDSEProvider:
    """Minimal CDSE adapter with clear env-based credentials and demo-safe behavior."""

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None) -> None:
        self.username = username or os.getenv('CDSE_USERNAME')
        self.password = password or os.getenv('CDSE_PASSWORD')

    def search_products(
        self,
        aoi: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        product_type: str = 'GRD',
        sensor: str = 'Sentinel-1',
        polarization: Optional[str] = None,
    ) -> List[CDSEProduct]:
        if not self.username or not self.password:
            raise ValueError('CDSE credentials are missing. Set CDSE_USERNAME and CDSE_PASSWORD in the environment.')
        return [
            CDSEProduct(
                id='demo-product',
                title='demo-product',
                platform='Sentinel-1',
                sensor='Sentinel-1',
                acquisition_time=start_time,
                polarization=polarization or 'VV/VH',
                orbit='unknown',
                metadata={'demo': True},
                download_url=None,
            )
        ]

    def get_product_metadata(self, product_id: str) -> Dict[str, Any]:
        if not self.username or not self.password:
            raise ValueError('CDSE credentials are missing. Set CDSE_USERNAME and CDSE_PASSWORD in the environment.')
        return {'product_id': product_id, 'provider': 'CDSE', 'status': 'demo'}

    def download_product(self, product_id: str, destination: str) -> str:
        if not self.username or not self.password:
            raise ValueError('CDSE credentials are missing. Set CDSE_USERNAME and CDSE_PASSWORD in the environment.')
        return destination
