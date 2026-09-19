from __future__ import annotations

import numpy as np


def linear_power_to_db(values: np.ndarray) -> np.ndarray:
    """Convert SAR linear power to dB without scene-wise rescaling."""
    values = np.asarray(values, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 10.0 * np.log10(values)
    result[values <= 0] = np.nan
    return result.astype(np.float32)


def prepare_vv_vh(values: np.ndarray, units: str = 'dB') -> np.ndarray:
    """Return finite, ordered VV/VH dB channels for the trained model."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError(f'Expected VV/VH array shaped (2,H,W), got {values.shape}')
    if units.upper() in {'LINEAR', 'LINEAR_POWER'}:
        values = linear_power_to_db(values)
    elif units.upper() != 'DB':
        raise ValueError(f'Unsupported SAR units: {units}')
    if not np.isfinite(values).all():
        raise ValueError('VV/VH raster contains non-finite or non-positive SAR values')
    return values