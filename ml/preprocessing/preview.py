from __future__ import annotations

import struct
import zlib

import numpy as np


def _png_gray(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError('Cannot preview a raster with no finite pixels')
    valid = values[finite]
    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        high = low + 1.0
    pixels = np.clip((values - low) / (high - low) * 255.0, 0, 255)
    pixels[~finite] = 0
    raw = b''.join(b'\x00' + pixels[row].astype(np.uint8).tobytes() for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)

    height, width = pixels.shape
    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)),
        chunk(b'IDAT', zlib.compress(raw, level=6)),
        chunk(b'IEND', b''),
    ])


def vv_vh_preview_data_urls(scene: np.ndarray) -> dict[str, str]:
    """Create display-only grayscale data URLs from actual VV/VH dB arrays."""
    import base64

    if scene.ndim != 3 or scene.shape[0] != 2:
        raise ValueError(f'Expected scene shape (2,H,W), got {scene.shape}')
    return {
        'vv': 'data:image/png;base64,' + base64.b64encode(_png_gray(scene[0])).decode('ascii'),
        'vh': 'data:image/png;base64,' + base64.b64encode(_png_gray(scene[1])).decode('ascii'),
    }
