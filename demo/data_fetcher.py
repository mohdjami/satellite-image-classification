"""
demo/data_fetcher.py
Downloads satellite tile imagery from ESRI World Imagery (public tile server).
No API key needed. Used for real demonstrations of Obj 2 and Obj 5.
"""

import math
import io
import time
import numpy as np
import requests
from PIL import Image

ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research prototype)"
}


def _lat_lon_to_tile(lat: float, lon: float, zoom: int):
    """Convert geographic coordinates to tile (x, y) at given zoom level."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile(x: int, y: int, z: int, retries: int = 3) -> Image.Image:
    """Fetch a single 256×256 satellite tile."""
    url = ESRI_URL.format(z=z, y=y, x=x)
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)


def fetch_region(lat: float, lon: float, zoom: int = 14, grid: int = 2) -> np.ndarray:
    """
    Fetch a satellite image centred at (lat, lon).
    Downloads a grid×grid mosaic of tiles, returns (H, W, 3) uint8 numpy array.
    
    At zoom=14: each tile ≈ 2.5 km², grid=2 → ~5×5 km scene.
    """
    cx, cy = _lat_lon_to_tile(lat, lon, zoom)
    # offset so the centre tile is in the middle of the grid
    x0 = cx - grid // 2
    y0 = cy - grid // 2

    rows = []
    for ty in range(y0, y0 + grid):
        row_tiles = []
        for tx in range(x0, x0 + grid):
            tile = fetch_tile(tx, ty, zoom)
            row_tiles.append(np.array(tile))
        rows.append(np.concatenate(row_tiles, axis=1))   # stack columns

    image = np.concatenate(rows, axis=0)                  # stack rows
    return image


def fetch_patches(lat: float, lon: float, zoom: int = 15,
                  grid: int = 6, patch_size: int = 256) -> list[np.ndarray]:
    """
    Fetch a grid×grid tile mosaic then slice into 256×256 patches.
    Returns list of (256, 256, 3) numpy arrays.
    """
    mosaic = fetch_region(lat, lon, zoom=zoom, grid=grid)
    h, w = mosaic.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patches.append(mosaic[y:y + patch_size, x:x + patch_size])
    return patches


# ─── Predefined geographic regions ───────────────────────────────────────────

TRANSFER_REGIONS = {
    "DeepGlobe / Source\n(Los Angeles, USA)":    (34.05,  -118.24, 14, 3),
    "Europe\n(Paris suburbs, France)":            (48.80,    2.35, 14, 3),
    "South Asia\n(Rural Gujarat, India)":         (22.30,   72.60, 14, 3),
    "West Africa\n(Sahel, Niger)":                (13.50,    2.10, 14, 3),
    "Latin America\n(Amazon, Brazil)":            (-3.47,  -62.22, 14, 3),
}

# Source-domain areas for coreset patch sampling
CORESET_SAMPLE_LOCATIONS = [
    (34.05,  -118.24),   # LA — urban
    (34.10,  -118.00),   # LA suburbs — mixed
    (36.17,  -115.14),   # Las Vegas — barren/urban
    (37.77,  -122.42),   # San Francisco — coastal/urban
    (38.58,  -121.49),   # Sacramento — agriculture
    (36.89,  -119.77),   # Fresno — farmland
]
