import numpy as np
from PIL import Image
import rasterio

def load_satellite_image(image_path):
    """Load satellite image (supports GeoTIFF, PNG, JPG)"""
    if image_path.endswith('.tif') or image_path.endswith('.tiff'):
        with rasterio.open(image_path) as src:
            img = src.read()  # Read all bands
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            # Normalize to 0-255 if needed
            if img.max() > 255:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            return img
    else:
        return np.array(Image.open(image_path))

def create_tiles(image, tile_size=256, overlap=0):
    """
    Split large image into manageable tiles
    
    Args:
        image: numpy array (H, W, C)
        tile_size: size of each tile (default 256x256)
        overlap: overlap between tiles in pixels (default 0)
    
    Returns:
        List of (tile, position) tuples
    """
    tiles = []
    h, w = image.shape[:2]
    
    # specific case: if image is smaller than tile_size, pad it or just return it as one tile
    # For simplicity in this prototype, if smaller, we'll just return the image as is 
    # (assuming classifier can handle it or we pad)
    # But to be safe with the fixed size expectation, let's pad if needed.
    
    # However, a simpler fix for "0 tiles" is to ensure the range covers everything.
    # If h < tile_size, range(0, h - tile_size + 1) is empty.
    
    # Let's use a more robust tiling strategy that pads if necessary
    import math
    
    # Pad image to be multiples of tile_size (or at least tile_size)
    pad_h = max(0, tile_size - h) if h < tile_size else (tile_size - (h % tile_size)) % tile_size
    pad_w = max(0, tile_size - w) if w < tile_size else (tile_size - (w % tile_size)) % tile_size
    
    if pad_h > 0 or pad_w > 0:
        # Pad with zeros (black)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        h, w = image.shape[:2]
        
    stride = tile_size - overlap
    
    for i in range(0, h - tile_size + 1, stride):
        for j in range(0, w - tile_size + 1, stride):
            tile = image[i:i+tile_size, j:j+tile_size]
            position = (i, j, i+tile_size, j+tile_size)
            tiles.append((tile, position))
            
    return tiles

def normalize_resolution(image, target_size=512):
    """Resize image to standard size for consistent processing"""
    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img_resized)
