import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define color map for visualization
CLASS_COLORS = {
    'vegetation': [0, 255, 0],      # Green
    'water': [0, 0, 255],           # Blue
    'buildings': [128, 128, 128],   # Gray
    'roads': [0, 0, 0],             # Black
    'agriculture': [255, 255, 0],   # Yellow
    'barren': [165, 42, 42],        # Brown
    'unknown': [255, 255, 255]      # White
}

def reconstruct_segmentation_map(original_shape, tile_results, tile_size=256):
    """
    Reconstruct full segmentation map from tile classifications
    
    Args:
        original_shape: (H, W) of original image
        tile_results: List of classification results with positions
        tile_size: Size of each tile
    
    Returns:
        segmentation_map: RGB image (H, W, 3)
    """
    h, w = original_shape[:2]
    seg_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    for result in tile_results:
        class_name = result.get('class', 'unknown').lower()
        # Handle case sensitivity or slight variations if needed
        if class_name not in CLASS_COLORS:
            class_name = 'unknown'
            
        pos = result['position']
        i1, j1, i2, j2 = pos
        
        color = CLASS_COLORS.get(class_name, CLASS_COLORS['unknown'])
        
        # Fill the tile area with the color
        # Note: This simple version fills the whole tile with one color.
        # A more advanced version would be pixel-wise if the model supported it.
        # For now, we are doing tile-based classification.
        seg_map[i1:i2, j1:j2] = color
    
    return seg_map

def create_overlay(original_image, seg_map, alpha=0.5):
    """Create overlay of segmentation on original image"""
    # Ensure sizes match
    if original_image.shape != seg_map.shape:
        # Resize seg_map to match original if needed, or vice versa
        # For now assuming they match due to reconstruction logic
        pass
        
    overlay = (original_image * (1 - alpha) + seg_map * alpha).astype(np.uint8)
    return overlay

def generate_statistics(tile_results):
    """Generate classification statistics"""
    from collections import Counter
    
    classes = [r.get('class', 'unknown') for r in tile_results]
    class_counts = Counter(classes)
    
    total = len(tile_results)
    stats = {
        'total_tiles': total,
        'class_distribution': {
            cls: {
                'count': count,
                'percentage': round(count / total * 100, 2)
            }
            for cls, count in class_counts.items()
        }
    }
    
    return stats
