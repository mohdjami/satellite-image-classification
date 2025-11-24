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

def reconstruct_segmentation_map(mask):
    """
    Convert segmentation mask to RGB image
    
    Args:
        mask: numpy array (H, W) with class indices
    
    Returns:
        segmentation_map: RGB image (H, W, 3)
    """
    h, w = mask.shape
    seg_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Map indices to colors
    # We need to ensure the model's class indices match these keys
    # Assuming model classes are:
    # 0: vegetation
    # 1: water
    # 2: buildings
    # 3: roads
    # 4: agriculture
    # 5: barren
    # 6: unknown (or background)
    
    # Create a lookup table for faster coloring
    # This is a simple way; for very large images, vectorization is better
    
    # DeepGlobe Land Cover Classes:
    # 0: Urban
    # 1: Agriculture
    # 2: Rangeland
    # 3: Forest
    # 4: Water
    # 5: Barren
    # 6: Unknown
    
    idx_to_color = {
        0: [0, 255, 255],      # Urban (Cyan)
        1: [255, 255, 0],      # Agriculture (Yellow)
        2: [255, 0, 255],      # Rangeland (Magenta)
        3: [0, 255, 0],        # Forest (Green)
        4: [0, 0, 255],        # Water (Blue)
        5: [255, 255, 255],    # Barren (White)
        6: [0, 0, 0]           # Unknown (Black)
    }
    
    for idx, color in idx_to_color.items():
        seg_map[mask == idx] = color
    
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
