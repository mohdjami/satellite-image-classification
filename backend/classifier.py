import numpy as np
from backend.model import SegmentationModel

# Initialize model once
# We can use a global instance or a singleton pattern
# For this prototype, a global instance is fine
print("Loading DeepLabV3+ model...")
model = SegmentationModel()
print("Model loaded.")

def segment_image(image_array):
    """
    Segment the entire image using DeepLabV3+
    
    Args:
        image_array: numpy array (H, W, C)
    
    Returns:
        segmentation_mask: numpy array (H, W) with class indices
    """
    return model.predict(image_array)

# Keep these for compatibility if needed, or remove them
# For now, we'll remove the old tile-based logic as we are moving to full image segmentation
# If the image is too large, we might need sliding window, but let's start simple.
