import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image

class SegmentationModel:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        model_id = "florian-morel22/segformer-b0-deepglobe-land-cover"
        print(f"Loading model: {model_id}")
        
        # Use generic processor as the fine-tuned model might miss preprocessor_config.json
        # The backbone is mit-b0, so this processor is compatible
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_array):
        """
        Predict segmentation mask for an image array (H, W, C)
        Uses sliding window inference for large images to preserve detail.
        Returns: (H, W) mask with class indices
        """
        # Ensure RGB
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
            
        h, w = image_array.shape[:2]
        
        # Sliding window parameters
        window_size = 1024  # Size of the window to feed to the model
        stride = 768       # Stride (overlap = window_size - stride = 256)
        
        # If image is small enough, just predict directly (but resize to window_size if needed)
        if h <= window_size and w <= window_size:
            return self._predict_single_window(image_array)
            
        # Initialize full mask and count map for averaging (though we use argmax, so voting is tricky. 
        # For simplicity in this prototype, we'll just overwrite or use a center-crop strategy.
        # A better approach for segmentation is to accumulate logits. Let's do that.)
        
        num_classes = 7 # DeepGlobe classes
        full_logits = np.zeros((num_classes, h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Iterate over windows
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + window_size, h)
                x_end = min(x + window_size, w)
                
                # Adjust start if we hit the boundary
                y_start = y
                x_start = x
                if y_end == h:
                    y_start = max(0, h - window_size)
                if x_end == w:
                    x_start = max(0, w - window_size)
                    
                # Extract crop
                crop = image_array[y_start:y_end, x_start:x_end]
                
                # Predict crop
                # We need logits here, so we'll make a helper that returns logits
                logits = self._predict_logits(crop) # Shape: (classes, crop_h, crop_w)
                
                # Add to full map
                full_logits[:, y_start:y_end, x_start:x_end] += logits
                count_map[y_start:y_end, x_start:x_end] += 1.0
                
        # Average logits
        # Avoid division by zero
        count_map = np.maximum(count_map, 1.0)
        full_logits /= count_map
        
        # Argmax to get final mask
        mask = np.argmax(full_logits, axis=0)
        return mask

    def _predict_single_window(self, image_array):
        """Helper for simple prediction of small images"""
        logits = self._predict_logits(image_array)
        return np.argmax(logits, axis=0)

    def _predict_logits(self, image_array):
        """
        Predict logits for a single image crop.
        Returns: numpy array (classes, H, W)
        """
        # Convert to PIL
        image = Image.fromarray(image_array)
        
        # Prepare input
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Upsample logits to original crop size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1], # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            
            return upsampled_logits.squeeze().cpu().numpy()
