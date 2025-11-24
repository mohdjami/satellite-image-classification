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
        Returns: (H, W) mask with class indices
        """
        # Preprocess
        # Ensure RGB
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
            
        # Convert to PIL for processor
        image = Image.fromarray(image_array)
        
        # Prepare input
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Upsample logits to original image size
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1], # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            
            # Get prediction
            mask = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy()
            
        return mask
