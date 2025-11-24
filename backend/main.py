from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io

from backend.preprocessing import load_satellite_image, create_tiles, normalize_resolution
from backend.classifier import segment_image
from backend.reconstruction import reconstruct_segmentation_map, generate_statistics
from backend.config import DEFAULT_TILE_SIZE, MAX_IMAGE_SIZE

app = FastAPI(title="Satellite Image Segmentation API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    tile_size: int = DEFAULT_TILE_SIZE,
    resolution: str = "10m"
):
    """
    Main endpoint for satellite image classification
    
    Args:
        file: Uploaded satellite image
        tile_size: Size of tiles for classification
        resolution: Image resolution (for context)
    
    Returns:
        JSON with segmentation map and statistics
    """
    try:
        # Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Preprocess
        if max(image_array.shape[:2]) > MAX_IMAGE_SIZE:
            # Downsample very large images for prototype
            image_array = normalize_resolution(image_array, target_size=MAX_IMAGE_SIZE)
        
        print(f"Processing image of size: {image_array.shape}")
        
        # Segment image
        mask = segment_image(image_array)
        
        # Reconstruct segmentation map (colorize)
        seg_map = reconstruct_segmentation_map(mask)
        
        # Generate statistics
        # We need to update generate_statistics to work with the mask
        # For now, let's create a simple stats generator here or update the function
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        # Map indices to names (matching DeepGlobe)
        idx_to_name = {
            0: 'urban',
            1: 'agriculture',
            2: 'rangeland',
            3: 'forest',
            4: 'water',
            5: 'barren',
            6: 'unknown'
        }
        
        stats = {
            'total_pixels': int(total_pixels),
            'class_distribution': {}
        }
        
        for idx, count in zip(unique, counts):
            name = idx_to_name.get(idx, f'class_{idx}')
            stats['class_distribution'][name] = {
                'count': int(count),
                'percentage': round(float(count) / total_pixels * 100, 2)
            }
        
        # Convert to base64 for response
        # We need to import encode_image again or move it to utils
        # It was in classifier.py, but we removed it.
        # Let's add it here or in a util file.
        # For quick fix, I'll add the helper here.
        from io import BytesIO
        import base64
        
        def encode_img(img_arr):
            img_pil = Image.fromarray(img_arr)
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        seg_map_b64 = encode_img(seg_map)
        
        return JSONResponse({
            "status": "success",
            "segmentation_map": seg_map_b64,
            "statistics": stats,
            "tile_results": [] # No longer used
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
