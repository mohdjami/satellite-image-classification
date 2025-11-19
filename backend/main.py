from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io

from backend.preprocessing import load_satellite_image, create_tiles, normalize_resolution
from backend.classifier import classify_all_tiles, encode_image
from backend.reconstruction import reconstruct_segmentation_map, generate_statistics
from backend.prompts.classification_prompt import SYSTEM_PROMPT, USER_PROMPT
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
        
        # Create tiles
        tiles = create_tiles(image_array, tile_size=tile_size)
        
        print(f"Created {len(tiles)} tiles from image")
        
        # Classify tiles
        # Use a smaller min_tile_size for refinement (e.g., 1/4 of tile_size or fixed 64)
        min_size = max(64, tile_size // 4)
        results = await classify_all_tiles(tiles, SYSTEM_PROMPT, USER_PROMPT, min_tile_size=min_size)
        
        # Reconstruct segmentation map
        seg_map = reconstruct_segmentation_map(
            image_array.shape, 
            results, 
            tile_size=tile_size
        )
        
        # Generate statistics
        stats = generate_statistics(results)
        
        # Convert to base64 for response
        seg_map_b64 = encode_image(seg_map)
        
        return JSONResponse({
            "status": "success",
            "segmentation_map": seg_map_b64,
            "statistics": stats,
            "tile_results": results[:10]  # Return first 10 for preview
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
