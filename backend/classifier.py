import os
import base64
from io import BytesIO
from openai import OpenAI
import json
from PIL import Image
from backend.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_array):
    """Convert numpy array to base64 for OpenAI API"""
    img_pil = Image.fromarray(image_array)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def classify_tile(tile, position, system_prompt, user_prompt):
    """
    Classify a single tile using GPT-4 Vision
    
    Args:
        tile: numpy array of image tile
        position: (row_start, col_start, row_end, col_end)
        system_prompt: System instructions
        user_prompt: User query template
    
    Returns:
        dict with classification results
    """
    base64_image = encode_image(tile)
    
    row, col = position[0] // 256, position[1] // 256
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt.format(row=row, col=col, resolution="10m", region="unknown")
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        result = json.loads(content.strip())
        result['position'] = position
        return result
        
    except Exception as e:
        print(f"Error classifying tile at {position}: {e}")
        return {
            "class": "unknown",
            "confidence": "low",
            "reasoning": f"Error: {str(e)}",
            "position": position
        }

async def classify_tile_adaptive(tile, position, system_prompt, user_prompt, min_tile_size=64):
    """
    Recursively classify tiles using adaptive splitting
    """
    # Base classification
    result = classify_tile(tile, position, system_prompt, user_prompt)
    
    h, w = tile.shape[:2]
    
    # Check if we should split
    # Split if:
    # 1. AI says it's mixed
    # 2. Tile is large enough to split (larger than min_tile_size)
    if result.get('is_mixed', False) and h > min_tile_size and w > min_tile_size:
        print(f"Splitting mixed tile at {position} ({h}x{w})")
        
        # Calculate new dimensions
        mid_h, mid_w = h // 2, w // 2
        row_start, col_start = position[0], position[1]
        
        sub_tiles = []
        # Top-Left
        sub_tiles.append((tile[:mid_h, :mid_w], (row_start, col_start, row_start+mid_h, col_start+mid_w)))
        # Top-Right
        sub_tiles.append((tile[:mid_h, mid_w:], (row_start, col_start+mid_w, row_start+mid_h, position[3])))
        # Bottom-Left
        sub_tiles.append((tile[mid_h:, :mid_w], (row_start+mid_h, col_start, position[2], col_start+mid_w)))
        # Bottom-Right
        sub_tiles.append((tile[mid_h:, mid_w:], (row_start+mid_h, col_start+mid_w, position[2], position[3])))
        
        results = []
        for sub_tile, sub_pos in sub_tiles:
            # Recursive call
            # Note: In a real async env, we would await these. 
            # Since classify_tile is sync for now (calling sync OpenAI), we just call it.
            # If we make classify_tile async, we would await here.
            sub_results = await classify_tile_adaptive(sub_tile, sub_pos, system_prompt, user_prompt, min_tile_size)
            results.extend(sub_results)
            
        return results
    else:
        return [result]

async def classify_all_tiles(tiles, system_prompt, user_prompt, min_tile_size=64):
    """
    Classify all tiles with adaptive refinement
    """
    results = []
    
    for tile, position in tiles:
        # Use adaptive classification
        tile_results = await classify_tile_adaptive(tile, position, system_prompt, user_prompt, min_tile_size)
        results.extend(tile_results)
    
    return results
