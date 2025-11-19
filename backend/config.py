import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables.")

# Constants
DEFAULT_TILE_SIZE = 256
DEFAULT_RESOLUTION = "10m"
MAX_IMAGE_SIZE = 2048
