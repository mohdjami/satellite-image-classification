# 🛰️ Satellite Image Segmentation System

A powerful AI-powered prototype for classifying land cover in satellite imagery using OpenAI's GPT-4 Vision. This system uses an **Adaptive Tiling** strategy to accurately segment large satellite images into categories like vegetation, water, buildings, roads, and agriculture.

![Demo](https://via.placeholder.com/800x400?text=Satellite+Image+Segmentation+Demo)

## ✨ Features

-   **AI-Powered Classification**: Uses GPT-4o-mini (Vision) for robust scene understanding.
-   **Adaptive Tiling (Smart Refinement)**: Automatically detects "mixed" tiles (e.g., a river crossing a forest) and recursively splits them into smaller sub-tiles (down to 64px) for high-precision segmentation.
-   **Multi-Resolution Support**: Handles imagery from 0.25m (high res) to 10m (Sentinel-2).
-   **Interactive UI**: Built with Streamlit for easy upload, visualization, and analysis.
-   **FastAPI Backend**: Asynchronous backend for efficient processing.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **Backend**: FastAPI
-   **AI Engine**: OpenAI API (GPT-4o-mini)
-   **Image Processing**: NumPy, PIL, Rasterio

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.9 or higher
-   An OpenAI API Key (with access to Vision models)

### 1. Clone the Repository

```bash
git clone https://github.com/mohdjami/satellite-image-classification.git
cd satellite-image-classification
```

### 2. Set Up Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1.  Create a `.env` file in the root directory (you can copy the example):
    ```bash
    cp .env.example .env
    ```
2.  Open `.env` and add your OpenAI API Key:
    ```env
    OPENAI_API_KEY=sk-your-api-key-here
    ```

---

## 🏃‍♂️ Running the Application

You need to run both the Backend (API) and the Frontend (UI). It's best to open two terminal windows.

### Step 1: Start the Backend

In your **first terminal**:
```bash
# Make sure venv is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the FastAPI server
uvicorn backend.main:app --reload
```
*You should see output indicating the server is running at `http://127.0.0.1:8000`.*

### Step 2: Start the Frontend

In your **second terminal**:
```bash
# Make sure venv is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the Streamlit app
streamlit run frontend/app.py
```
*This will automatically open your web browser to `http://localhost:8501`.*

---

## 📖 Usage Guide

1.  **Upload**: Go to the **"Upload & Classify"** tab in the web interface.
2.  **Configure**:
    *   **Tile Size**: Choose the starting tile size (default 256). Smaller tiles = more detail but more API calls.
    *   **Resolution**: Select the approximate resolution of your image (e.g., `0.25m` for high-res Google Earth style, `10m` for Sentinel/Landsat).
3.  **Classify**: Click the **"🚀 Classify Image"** button.
    *   *Note*: The system will automatically refine "mixed" areas by splitting them into smaller tiles.
4.  **Analyze**: Switch to the **"📊 Results"** tab to see:
    *   **Segmentation Map**: The color-coded classification.
    *   **Overlay**: The map overlaid on your original image.
    *   **Statistics**: Percentage breakdown of land cover types.
5.  **Download**: You can download the raw JSON results for further analysis.

## 📂 Project Structure

```
satellite-segmentation/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── classifier.py           # OpenAI integration & Adaptive Tiling logic
│   ├── preprocessing.py        # Image tiling & normalization
│   ├── reconstruction.py       # Merging tiles & visualization
│   ├── config.py               # Settings
│   └── prompts/
│       └── classification_prompt.py  # System prompts for GPT-4
├── frontend/
│   └── app.py                  # Streamlit User Interface
├── data/                       # Directory for sample data & outputs
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## ❓ Troubleshooting

*   **`ConnectionError`**: Ensure the backend (`uvicorn`) is running on port 8000.
*   **`OpenAI Error`**: Check your API key in `.env`. Ensure you have credits/quota available.
*   **Black Output / 0 Tiles**: This usually happens with very small images. The system now auto-pads them, but try using a larger image if issues persist.
*   **Slow Processing**: Large images with "Adaptive Tiling" can generate many API calls. Try increasing the "Tile Size" slider or using a smaller image crop.

## 📄 License

[MIT](LICENSE)
