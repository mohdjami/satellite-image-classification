# 🛰️ Satellite Image Segmentation System

A powerful AI-powered prototype for classifying land cover in satellite imagery using **SegFormer** (Vision Transformer). This system provides high-precision, pixel-wise segmentation for categories like urban areas, forests, water bodies, and agriculture.

![Demo](https://via.placeholder.com/800x400?text=Satellite+Image+Segmentation+Demo)

## ✨ Features

-   **State-of-the-Art AI**: Uses **SegFormer (MiT-B0)** fine-tuned on the DeepGlobe Land Cover dataset.
-   **Pixel-Wise Precision**: Classifies every single pixel, offering far superior detail compared to tile-based methods.
-   **Real-Time Inference**: Optimized for local execution (CPU/MPS), processing images in seconds without external APIs.
-   **Interactive Dashboard**: Built with Streamlit for easy upload, visualization, and statistical analysis.
-   **Auto-Scaling**: Automatically handles images of any resolution by padding and resizing.

## ✅ Completed Objectives

1.  **High-Precision Semantic Segmentation**: Implemented a Vision Transformer (SegFormer) to classify 6 distinct land cover categories with high accuracy.
2.  **Real-Time Inference System**: Developed a lightweight, local inference pipeline using FastAPI and PyTorch.
3.  **Interactive Geospatial Dashboard**: Built a user-friendly UI for instant visualization and automated land use statistics.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **Backend**: FastAPI
-   **AI Model**: Hugging Face Transformers (SegFormer)
-   **Image Processing**: NumPy, PIL, PyTorch

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.9 or higher

### 1. Clone the Repository

```bash
git clone https://github.com/mohdjami/satellite-image-classification.git
cd satellite-image-classification
```

### 2. Set Up Virtual Environment

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

---

## 🏃‍♂️ Running the Application

You need to run both the Backend (API) and the Frontend (UI). Open two terminal windows.

### Step 1: Start the Backend

In your **first terminal**:
```bash
# Make sure venv is activated
source venv/bin/activate

# Run the FastAPI server
uvicorn backend.main:app --reload
```
*Server running at `http://127.0.0.1:8000`*

### Step 2: Start the Frontend

In your **second terminal**:
```bash
# Make sure venv is activated
source venv/bin/activate

# Run the Streamlit app
streamlit run frontend/app.py
```
*Browser opens at `http://localhost:8501`*

---

## 📖 Usage Guide

1.  **Upload**: Go to the **"Upload & Classify"** tab.
2.  **Classify**: Click **"🚀 Classify Image"**. The system will process the image locally.
3.  **Analyze**: View the **Segmentation Map** and **Land Cover Statistics** in the Results tab.
    *   **Urban**: Cyan
    *   **Agriculture**: Yellow
    *   **Rangeland**: Magenta
    *   **Forest**: Green
    *   **Water**: Blue
    *   **Barren**: White

## 📂 Project Structure

```
satellite-segmentation/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── model.py                # SegFormer model wrapper
│   ├── classifier.py           # Inference logic
│   ├── reconstruction.py       # Visualization utilities
│   └── preprocessing.py        # Image utilities
├── frontend/
│   └── app.py                  # Streamlit User Interface
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## 📄 License

[MIT](LICENSE)
