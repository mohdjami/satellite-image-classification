# Satellite Image Segmentation Prototype

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration:**
    - Copy `.env.example` to `.env`
    - Add your OpenAI API Key to `.env`

4.  **Run the Backend:**
    ```bash
    uvicorn backend.main:app --reload
    ```

5.  **Run the Frontend:**
    ```bash
    streamlit run frontend/app.py
    ```
