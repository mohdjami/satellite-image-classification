import streamlit as st
import requests
import json
from PIL import Image
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Satellite Image Segmentation",
    page_icon="🛰️",
    layout="wide"
)

# API URL
API_URL = "http://localhost:8000"

st.title("🛰️ Satellite Image Segmentation System")
st.markdown("### AI-Powered Land Cover Classification")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    tile_size = st.slider(
        "Tile Size",
        min_value=64,
        max_value=512,
        value=256,
        step=64,
        help="Larger tiles = fewer API calls but less detail"
    )
    
    resolution = st.selectbox(
        "Image Resolution",
        ["0.25m", "0.5m", "1m", "5m", "10m"],
        index=4
    )
    
    st.markdown("---")
    st.markdown("**Land Cover Categories:**")
    st.markdown("""
    - 🌳 Vegetation
    - 💧 Water
    - 🏢 Buildings
    - 🛣️ Roads
    - 🌾 Agriculture
    - 🏜️ Barren Land
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Classify", "📊 Results", "ℹ️ About"])

with tab1:
    st.header("Upload Satellite Image")
    
    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Supported formats: PNG, JPG, GeoTIFF"
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file:
        # Display original image
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            st.info(f"""
            **Image Info:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Format: {image.format}
            - Mode: {image.mode}
            """)
        
        # Classification button
        if st.button("🚀 Classify Image", type="primary", use_container_width=True):
            with st.spinner("Processing... This may take a few minutes"):
                
                # Send to API
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                
                # Add params to URL since requests.post data/files handling can be tricky with FastAPI
                params = {
                    "tile_size": tile_size,
                    "resolution": resolution
                }
                
                try:
                    response = requests.post(
                        f"{API_URL}/classify",
                        files=files,
                        params=params
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store in session state
                        st.session_state['result'] = result
                        st.session_state['original_image'] = image
                        
                        st.success("✅ Classification completed!")
                        st.balloons()
                    else:
                        st.error(f"Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
                    st.info("Make sure the backend is running: `uvicorn backend.main:app --reload`")

with tab2:
    st.header("Classification Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        original_image = st.session_state['original_image']
        
        # Display segmentation map
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Segmentation Map")
            seg_map_b64 = result['segmentation_map']
            seg_map_bytes = base64.b64decode(seg_map_b64)
            seg_map_img = Image.open(io.BytesIO(seg_map_bytes))
            st.image(seg_map_img, use_container_width=True)
        
        with col2:
            st.subheader("Overlay")
            # Create overlay
            seg_array = np.array(seg_map_img)
            # Resize original to match seg map if needed (should be same size)
            orig_array = np.array(original_image.resize(seg_map_img.size))
            
            # Simple overlay
            overlay = (orig_array * 0.5 + seg_array * 0.5).astype(np.uint8)
            st.image(overlay, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Land Cover Statistics")
        
        stats = result['statistics']
        class_dist = stats['class_distribution']
        
        # Create pie chart
        if class_dist:
            labels = list(class_dist.keys())
            sizes = [v['percentage'] for v in class_dist.values()]
            
            # Map colors
            color_map = {
                'vegetation': 'green',
                'water': 'blue',
                'buildings': 'gray',
                'roads': 'black',
                'agriculture': 'yellow',
                'barren': 'brown',
                'unknown': 'white'
            }
            colors = [color_map.get(l, 'white') for l in labels]
            
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            ax.set_title("Land Cover Distribution")
            st.pyplot(fig)
            
            # Detailed stats
            st.markdown("### Detailed Breakdown")
            cols = st.columns(len(class_dist))
            for i, (cls, data) in enumerate(class_dist.items()):
                with cols[i % 3]: # Wrap around if many columns
                    st.metric(
                        label=cls.capitalize(),
                        value=f"{data['percentage']}%",
                        delta=f"{data['count']} tiles"
                    )
        
        # Download button
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(result, indent=2),
            file_name="segmentation_results.json",
            mime="application/json"
        )
    else:
        st.info("👆 Upload and classify an image in the Upload tab first")

with tab3:
    st.header("About This System")
    
    st.markdown("""
    ## 🛰️ Satellite Image Segmentation System
    
    ### Technology Stack
    - **Frontend:** Streamlit
    - **Backend:** FastAPI
    - **AI Model:** OpenAI GPT-4o-mini (Vision)
    - **Image Processing:** PIL, NumPy, Rasterio
    
    ### How It Works
    
    1. **Upload:** User uploads satellite image
    2. **Tiling:** Image split into tiles (default 256x256)
    3. **Classification:** Each tile analyzed by GPT-4 Vision
    4. **Reconstruction:** Tiles merged into full segmentation map
    5. **Visualization:** Results displayed with statistics
    
    ### Supported Categories
    - Vegetation
    - Water
    - Buildings
    - Roads
    - Agriculture
    - Barren Land
    """)
