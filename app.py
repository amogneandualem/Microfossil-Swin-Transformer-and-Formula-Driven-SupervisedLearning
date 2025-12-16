"""
Microfossil Classifier - Streamlit Cloud Optimized
Lightweight version with better error handling
"""

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import requests
import numpy as np
import json
from datetime import datetime
import pandas as pd

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Microfossil Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check if we're in Streamlit Cloud
IS_STREAMLIT_CLOUD = os.environ.get("IS_STREAMLIT_CLOUD", False)

# Hugging Face model URL
HUGGINGFACE_MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/model.pth"
MODEL_PATH = "model.pth"

# Class names
CLASS_NAMES = [
    'Acanthodesmia_micropora', 'Actinomma_leptoderma_boreale',
    'Antarctissa_denticulata-cyrindrica', 'Antarctissa_juvenile',
    'Antarctissa_longa-strelkovi', 'Botryocampe_antarctica',
    'Botryocampe_inflatum-conithorax', 'Ceratocyrtis_historicosus',
    'Cycladophora_bicornis', 'Cycladophora_cornutoides',
    'Cycladophora_davisiana', 'Diatoms',
    'Druppatractus_irregularis-bensoni', 'Eucyrtidium_spp',
    'Fragments', 'Larcids_inner', 'Lithocampe_furcaspiculate',
    'Lithocampe_platycephala', 'Lithomelissa_setosa-borealis',
    'Lophophana_spp', 'Other_Nassellaria', 'Other_Spumellaria',
    'Phormospyris_stabilis_antarctica', 'Phorticym_clevei-pylonium',
    'Plectacantha_oikiskos', 'Pseudodictyophimus_gracilipes',
    'Sethoconus_tablatus', 'Siphocampe_arachnea_group',
    'Spongodiscus', 'Spongurus_pylomaticus',
    'Sylodictya_spp', 'Zygocircus'
]

# ========== MODEL DOWNLOAD (SIMPLIFIED) ==========
def download_model():
    """Download model with minimal dependencies"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
        return True, f"Model loaded ({file_size:.1f} MB)"
    
    try:
        # Create a simple progress indicator
        status_text = st.empty()
        status_text.text("📥 Downloading model from Hugging Face...")
        
        # Download the file
        response = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(MODEL_PATH, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
            status_text.empty()
            return True, f"✅ Model downloaded ({file_size:.1f} MB)"
        else:
            return False, "❌ Download failed"
            
    except Exception as e:
        return False, f"❌ Error: {str(e)[:100]}"

# ========== LOAD MODEL (WITH FALLBACK) ==========
@st.cache_resource
def load_model():
    """Load model with fallback options"""
    try:
        # First try to download
        success, message = download_model()
        if not success:
            st.error(message)
            return None
        
        # Import timm
        try:
            import timm
        except ImportError:
            st.error("Please add 'timm' to requirements.txt")
            return None
        
        # Create model
        model = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load weights
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Clean state dict
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            cleaned_state_dict[k] = v
        
        # Load weights
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)[:200]}")
        return None

# ========== SIMPLE IMAGE PROCESSING ==========
def preprocess_image(image):
    """Simple image preprocessing"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ========== MAIN APP ==========
def main():
    # Custom minimal CSS
    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 10px;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f8ff;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔬 Microfossil Classifier")
    st.markdown("*AI-powered classification using Swin Transformer*")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model status
        if st.button("🔄 Check Model Status"):
            success, message = download_model()
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.99,
            value=0.8,
            step=0.01
        )
        
        # About
        with st.expander("ℹ️ About"):
            st.write("""
            **Model Details:**
            - Swin-Large Transformer
            - 32 microfossil classes
            - Hosted on Hugging Face
            
            **First Run:**
            Model will be downloaded automatically
            (~349 MB, may take a few minutes)
            """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a microfossil image",
            type=['jpg', 'jpeg', 'png'],
            help="JPG or PNG format recommended"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Store image in session state
                st.session_state.image = image
                st.session_state.image_ready = True
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        st.subheader("🔍 Classification Results")
        
        if 'image_ready' in st.session_state and st.session_state.image_ready:
            if st.button("🚀 Classify Image", type="primary"):
                with st.spinner("Processing..."):
                    # Load model if not already loaded
                    if not st.session_state.model_loaded:
                        model = load_model()
                        if model:
                            st.session_state.model = model
                            st.session_state.model_loaded = True
                        else:
                            st.error("Failed to load model. Please check the requirements.")
                            return
                    
                    # Get image from session state
                    image = st.session_state.image
                    
                    # Preprocess
                    img_tensor = preprocess_image(image)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = st.session_state.model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        top_prob, top_idx = torch.max(probs, 1)
                    
                    # Display results
                    idx = top_idx.item()
                    confidence = top_prob.item() * 100
                    
                    if idx < len(CLASS_NAMES):
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🎯 Primary Identification</h3>
                            <h2>{CLASS_NAMES[idx]}</h2>
                            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence indicator
                        if confidence > 90:
                            st.success(f"High confidence ({confidence:.1f}%)")
                        elif confidence > 70:
                            st.warning(f"Moderate confidence ({confidence:.1f}%)")
                        else:
                            st.info(f"Low confidence ({confidence:.1f}%)")
                        
                        # Show top 3 predictions
                        if confidence < confidence_threshold * 100:
                            st.info("⚠️ Confidence below threshold. Consider these alternatives:")
                        
                        top_probs, top_indices = torch.topk(probs, 3)
                        st.subheader("Top 3 Predictions:")
                        
                        for i in range(3):
                            idx_i = top_indices[0][i].item()
                            if idx_i < len(CLASS_NAMES):
                                conf_i = top_probs[0][i].item() * 100
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.write(f"{i+1}. {CLASS_NAMES[idx_i]}")
                                with col_b:
                                    st.write(f"{conf_i:.1f}%")
                        
                        # Export results
                        st.divider()
                        st.subheader("📊 Export Results")
                        
                        results = {
                            "prediction": CLASS_NAMES[idx],
                            "confidence": float(confidence),
                            "top_predictions": [
                                {
                                    "class": CLASS_NAMES[top_indices[0][i].item()],
                                    "confidence": float(top_probs[0][i].item() * 100)
                                } for i in range(3) if top_indices[0][i].item() < len(CLASS_NAMES)
                            ],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Convert to JSON for download
                        json_str = json.dumps(results, indent=2)
                        
                        st.download_button(
                            label="📥 Download Results (JSON)",
                            data=json_str,
                            file_name=f"microfossil_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    else:
                        st.error("Prediction index out of range")
        
        else:
            # Instructions
            st.info("👆 Upload an image to begin classification")
            
            # Quick guide
            with st.expander("📚 Quick Guide"):
                st.write("""
                1. **Upload** a microfossil image
                2. Click **"Classify Image"**
                3. First run downloads the model
                4. View AI predictions
                5. Download results as JSON
                """)
            
            # Stats
            st.metric("Classes", len(CLASS_NAMES))
            st.metric("Model", "Swin-Large")
    
    # Footer
    st.markdown("---")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        st.caption(f"© {datetime.now().year} Microfossil Classifier | Hosted on Streamlit Cloud")
    with col_f2:
        st.caption("Model: Swin Transformer")

# ========== ERROR HANDLING ==========
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("""
        **Troubleshooting steps:**
        1. Check that all dependencies are in requirements.txt
        2. Ensure the model file is available on Hugging Face
        3. Check Streamlit Cloud logs for details
        4. Try reducing the model size if memory is an issue
        """)
