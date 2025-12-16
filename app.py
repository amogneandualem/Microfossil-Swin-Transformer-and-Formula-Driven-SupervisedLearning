"""
Microfossil Classifier - Optimized for Streamlit Cloud
Simple and reliable version
"""

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
import os
import time

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Microfossil Classifier",
    page_icon="🔬",
    layout="wide"
)

# Model URL from Hugging Face
HUGGINGFACE_MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/model.pth"
MODEL_PATH = "model.pth"

# Class names (32 classes)
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

# ========== DOWNLOAD MODEL ==========
def download_model():
    """Download model from Hugging Face"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
        return True, f"✅ Model loaded ({file_size:.1f} MB)"
    
    try:
        st.info("📥 Downloading model from Hugging Face... This may take 2-3 minutes.")
        
        # Download with progress
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
            return True, f"✅ Model downloaded ({file_size:.1f} MB)"
        else:
            return False, "❌ Download failed"
            
    except Exception as e:
        return False, f"❌ Error: {str(e)[:100]}"

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load the PyTorch model"""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            success, message = download_model()
            if not success:
                st.error(message)
                return None
        
        # Import timm
        import timm
        
        # Create model
        model = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # Handle state dict
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
                k = k[7:]  # Remove 'module.' prefix
            cleaned_state_dict[k] = v
        
        # Load weights
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)[:200]}")
        return None

# ========== MAIN APP ==========
def main():
    # App header
    st.title("🔬 Microfossil Classifier")
    st.markdown("**AI-powered identification using Swin Transformer**")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model status
        if st.button("Check Model Status", use_container_width=True):
            success, message = download_model()
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # About
        with st.expander("ℹ️ About"):
            st.write("""
            **How to use:**
            1. Upload a microfossil image
            2. Click 'Classify'
            3. View AI predictions
            
            **First time:** Model downloads automatically
            **Model:** Swin-Large (32 classes)
            **Hosted:** Hugging Face
            """)
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            help="Supported: JPG, PNG"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if uploaded_file:
            if st.button("🚀 Classify", type="primary", use_container_width=True):
                with st.spinner("Loading model..."):
                    # Load model
                    if st.session_state.model is None:
                        model = load_model()
                        if model is None:
                            st.error("Failed to load model")
                            return
                        st.session_state.model = model
                    
                    # Transform image
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
                    ])
                    
                    img_tensor = transform(image).unsqueeze(0)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = st.session_state.model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        top_prob, top_idx = torch.max(probs, 1)
                    
                    # Show results
                    idx = top_idx.item()
                    if idx < len(CLASS_NAMES):
                        # Top prediction
                        st.success(f"🎯 **Prediction:** {CLASS_NAMES[idx]}")
                        st.metric("Confidence", f"{top_prob.item()*100:.1f}%")
                        
                        # Confidence indicator
                        confidence = top_prob.item() * 100
                        if confidence > 90:
                            st.success("High confidence")
                        elif confidence > 70:
                            st.warning("Moderate confidence")
                        else:
                            st.info("Low confidence")
                        
                        # Show top 5 predictions
                        st.subheader("📊 Top 5 Predictions:")
                        top_probs, top_indices = torch.topk(probs, 5)
                        
                        for i in range(5):
                            idx_i = top_indices[0][i].item()
                            if idx_i < len(CLASS_NAMES):
                                conf = top_probs[0][i].item() * 100
                                st.write(f"{i+1}. **{CLASS_NAMES[idx_i]}**: {conf:.1f}%")
                    else:
                        st.error("Prediction index out of range")
        else:
            st.info("👆 Upload an image to begin classification")
            
            # Quick stats
            with st.expander("📚 Quick Facts"):
                st.write(f"**Classes:** {len(CLASS_NAMES)}")
                st.write(f"**Model:** Swin-Large Transformer")
                st.write("**Training:** Fine-tuned on microfossils")
                st.write("**First run:** Downloads 349MB model")
    
    # Footer
    st.markdown("---")
    st.caption("Deployed on Streamlit Cloud | Model hosted on Hugging Face")

# ========== RUN APP ==========
if __name__ == "__main__":
    main()
