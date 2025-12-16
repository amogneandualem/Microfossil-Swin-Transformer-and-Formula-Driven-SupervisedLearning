"""
Microfossil Classifier - GitHub + Streamlit Cloud
Model hosted on Hugging Face
"""

import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import requests
import time

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Microfossil Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONFIGURATION ==========
# Your model on Hugging Face
HUGGINGFACE_MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/model.pth"
MODEL_PATH = "model.pth"

# Use SWIN-LARGE (not BASE) since that's what you trained
MODEL_NAME = "swin_large_patch4_window7_224"
IMAGE_SIZE = 224

# Correct 32 class names (from your training)
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

# ========== DOWNLOAD MODEL FROM HUGGING FACE ==========
@st.cache_resource
def download_model_from_huggingface():
    """Download model from Hugging Face if not exists"""
    if not os.path.exists(MODEL_PATH):
        try:
            st.info("📥 Downloading model from Hugging Face...")
            
            # Download with progress bar
            response = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with st.spinner(f"Downloading {total_size/1024/1024:.1f} MB..."):
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            st.success(f"✅ Model downloaded: {MODEL_PATH}")
            return MODEL_PATH
            
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            return None
    
    return MODEL_PATH

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load the Swin-Large model"""
    try:
        # Download model first
        model_path = download_model_from_huggingface()
        if not model_path:
            return None
        
        # Create model - MUST BE SWIN-LARGE
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Clean keys (remove DataParallel prefix if present)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            cleaned_state_dict[k] = v
        
        # Load weights with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            st.warning(f"⚠️ {len(missing_keys)} keys missing (using defaults)")
        if unexpected_keys:
            st.info(f"ℹ️ {len(unexpected_keys)} unexpected keys ignored")
        
        model.eval()
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

# ========== IMAGE TRANSFORMS ==========
def get_transforms():
    """Get image preprocessing transforms (matches training)"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# ========== PREDICTION FUNCTION ==========
def predict_image(model, device, image):
    """Make prediction on single image"""
    transform = get_transforms()
    
    # Convert to PIL if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, min(5, len(CLASS_NAMES)))
    
    # Format results
    predictions = []
    for i in range(len(top_indices[0])):
        idx = top_indices[0][i].item()
        if idx < len(CLASS_NAMES):
            predictions.append({
                'class': CLASS_NAMES[idx],
                'confidence': top_probs[0][i].item() * 100,
                'rank': i + 1
            })
    
    return predictions

# ========== MAIN APP ==========
def main():
    # Header
    st.title("🔬 Microfossil Classification System")
    st.markdown("**Swin-Large Transformer | 32 Classes | Model Hosted on Hugging Face**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
        st.title("⚙️ Settings")
        
        # Model status
        st.subheader("Model Status")
        if st.button("🔄 Reload Model", type="secondary"):
            st.cache_resource.clear()
            st.rerun()
        
        # System info
        st.subheader("System Info")
        st.write(f"PyTorch: {torch.__version__}")
        st.write(f"Device: {'GPU 🚀' if torch.cuda.is_available() else 'CPU ⚡'}")
        
        # About
        with st.expander("ℹ️ About This App"):
            st.write("""
            **Features:**
            - AI-powered microfossil classification
            - 32 distinct microfossil classes
            - Swin-Large Transformer architecture
            - Model hosted on Hugging Face
            
            **How to use:**
            1. Upload a microfossil image
            2. Click 'Classify Image'
            3. View predictions with confidence scores
            4. See top 5 possible classifications
            """)
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        upload_method = st.radio(
            "Select input method:",
            ["Upload File", "Drag & Drop"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        uploaded_file = st.file_uploader(
            "Choose a microfossil image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, PNG, BMP. Clear images work best."
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Image info
                with st.expander("📊 Image Details"):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"**Name:** {uploaded_file.name}")
                        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                    with col_info2:
                        st.write(f"**Dimensions:** {image.size[0]}×{image.size[1]}")
                        st.write(f"**Format:** {image.format}")
            
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                image = None
        else:
            image = None
            st.info("👆 Upload an image to begin classification")
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if image:
            if st.button("🚀 Classify Image", type="primary", use_container_width=True):
                with st.spinner("🧠 Loading model from Hugging Face..."):
                    start_time = time.time()
                    
                    # Load model
                    model, device = load_model()
                    
                    if model is None or device is None:
                        st.error("Failed to load model. Please try again.")
                        return
                    
                    # Make prediction
                    predictions = predict_image(model, device, image)
                    inference_time = time.time() - start_time
                    
                    if predictions:
                        top_pred = predictions[0]
                        
                        # Display main prediction
                        st.markdown(f"""
                        <div style='
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 25px;
                            border-radius: 15px;
                            color: white;
                            margin: 20px 0;
                        '>
                            <h2 style='margin: 0; font-size: 28px;'>{top_pred['class'].replace('_', ' ')}</h2>
                            <p style='font-size: 18px; margin: 10px 0 0 0;'>
                                Confidence: <strong>{top_pred['confidence']:.2f}%</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence visualization
                        confidence = top_pred['confidence']
                        st.progress(confidence / 100, text=f"Confidence: {confidence:.1f}%")
                        
                        # Confidence assessment
                        if confidence >= 90:
                            st.success(f"✅ High confidence prediction ({confidence:.1f}%)")
                        elif confidence >= 70:
                            st.warning(f"⚠️ Moderate confidence ({confidence:.1f}%)")
                        else:
                            st.info(f"ℹ️ Low confidence ({confidence:.1f}%) - manual verification recommended")
                        
                        # Top 5 predictions
                        st.subheader("🏆 Top 5 Predictions")
                        for pred in predictions:
                            cols = st.columns([3, 1])
                            with cols[0]:
                                st.write(f"{pred['rank']}. {pred['class'].replace('_', ' ')}")
                            with cols[1]:
                                st.write(f"**{pred['confidence']:.1f}%**")
                        
                        # Performance metrics
                        with st.expander("📈 Performance Details"):
                            col_metric1, col_metric2, col_metric3 = st.columns(3)
                            with col_metric1:
                                st.metric("Inference Time", f"{inference_time:.2f}s")
                            with col_metric2:
                                st.metric("Model", "Swin-Large")
                            with col_metric3:
                                st.metric("Device", "GPU" if device.type == "cuda" else "CPU")
                        
                        # Export results
                        st.subheader("💾 Export Results")
                        
                        results_text = f"Microfossil Classification Results\n"
                        results_text += f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        results_text += f"Model: {MODEL_NAME}\n"
                        results_text += f"Image: {uploaded_file.name}\n"
                        results_text += f"Inference Time: {inference_time:.2f}s\n\n"
                        results_text += "Predictions:\n"
                        
                        for pred in predictions:
                            results_text += f"{pred['rank']}. {pred['class']}: {pred['confidence']:.2f}%\n"
                        
                        st.download_button(
                            label="📥 Download Results (TXT)",
                            data=results_text,
                            file_name=f"microfossil_result_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
        
        elif not image and uploaded_file:
            st.warning("Could not process the uploaded image")
        else:
            # Welcome message
            st.info("👆 **Upload an image to get started**")
            
            with st.expander("📚 Quick Guide"):
                st.write("""
                1. **Upload** a microfossil image using the uploader
                2. Click **"Classify Image"** button
                3. View **AI prediction** with confidence score
                4. See **top 5** possible classifications
                5. **Download** results for records
                
                **Tips for best results:**
                - Use clear, well-focused images
                - Ensure good lighting
                - Upload images of individual microfossils
                - Supported formats: JPG, PNG, BMP
                """)
            
            # Show available classes
            with st.expander("📋 Classification Categories (32 total)"):
                categories = {
                    "Spumellaria": ["Actinomma", "Spongodiscus", "Spongurus", "Sylodictya"],
                    "Nassellaria": ["Lithomelissa", "Lophophana", "Ceratocyrtis", "Cycladophora"],
                    "Antarctissa": ["Antarctissa denticulata", "Antarctissa juvenile", "Antarctissa longa"],
                    "Others": ["Diatoms", "Fragments", "Eucyrtidium", "Zygocircus"]
                }
                
                for category, examples in categories.items():
                    st.write(f"**{category}**")
                    for ex in examples:
                        st.write(f"  • {ex}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "🔬 Microfossil Classifier | Model: Swin-Large | Hosted on Hugging Face | "
        "Deployed via GitHub + Streamlit Cloud"
        "</div>",
        unsafe_allow_html=True
    )

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    main()
