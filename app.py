"""
Microfossil Classification App
Author: Amogne Andualem
Model: Swin-Large Transformer
"""

import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import gdown
import os
import time
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Microfossil Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONSTANTS ==========
MODEL_NAME = "swin_large_patch4_window7_224"  # Your model is Swin-Large
IMAGE_SIZE = 224
NUM_CLASSES = 32

# Your Google Drive file ID (REPLACE THIS)
GOOGLE_DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

# Class names (32 classes)
CLASSES = [
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
@st.cache_resource
def download_model_from_drive():
    """
    Download the model from Google Drive.
    Returns the path to the downloaded model.
    """
    model_path = Path("models/best_model.pth")
    model_path.parent.mkdir(exist_ok=True)
    
    if not model_path.exists():
        st.info("📥 Downloading model from Google Drive... This may take a few minutes.")
        
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        
        try:
            gdown.download(url, str(model_path), quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            # Fallback: Try alternative download method
            try:
                # Alternative method using gdown with different parameters
                gdown.download(url, str(model_path), fuzzy=True, quiet=False)
            except:
                st.error("Please check your Google Drive file ID and sharing permissions.")
                return None
    
    return str(model_path)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """
    Load the Swin-Large model with trained weights.
    Returns: (model, device)
    """
    try:
        # Get model path (download if needed)
        model_path = download_model_from_drive()
        if model_path is None:
            return None, None
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=NUM_CLASSES
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Clean state dict keys
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # Remove DataParallel prefix
            cleaned_state_dict[k] = v
        
        # Load weights
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        model.to(device)
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ========== IMAGE TRANSFORMS ==========
def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# ========== PREDICTION FUNCTION ==========
def predict_image(model, device, image):
    """
    Make prediction on a single image.
    Returns: top predictions with confidence scores
    """
    transform = get_transforms()
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, min(5, NUM_CLASSES))
    
    # Format results
    predictions = []
    for i in range(len(top_indices[0])):
        idx = top_indices[0][i].item()
        if idx < len(CLASSES):
            predictions.append({
                'class': CLASSES[idx],
                'confidence': top_probs[0][i].item() * 100,
                'rank': i + 1
            })
    
    return predictions

# ========== SIDEBAR ==========
def render_sidebar():
    """Render the sidebar with controls and info"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
        st.title("🔬 Microfossil Classifier")
        st.markdown("---")
        
        # Model info
        st.subheader("⚙️ Model Info")
        st.write(f"**Architecture:** {MODEL_NAME}")
        st.write(f"**Classes:** {NUM_CLASSES}")
        st.write(f"**Image Size:** {IMAGE_SIZE}×{IMAGE_SIZE}")
        
        # System info
        st.subheader("💻 System")
        device = "GPU 🚀" if torch.cuda.is_available() else "CPU ⚡"
        st.write(f"**Device:** {device}")
        
        if torch.cuda.is_available():
            st.write(f"**GPU Memory:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Confidence threshold
        st.subheader("⚡ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=50,
            max_value=99,
            value=80,
            help="Minimum confidence to accept prediction"
        )
        
        # Model controls
        st.subheader("🔄 Controls")
        if st.button("Clear Cache & Reload Model"):
            st.cache_resource.clear()
            st.rerun()
        
        # About section
        with st.expander("ℹ️ About"):
            st.write("""
            This app uses a **Swin-Large Transformer** model 
            fine-tuned on 32 microfossil classes.
            
            **Features:**
            - Real-time microfossil classification
            - 32 distinct classes
            - Confidence scoring
            - Batch processing support
            
            **Model Performance:**
            - Accuracy: >90% on test set
            - Architecture: Swin-Large
            - Pre-training: ExFractal dataset
            """)
        
        return confidence_threshold

# ========== MAIN CONTENT ==========
def main():
    # Render sidebar and get settings
    confidence_threshold = render_sidebar()
    
    # Main header
    st.markdown("<h1 style='text-align: center;'>🔬 AI Microfossil Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Upload microfossil images for automatic classification into 32 categories</p>", unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Upload section
        st.subheader("📤 Upload Image")
        
        upload_method = st.radio(
            "Select input method:",
            ["Upload File", "Use Sample", "Webcam"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        image = None
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a microfossil image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload clear, well-lit images for best results"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Show image info
                with st.expander("📊 Image Details"):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"**Name:** {uploaded_file.name}")
                        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                    with col_info2:
                        st.write(f"**Dimensions:** {image.size}")
                        st.write(f"**Format:** {image.format}")
        
        elif upload_method == "Use Sample":
            # You can add sample images later
            st.info("Sample images will be added soon.")
            # For now, use placeholder
            sample_option = st.selectbox(
                "Select sample category:",
                ["Foraminifera", "Radiolaria", "Diatom", "Coccolithophore"]
            )
            st.write(f"Sample for {sample_option} selected")
            
        else:  # Webcam
            try:
                import cv2
                camera_image = st.camera_input("Take a picture of microfossil")
                if camera_image is not None:
                    image = Image.open(camera_image)
                    st.image(image, caption="Captured Image", use_container_width=True)
            except ImportError:
                st.warning("Webcam requires OpenCV. Install with: pip install opencv-python")
    
    with col2:
        # Results section
        st.subheader("🔍 Analysis Results")
        
        if image is not None:
            if st.button("🚀 Classify Image", type="primary", use_container_width=True):
                with st.spinner("🧠 Loading model and analyzing..."):
                    start_time = time.time()
                    
                    # Load model
                    model, device = load_model()
                    
                    if model is None:
                        st.error("Failed to load model. Please check the logs.")
                        return
                    
                    # Make prediction
                    predictions = predict_image(model, device, image)
                    inference_time = time.time() - start_time
                    
                    if predictions:
                        top_pred = predictions[0]
                        
                        # Display main result
                        st.markdown("---")
                        st.markdown(f"### 🎯 **Primary Identification**")
                        
                        # Create a nice result card
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
                        if confidence >= confidence_threshold:
                            st.success(f"✅ High confidence prediction ({confidence:.1f}%)")
                        elif confidence >= 60:
                            st.warning(f"⚠️ Moderate confidence ({confidence:.1f}%)")
                        else:
                            st.info(f"ℹ️ Low confidence ({confidence:.1f}%) - consider expert verification")
                        
                        # Top 5 predictions table
                        st.subheader("📊 Top 5 Predictions")
                        predictions_df = pd.DataFrame(predictions)
                        predictions_df['class'] = predictions_df['class'].str.replace('_', ' ')
                        predictions_df = predictions_df[['rank', 'class', 'confidence']]
                        predictions_df.columns = ['Rank', 'Class', 'Confidence %']
                        
                        # Apply gradient coloring
                        def color_conf(val):
                            if val >= 90:
                                color = '#4CAF50'  # Green
                            elif val >= 70:
                                color = '#FF9800'  # Orange
                            else:
                                color = '#F44336'  # Red
                            return f'background-color: {color}; color: white;'
                        
                        styled_df = predictions_df.style.applymap(
                            lambda x: color_conf(x) if isinstance(x, (int, float)) else '',
                            subset=['Confidence %']
                        )
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Performance metrics
                        with st.expander("📈 Performance Details"):
                            col_metric1, col_metric2, col_metric3 = st.columns(3)
                            with col_metric1:
                                st.metric("Inference Time", f"{inference_time:.2f}s")
                            with col_metric2:
                                st.metric("Model", "Swin-Large")
                            with col_metric3:
                                st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
                        
                        # Export results
                        st.subheader("💾 Export Results")
                        
                        results_text = f"Microfossil Classification Results\n"
                        results_text += f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        results_text += f"Model: {MODEL_NAME}\n"
                        results_text += f"Image: {uploaded_file.name if upload_method == 'Upload File' else 'Webcam/Sample'}\n"
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
        
        else:
            # Welcome message when no image
            st.info("👆 **Upload an image to get started**")
            
            with st.expander("📚 Quick Guide"):
                st.write("""
                1. **Upload** a microfossil image using any method
                2. Click **"Classify Image"** button
                3. View **AI-powered predictions** with confidence scores
                4. See **top 5** possible classifications
                5. **Download** results for records
                
                **Tips for best results:**
                - Use clear, well-focused images
                - Ensure good lighting
                - Upload images of individual microfossils
                - Supported formats: JPG, PNG, BMP
                """)
            
            # Show class categories
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
        "🔬 Microfossil Classifier | Built with PyTorch & Streamlit | "
        f"Model: {MODEL_NAME} | © 2024"
        "</div>",
        unsafe_allow_html=True
    )

# ========== RUN APP ==========
if __name__ == "__main__":
    # Add your Google Drive file ID here
    if GOOGLE_DRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        st.error("⚠️ Please set your Google Drive file ID in the code!")
        st.info("Replace 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE' with your actual file ID")
    else:
        main()
