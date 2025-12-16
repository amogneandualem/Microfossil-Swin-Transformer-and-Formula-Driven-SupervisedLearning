"""
Microfossil Classifier - Full Working App
Downloads model from: https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Microfossil Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your Hugging Face model URL
HUGGINGFACE_MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

# Class names (32 classes - update these if your model has different classes)
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
    """Download model directly from Hugging Face URL"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
        return True, f"✅ Model already downloaded ({file_size:.1f} MB)", MODEL_PATH
    
    try:
        # Show download progress
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        status_text.text("📥 Downloading model from Hugging Face...")
        
        # Download the model
        response = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(MODEL_PATH, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(min(progress, 1.0))
                        
                        # Update status text
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        status_text.text(f"📥 {mb_downloaded:.1f}/{mb_total:.1f} MB")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
            return True, f"✅ Model downloaded successfully! ({file_size:.1f} MB)", MODEL_PATH
        else:
            return False, "❌ Download failed - file not created", None
            
    except Exception as e:
        return False, f"❌ Download error: {str(e)[:100]}", None

# ========== LOAD PYTORCH MODEL ==========
@st.cache_resource
def load_pytorch_model():
    """Load the PyTorch model from downloaded file"""
    try:
        # First ensure model is downloaded
        success, message, model_path = download_model_from_huggingface()
        if not success:
            st.error(message)
            return None
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")
            return None
        
        # Import timm for Swin Transformer
        try:
            import timm
        except ImportError:
            st.error("Please add 'timm' to requirements.txt")
            st.info("Add this line to requirements.txt: timm==0.9.2")
            return None
        
        # Determine model architecture (Swin Large based on your previous code)
        model = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load the model checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Try different possible state dict keys
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the entire dict is the state dict
                state_dict = checkpoint
        else:
            # Checkpoint is directly the state dict
            state_dict = checkpoint
        
        # Clean state dict keys (remove 'module.' prefix if present)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Remove DataParallel wrapper prefix if exists
            if key.startswith('module.'):
                key = key[7:]
            cleaned_state_dict[key] = value
        
        # Load the state dict
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()  # Set to evaluation mode
        
        st.sidebar.success("✅ Model loaded successfully")
        return model
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)[:200]}")
        return None

# ========== IMAGE PREPROCESSING ==========
def preprocess_image(image):
    """Preprocess image for Swin Transformer"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Swin expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert PIL Image to tensor
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)  # Add batch dimension

# ========== PREDICTION FUNCTION ==========
def predict_image(model, image_tensor, top_k=5):
    """Make prediction on image tensor"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        
        # Convert to numpy arrays
        top_probs = top_probs.squeeze().numpy()
        top_indices = top_indices.squeeze().numpy()
        
        return top_probs, top_indices

# ========== MAIN APP ==========
def main():
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f8ff;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .confidence-bar {
        height: 20px;
        background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<h1 class="main-header">🔬 Microfossil Classification System</h1>', unsafe_allow_html=True)
    st.markdown(f"**Model URL:** `{HUGGINGFACE_MODEL_URL}`")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model status
        st.subheader("Model Status")
        if st.button("🔄 Check/Download Model", use_container_width=True):
            success, message, _ = download_model_from_huggingface()
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # Display model info if exists
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
            st.info(f"**Model size:** {file_size:.1f} MB")
            st.info(f"**Last modified:** {datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Prediction settings
        st.subheader("Prediction Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=50,
            max_value=99,
            value=75,
            help="Minimum confidence to show prediction"
        )
        
        top_k = st.slider(
            "Show Top N Predictions",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of top predictions to display"
        )
        
        # About section
        with st.expander("ℹ️ About This App", expanded=True):
            st.write(f"""
            **Model Source:** Hugging Face
            **URL:** {HUGGINGFACE_MODEL_URL}
            
            **Features:**
            - Downloads model directly from Hugging Face
            - Caches model for faster subsequent runs
            - Shows confidence scores for predictions
            - Displays top N predictions
            
            **Classes:** {len(CLASS_NAMES)} microfossil types
            """)
    
    # Main content - Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # Upload methods
        upload_method = st.radio(
            "Select input method:",
            ["Upload File", "Camera", "URL"],
            horizontal=True
        )
        
        image = None
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a microfossil image",
                type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
                help="Supported formats: JPG, PNG, TIFF, BMP"
            )
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        
        elif upload_method == "Camera":
            camera_image = st.camera_input("Take a picture")
            if camera_image is not None:
                image = Image.open(camera_image)
        
        elif upload_method == "URL":
            image_url = st.text_input("Enter image URL")
            if image_url:
                try:
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        image = Image.open(response.raw)
                        st.image(image, caption="Image from URL", use_container_width=True)
                    else:
                        st.error(f"Failed to fetch image. Status code: {response.status_code}")
                except Exception as e:
                    st.error(f"Error loading image from URL: {str(e)}")
        
        # Store image in session state
        if image is not None:
            st.session_state.current_image = image
            st.session_state.has_image = True
    
    with col2:
        st.subheader("🔍 Classification Results")
        
        if 'has_image' in st.session_state and st.session_state.has_image:
            if st.button("🚀 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    # Load model if not already loaded
                    if not st.session_state.model_loaded:
                        model = load_pytorch_model()
                        if model is None:
                            st.error("Failed to load model. Please check the console for errors.")
                            return
                        st.session_state.model = model
                        st.session_state.model_loaded = True
                    
                    # Get image from session state
                    image = st.session_state.current_image
                    
                    # Preprocess image
                    img_tensor = preprocess_image(image)
                    
                    # Make prediction
                    top_probs, top_indices = predict_image(
                        st.session_state.model, 
                        img_tensor, 
                        top_k=top_k
                    )
                    
                    # Display results
                    st.markdown("---")
                    
                    # Top prediction with confidence
                    top_class = CLASS_NAMES[top_indices[0]]
                    top_confidence = top_probs[0] * 100
                    
                    # Color code based on confidence
                    if top_confidence > 90:
                        confidence_color = "🟢"
                        confidence_text = "High confidence"
                    elif top_confidence > 70:
                        confidence_color = "🟡"
                        confidence_text = "Moderate confidence"
                    else:
                        confidence_color = "🟠"
                        confidence_text = "Low confidence"
                    
                    # Display top result
                    st.markdown(f"""
                    <div class="result-box">
                        <h3>🎯 Primary Identification</h3>
                        <h2>{top_class}</h2>
                        <p><strong>Confidence:</strong> {confidence_color} {top_confidence:.1f}% ({confidence_text})</p>
                        <div class="confidence-bar" style="width: {min(top_confidence, 100)}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show all top predictions in a table
                    st.subheader(f"📊 Top {top_k} Predictions")
                    
                    results_data = []
                    for i in range(len(top_indices)):
                        class_idx = top_indices[i]
                        if class_idx < len(CLASS_NAMES):
                            confidence = top_probs[i] * 100
                            results_data.append({
                                'Rank': i+1,
                                'Class': CLASS_NAMES[class_idx],
                                'Confidence (%)': f"{confidence:.1f}%",
                                'Confidence Value': confidence
                            })
                    
                    # Create DataFrame for display
                    results_df = pd.DataFrame(results_data)
                    
                    # Display as table with formatting
                    st.dataframe(
                        results_df[['Rank', 'Class', 'Confidence (%)']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualize confidence scores
                    st.subheader("📈 Confidence Visualization")
                    
                    # Create a simple bar chart using matplotlib
                    try:
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        y_pos = np.arange(len(results_data))
                        
                        # Get confidence values for bar heights
                        conf_values = [r['Confidence Value'] for r in results_data]
                        class_names = [r['Class'] for r in results_data]
                        
                        # Create horizontal bar chart
                        bars = ax.barh(y_pos, conf_values, color='#1f77b4')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(class_names)
                        ax.invert_yaxis()  # Highest confidence at top
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Prediction Confidence Scores')
                        
                        # Add value labels on bars
                        for i, (bar, val) in enumerate(zip(bars, conf_values)):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                                   f'{val:.1f}%', va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except ImportError:
                        # Fallback if matplotlib is not available
                        st.info("Install matplotlib for confidence visualization")
                    
                    # Export results
                    st.markdown("---")
                    st.subheader("📥 Export Results")
                    
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        # Export as CSV
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv_data,
                            file_name=f"microfossil_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col_export2:
                        # Export as JSON
                        json_data = {
                            'timestamp': datetime.now().isoformat(),
                            'image_processed': True,
                            'top_prediction': {
                                'class': top_class,
                                'confidence': float(top_confidence)
                            },
                            'all_predictions': [
                                {
                                    'class': CLASS_NAMES[top_indices[i]],
                                    'confidence': float(top_probs[i] * 100)
                                } for i in range(len(top_indices))
                            ]
                        }
                        
                        import json as json_module
                        json_str = json_module.dumps(json_data, indent=2)
                        
                        st.download_button(
                            label="💾 Download JSON",
                            data=json_str,
                            file_name=f"microfossil_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        
        else:
            # Welcome message when no image is uploaded
            st.info("👆 Upload an image to begin classification")
            
            # Quick guide
            with st.expander("📚 Quick Start Guide", expanded=True):
                st.write("""
                1. **Upload** a microfossil image using any method
                2. Click **"Classify Image"** button
                3. **First run:** Model downloads automatically from Hugging Face
                4. View **AI predictions** with confidence scores
                5. **Export** results as CSV or JSON
                
                **Note:** First download may take 2-3 minutes depending on model size.
                """)
            
            # Display class count
            st.metric("Available Classes", len(CLASS_NAMES))
    
    # Footer
    st.markdown("---")
    st.caption(f"🔬 Microfossil Classifier • Model from Hugging Face • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ========== ERROR HANDLING & EXECUTION ==========
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Check your internet connection
        2. Verify the Hugging Face URL is accessible
        3. Ensure all dependencies are installed
        4. Check console for detailed error messages
        """)
