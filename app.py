"""
Microfossil Classifier - Streamlit Cloud
Enhanced version with more features and better UX
"""

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import requests
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
import io

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Microfossil Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your Hugging Face Space URL
HUGGINGFACE_MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/model.pth"
MODEL_PATH = "model.pth"
CONFIG_PATH = "model_config.json"

# Enhanced class names with descriptions
CLASS_INFO = {
    'Acanthodesmia_micropora': {"group": "Nassellaria", "description": "Bell-shaped cephalis with delicate spines"},
    'Actinomma_leptoderma_boreale': {"group": "Spumellaria", "description": "Spherical with delicate cortical shell"},
    'Antarctissa_denticulata-cyrindrica': {"group": "Spumellaria", "description": "Cylindrical with denticulate margin"},
    'Antarctissa_juvenile': {"group": "Spumellaria", "description": "Juvenile specimens"},
    'Antarctissa_longa-strelkovi': {"group": "Spumellaria", "description": "Elongated cylindrical form"},
    'Botryocampe_antarctica': {"group": "Nassellaria", "description": "Cluster-shaped with multiple segments"},
    'Botryocampe_inflatum-conithorax': {"group": "Nassellaria", "description": "Inflated conical thorax"},
    'Ceratocyrtis_historicosus': {"group": "Nassellaria", "description": "Horned cephalis with pores"},
    'Cycladophora_bicornis': {"group": "Nassellaria", "description": "Two-horned cephalis"},
    'Cycladophora_cornutoides': {"group": "Nassellaria", "description": "Horn-like projections"},
    'Cycladophora_davisiana': {"group": "Nassellaria", "description": "Well-known stratigraphic marker"},
    'Diatoms': {"group": "Other", "description": "Photosynthetic microorganisms"},
    'Druppatractus_irregularis-bensoni': {"group": "Spumellaria", "description": "Irregular ellipsoidal form"},
    'Eucyrtidium_spp': {"group": "Nassellaria", "description": "Various conical forms"},
    'Fragments': {"group": "Fragment", "description": "Broken or incomplete specimens"},
    'Larcids_inner': {"group": "Spumellaria", "description": "Inner shell of larcid forms"},
    'Lithocampe_furcaspiculate': {"group": "Nassellaria", "description": "Forked terminal spines"},
    'Lithocampe_platycephala': {"group": "Nassellaria", "description": "Flat-headed cephalis"},
    'Lithomelissa_setosa-borealis': {"group": "Nassellaria", "description": "Bristle-like spines"},
    'Lophophana_spp': {"group": "Nassellaria", "description": "Crested cephalis forms"},
    'Other_Nassellaria': {"group": "Nassellaria", "description": "Other nassellarian forms"},
    'Other_Spumellaria': {"group": "Spumellaria", "description": "Other spumellarian forms"},
    'Phormospyris_stabilis_antarctica': {"group": "Nassellaria", "description": "Stable antarctic form"},
    'Phorticym_clevei-pylonium': {"group": "Nassellaria", "description": "Mesh-like structure"},
    'Plectacantha_oikiskos': {"group": "Nassellaria", "description": "Basket-like structure"},
    'Pseudodictyophimus_gracilipes': {"group": "Nassellaria", "description": "Slender feet/pores"},
    'Sethoconus_tablatus': {"group": "Nassellaria", "description": "Table-like conical form"},
    'Siphocampe_arachnea_group': {"group": "Nassellaria", "description": "Spider-web like structure"},
    'Spongodiscus': {"group": "Spumellaria", "description": "Spongy disc-shaped"},
    'Spongurus_pylomaticus': {"group": "Spumellaria", "description": "Spongy cylindrical form"},
    'Sylodictya_spp': {"group": "Spumellaria", "description": "Net-like skeletal structure"},
    'Zygocircus': {"group": "Nassellaria", "description": "Circular or ring-shaped"}
}

CLASS_NAMES = list(CLASS_INFO.keys())

# ========== DOWNLOAD MODEL FROM HUGGING FACE ==========
@st.cache_resource
def download_model():
    """Download model from Hugging Face with progress bar"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
        st.sidebar.success(f"✅ Model ready: {file_size:.1f} MB")
        return MODEL_PATH
    
    st.sidebar.warning("⚠️ Downloading model from Hugging Face...")
    
    try:
        # Create progress bar
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Download model
        response = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(MODEL_PATH, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = downloaded / total_size
                    progress_bar.progress(min(progress, 1.0))
                    mb_downloaded = downloaded / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    status_text.text(f"📥 {mb_downloaded:.1f}/{mb_total:.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
            st.sidebar.success(f"✅ Model downloaded! ({file_size:.1f} MB)")
            return MODEL_PATH
        else:
            st.sidebar.error("❌ Download failed")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:100]}")
        return None

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load the PyTorch model"""
    try:
        # Download model first
        model_path = download_model()
        if not model_path:
            return None
        
        # Import timm
        import timm
        
        # Create Swin Large model
        model = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
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
        
        # Clean state dict keys
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
        st.error(f"❌ Model loading failed: {str(e)[:200]}")
        return None

# ========== IMAGE PREPROCESSING ==========
def preprocess_image(image):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ========== VISUALIZATION FUNCTIONS ==========
def create_confidence_chart(probabilities, top_n=10):
    """Create horizontal bar chart of top predictions"""
    probs, indices = torch.topk(probabilities, top_n)
    probs = probs.squeeze().numpy() * 100
    indices = indices.squeeze().numpy()
    
    labels = [CLASS_NAMES[i] for i in indices]
    colors = ['#1f77b4' if i < 3 else '#7f7f7f' for i in range(len(labels))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    
    ax.barh(y_pos, probs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top Predictions')
    
    # Add value labels
    for i, v in enumerate(probs):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    return fig

# ========== MAIN APP ==========
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 1rem 0;
    }
    .confidence-meter {
        height: 10px;
        background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-title">🔬 Microfossil Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered identification of microfossils using Swin Transformer</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Model info
        st.subheader("Model Status")
        if st.button("🔄 Check & Download Model", use_container_width=True):
            with st.spinner("Checking..."):
                model_path = download_model()
                if model_path:
                    size = os.path.getsize(model_path) / 1024 / 1024
                    st.success(f"Model: {size:.1f} MB")
        
        # Settings
        st.subheader("Analysis Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=50,
            max_value=99,
            value=80,
            step=1
        )
        
        show_top_n = st.slider(
            "Show Top N Predictions",
            min_value=3,
            max_value=15,
            value=5,
            step=1
        )
        
        # Class group filter
        st.subheader("Filter by Group")
        groups = sorted(set(info["group"] for info in CLASS_INFO.values()))
        selected_groups = st.multiselect(
            "Show predictions from:",
            groups,
            default=groups
        )
        
        # Sample images
        st.subheader("Quick Test")
        if st.button("🎲 Use Random Example", use_container_width=True):
            # In production, replace with actual sample images
            st.session_state['use_example'] = True
        
        # About section
        with st.expander("📚 About This Tool", expanded=False):
            st.write("""
            **How it works:**
            1. Upload a microfossil image (microscope photo)
            2. AI model processes the image
            3. Get instant classification with confidence scores
            
            **Model Details:**
            - Architecture: Swin-Large Transformer
            - Classes: 32 microfossil types
            - Training: Fine-tuned on microfossil dataset
            - Hosted on: Hugging Face Hub
            
            **Tips for best results:**
            - Use clear, well-lit microscope images
            - Ensure the specimen is centered
            - Avoid blurry or low-resolution images
            """)
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Image Input")
        
        # Upload methods
        upload_method = st.radio(
            "Select input method:",
            ["Upload Image", "Take Photo", "Use Example"],
            horizontal=True
        )
        
        uploaded_file = None
        
        if upload_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
                help="Supported formats: JPG, PNG, TIFF, BMP"
            )
        
        elif upload_method == "Take Photo":
            uploaded_file = st.camera_input("Take a photo")
        
        elif upload_method == "Use Example":
            # Add example images here
            st.info("Example images coming soon. Please upload your own image.")
        
        # Display image if uploaded
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Image preview with options
            col_img1, col_img2 = st.columns([2, 1])
            with col_img1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col_img2:
                st.write("**Image Info:**")
                st.write(f"Size: {image.size}")
                st.write(f"Mode: {image.mode}")
                st.write(f"Format: {image.format if hasattr(image, 'format') else 'Unknown'}")
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Start Classification", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Load model
                    model = load_model()
                    
                    if model:
                        # Preprocess image
                        img_tensor = preprocess_image(image)
                        
                        # Predict
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        
                        # Get top predictions
                        top_probs, top_indices = torch.topk(probabilities, show_top_n)
                        top_probs = top_probs.squeeze().numpy() * 100
                        top_indices = top_indices.squeeze().numpy()
                        
                        # Filter by selected groups
                        filtered_results = []
                        for i in range(len(top_indices)):
                            idx = top_indices[i]
                            if idx < len(CLASS_NAMES):
                                class_name = CLASS_NAMES[idx]
                                group = CLASS_INFO[class_name]["group"]
                                if group in selected_groups:
                                    filtered_results.append({
                                        'class': class_name,
                                        'confidence': top_probs[i],
                                        'group': group,
                                        'description': CLASS_INFO[class_name]["description"]
                                    })
                        
                        # Display results
                        if filtered_results:
                            # Top prediction
                            top_result = filtered_results[0]
                            
                            # Confidence visualization
                            confidence_value = top_result['confidence']
                            confidence_color = (
                                "🟢" if confidence_value > 90 else
                                "🟡" if confidence_value > 70 else
                                "🟠" if confidence_value > 50 else
                                "🔴"
                            )
                            
                            # Display top result in a nice box
                            with st.container():
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h3>🎯 Primary Identification</h3>
                                    <h2>{top_result['class']}</h2>
                                    <p><strong>Group:</strong> {top_result['group']}</p>
                                    <p><strong>Description:</strong> {top_result['description']}</p>
                                    <p><strong>Confidence:</strong> {confidence_color} {confidence_value:.1f}%</p>
                                    <div class="confidence-meter" style="width: {min(confidence_value, 100)}%"></div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show all predictions
                            st.subheader(f"📊 All Predictions (Top {len(filtered_results)})")
                            
                            # Create DataFrame for display
                            results_df = pd.DataFrame(filtered_results)
                            results_df['Confidence %'] = results_df['confidence'].apply(lambda x: f"{x:.1f}%")
                            results_df = results_df[['class', 'Confidence %', 'group', 'description']]
                            results_df.columns = ['Class', 'Confidence', 'Group', 'Description']
                            
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Visualization
                            st.subheader("📈 Confidence Visualization")
                            fig = create_confidence_chart(probabilities, min(10, len(CLASS_NAMES)))
                            st.pyplot(fig)
                            
                            # Export options
                            col_export1, col_export2 = st.columns(2)
                            with col_export1:
                                # Download results as CSV
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results (CSV)",
                                    data=csv,
                                    file_name=f"microfossil_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col_export2:
                                # Download results as JSON
                                json_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'image_info': {
                                        'size': image.size,
                                        'mode': image.mode
                                    },
                                    'predictions': filtered_results,
                                    'model_info': {
                                        'name': 'Swin-Large',
                                        'classes': len(CLASS_NAMES)
                                    }
                                }
                                json_str = json.dumps(json_data, indent=2)
                                st.download_button(
                                    label="📥 Download Results (JSON)",
                                    data=json_str,
                                    file_name=f"microfossil_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        else:
                            st.warning("No predictions match the selected groups.")
                    else:
                        st.error("Model failed to load. Please check the model download.")
        else:
            # Welcome/instruction message
            st.info("👆 Upload an image or take a photo to begin classification")
            
            # Quick stats
            with st.expander("📊 Dataset Statistics", expanded=False):
                # Count by group
                group_counts = {}
                for class_name, info in CLASS_INFO.items():
                    group = info["group"]
                    group_counts[group] = group_counts.get(group, 0) + 1
                
                stats_df = pd.DataFrame({
                    'Group': list(group_counts.keys()),
                    'Count': list(group_counts.values())
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                st.write(f"**Total classes:** {len(CLASS_NAMES)}")
                st.write(f"**Model:** Swin-Large Transformer")
                st.write(f"**Last updated:** Check Hugging Face for updates")
    
    # Footer
    st.markdown("---")
    col_foot1, col_foot2, col_foot3 = st.columns(3)
    
    with col_foot1:
        st.caption("🔄 Model hosted on Hugging Face")
    
    with col_foot2:
        st.caption("⚡ Powered by Swin Transformer")
    
    with col_foot3:
        st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d')}")

# ========== RUN APP ==========
if __name__ == "__main__":
    main()
