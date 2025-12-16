import streamlit as st
import torch
import timm
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 32

st.set_page_config(page_title="Microfossil Swin-ID", page_icon="🔬", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .fast-text {
        color: #28a745;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Microfossil Identification System")
st.markdown("**Optimized for Speed - Uses Caching**")

# --- 2. OPTIMIZED MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    """Load model once and cache it"""
    start_time = time.time()
    
    # Download if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading weights..."):
            import requests
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
    
    try:
        # OPTIMIZATION: Use exact architecture to avoid rebuilding
        model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=NUM_CLASSES,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32]
        )
        
        # OPTIMIZATION: Use weights_only=True for safety
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Clean keys efficiently
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('backbone.', '')
            new_state_dict[new_key] = v
        
        # Load with strict=False for any minor mismatches
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # Force garbage collection
        del checkpoint
        del state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        load_time = time.time() - start_time
        st.sidebar.success(f"✓ Model loaded in {load_time:.1f}s")
        
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)[:100]}")
        return None

# --- 3. CACHED PREDICTION FUNCTION ---
@st.cache_data(ttl=3600)  # Cache predictions for 1 hour
def predict_image(_model, image_tensor):
    """Cached prediction function"""
    with torch.no_grad():
        outputs = _model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)
        return top5_prob, top5_catid

# --- 4. PREPROCESSING (CACHED) ---
@st.cache_data
def preprocess_image(image):
    """Cache preprocessed images"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 5. UI LAYOUT ---
# Load model once
model = load_model()

if model:
    # Sidebar with performance info
    with st.sidebar:
        st.subheader("⚡ Performance")
        st.write(f"Model: Swin-Base")
        st.write(f"Parameters: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        st.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        use_cache = st.checkbox("Enable Prediction Cache", value=True, 
                               help="Caches predictions for faster repeated use")
        show_details = st.checkbox("Show Processing Details", value=False)
    
    # Main columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Drop microfossil image here", 
            type=["jpg", "png", "jpeg", "tiff", "bmp"],
            help="Supported formats: JPG, PNG, TIFF, BMP"
        )
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Target Fossil", use_container_width=True)
            
            # Preprocess with timing
            if show_details:
                with st.spinner("Preprocessing..."):
                    start_preprocess = time.time()
                    input_tensor = preprocess_image(image)
                    preprocess_time = time.time() - start_preprocess
            else:
                input_tensor = preprocess_image(image)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if uploaded_file:
            # Inference with timing
            if show_details:
                with st.spinner("Running inference..."):
                    start_inference = time.time()
                    if use_cache:
                        top5_prob, top5_catid = predict_image(model, input_tensor)
                    else:
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs[0], dim=0)
                            top5_prob, top5_catid = torch.topk(probs, 5)
                    inference_time = time.time() - start_inference
            else:
                if use_cache:
                    top5_prob, top5_catid = predict_image(model, input_tensor)
                else:
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        top5_prob, top5_catid = torch.topk(probs, 5)
            
            # Display results
            top_class = top5_catid[0].item()
            top_conf = top5_prob[0].item() * 100
            
            # Result cards
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Predicted Class", 
                    value=f"#{top_class}",
                    help=f"Class index {top_class}"
                )
            with col_b:
                st.metric(
                    label="Confidence", 
                    value=f"{top_conf:.1f}%",
                    delta="high" if top_conf > 80 else "medium" if top_conf > 60 else "low"
                )
            
            # Top 5 Chart
            st.markdown("**Top 5 Predictions:**")
            
            # Create dataframe for display
            chart_data = pd.DataFrame({
                'Class': [f"Class {i.item()}" for i in top5_catid],
                'Confidence': [p.item() * 100 for p in top5_prob],
                'Probability': [p.item() for p in top5_prob]
            })
            
            # Display as bar chart
            st.bar_chart(chart_data.set_index('Class')['Confidence'])
            
            # Show table with details
            with st.expander("📋 Detailed Results", expanded=False):
                chart_data['Confidence'] = chart_data['Confidence'].apply(lambda x: f"{x:.2f}%")
                st.table(chart_data[['Class', 'Confidence']])
            
            # Show timings if enabled
            if show_details:
                st.markdown("---")
                st.markdown("**⏱️ Performance Timings:**")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.write(f"Preprocessing: {preprocess_time:.3f}s")
                with col_t2:
                    st.write(f"Inference: {inference_time:.3f}s")
                st.write(f"Total: {(preprocess_time + inference_time):.3f}s")
            
            # Export options
            st.markdown("---")
            st.markdown("**💾 Export Results:**")
            csv = chart_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"microfossil_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            # Welcome/instruction panel
            st.info("👆 **Upload an image to begin analysis**")
            
            # Tips for better performance
            with st.expander("💡 Tips for faster processing", expanded=True):
                st.write("""
                1. **Enable caching** in sidebar (default: ON)
                2. **Use JPG/PNG** (faster than TIFF)
                3. **Optimal size**: 500-1500px width
                4. **Clear cache** if results seem stale
                5. **First run** loads model (takes 10-20s)
                """)
            
            # Stats
            st.metric("Model Status", "✅ Ready")
            st.metric("Caching", "✅ Enabled" if use_cache else "❌ Disabled")
            
else:
    st.error("❌ Failed to load model. Check if model file exists and architecture matches.")

# Footer
st.markdown("---")
st.caption("⚡ **Optimized Microfossil Classifier v2.0** | Powered by Swin Transformer | Caching enabled for faster performance")
