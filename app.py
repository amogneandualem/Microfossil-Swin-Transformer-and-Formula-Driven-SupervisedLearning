import streamlit as st
import torch
import timm
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

# --- 1. CONFIGURATION ---
# Your specific model URL and path
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 32

st.set_page_config(page_title="Microfossil Swin-ID", page_icon="🔬", layout="wide")

# Custom CSS to improve UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Microfossil Identification System")
st.write("Using Swin Transformer Base Architecture")

# --- 2. MODEL LOADING LOGIC ---
@st.cache_resource
def load_model():
    # Download weights
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading weights from Hugging Face..."):
            torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH)
    
    try:
        # Build Swin-Base with your specific dimensions (128 embed_dim)
        model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=NUM_CLASSES,
            embed_dim=128,           
            depths=(2, 2, 18, 2),    
            num_heads=(4, 8, 16, 32)
        )
        
        # Memory-efficient load
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Clean keys
        new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                          for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # CRITICAL: Delete temporary objects to save RAM
        del checkpoint
        del state_dict
        
        return model
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

model = load_model()

# --- 3. UI AND INFERENCE ---
if model:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Drop a microfossil image here", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Target Fossil", use_container_width=True)
            
            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Inference
            with st.spinner("Analyzing morphology..."):
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    
                    # Get Top 5 results
                    top5_prob, top5_catid = torch.topk(probs, 5)

    with col2:
        st.subheader("📊 Analysis Result")
        if uploaded_file:
            # Main Result
            top_class = top5_catid[0].item()
            top_conf = top5_prob[0].item()
            
            st.metric(label="Predicted Identification", value=f"Class {top_class}")
            st.metric(label="Confidence Score", value=f"{top_conf:.2%}")
            
            # Top 5 Chart
            st.write("Top 5 Likely Classes:")
            chart_data = pd.DataFrame({
                'Class': [f"Class {i.item()}" for i in top5_catid],
                'Probability': [p.item() for p in top5_prob]
            })
            st.bar_chart(chart_data, x='Class', y='Probability')
        else:
            st.info("Please upload an image to see the classification.")

else:
    st.error("Failed to initialize model. Please check logs for RAM limits.")

st.markdown("---")
st.caption("Microfossil Classifier v1.0 | Powered by Swin Transformer")
