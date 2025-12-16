import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

# IMPORTANT: This path points inside your folder where the LFS file is
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
MODEL_NAME = "swin_base_patch4_window7_224"

@st.cache_resource
def load_model():
    # Initialize architecture
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=32)
    # Load the 349MB LFS file
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint
    
    # Clean keys
    cleaned_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                    for k, v in state_dict.items() if not k.startswith('head.')}
    
    model.load_state_dict(cleaned_dict, strict=False)
    model.eval()
    return model

st.title("🔬 Microfossil Classification")
st.write("Upload an image to identify the microfossil species.")

try:
    model = load_model()
    st.success("Model loaded successfully from sub-folder!")
except Exception as e:
    st.error(f"Could not find model at {MODEL_PATH}. Error: {e}")

# ... (Add your image uploader and prediction code below)
