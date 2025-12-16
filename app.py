import streamlit as st
import torch
import timm
import os
from PIL import Image
from torchvision import transforms

# --- 1. MODEL SOURCE CONFIGURATION ---
# Using your exact Hugging Face direct link
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

st.set_page_config(page_title="Microfossil ID", page_icon="🔬", layout="centered")
st.title("🔬 Microfossil Identification System")

# --- 2. MODEL LOADING LOGIC ---
@st.cache_resource
def load_microfossil_model():
    # Step A: Download the model if it doesn't exist locally
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading weights from Hugging Face... this may take a minute."):
            torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH)
    
    # Step B: Initialize the Swin-Base Architecture
    # Note: These parameters (embed_dim, depths) must match your training exactly
    try:
        model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=32,
            embed_dim=128,           
            depths=(2, 2, 18, 2),    
            num_heads=(4, 8, 16, 32)
        )
        
        # Step C: Load weights onto CPU
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        
        # Extract state_dict if it's a nested dictionary
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Clean keys (removes 'module.' from parallel training)
        new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                          for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # Free up RAM immediately
        del checkpoint
        del state_dict
        
        return model
    except Exception as e:
        st.error(f"Error during model initialization: {e}")
        return None

# Trigger the load
model = load_microfossil_model()

if model:
    st.success("✅ Model weights loaded from URL!")
else:
    st.stop()

# --- 3. PREDICTION INTERFACE ---
uploaded_file = st.file_uploader("Upload a microfossil image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    with st.spinner("Classifying..."):
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probabilities, dim=0)
            
    st.markdown("---")
    st.header(f"Result: **Class {pred.item()}**")
    st.write(f"Confidence: {conf.item():.2%}")
