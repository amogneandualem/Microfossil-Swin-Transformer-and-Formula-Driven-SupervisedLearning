import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import urllib.request
import os

# --- 1. CONFIGURATION (Strictly matching your training script) ---
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"
MODEL_NAME = "swin_base_patch4_window7_224"
NUM_CLASSES = 32

# Mapping for your 32 classes - ensure these match your folder order!
CLASS_NAMES = [
    "Diatoms", "Druppatractus_irregularis", "Eucyrtidium_spp", "Fragments", "Others",
    "Class 5", "Class 6", "Class 7", "Class 8", "Class 9", "Class 10",
    "Class 11", "Class 12", "Class 13", "Class 14", "Class 15", "Class 16",
    "Class 17", "Class 18", "Class 19", "Class 20", "Class 21", "Class 22",
    "Class 23", "Class 24", "Class 25", "Class 26", "Class 27", "Class 28",
    "Class 29", "Class 30", "Class 31"
]

st.set_page_config(page_title="Microfossil ID", page_icon="🔬", layout="centered")
st.title("🔬 Microfossil Identification System")
st.markdown("---")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model weights..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # We explicitly define dimensions to stop the "Size Mismatch" errors.
    # embed_dim=128 ensures the stages scale to 256, 512, 1024, and 2048.
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
        embed_dim=128,           # Matching your checkpoint's base width
        depths=(2, 2, 18, 2),     # Standard Swin-Base depth
        num_heads=(4, 8, 16, 32)  # Standard Swin-Base heads
    )
    
    # Load checkpoint and handle the 'model_state_dict' wrapper
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Clean the keys (removes 'module.' from DataParallel training)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('backbone.', '')
        new_state_dict[name] = v

    # strict=False is critical to ignore non-matching position buffers
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# Initialize Model
try:
    model = load_model()
    st.success(f"✅ Model {MODEL_NAME} ready!")
except Exception as e:
    st.error(f"🚨 Setup Error: {e}")
    st.stop()

# --- 2. USER INTERFACE ---
st.write("Upload a microscope image of a microfossil to identify its class.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Sample", use_container_width=True)
    
    # Preprocessing (Matching your training 'eval_transform')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        with st.spinner("Analyzing morphology..."):
            outputs = model(input_tensor)
            prob = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, prediction = torch.max(prob, 0)
    
    # Results
    st.subheader("Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Identification", CLASS_NAMES[prediction.item()])
    
    with col2:
        st.metric("Confidence", f"{confidence.item()*100:.2f}%")

    # Detailed Probability Breakdown
    with st.expander("See full classification probabilities"):
        # Sort probabilities to show top guesses first
        probs_dict = {CLASS_NAMES[i]: p.item() for i, p in enumerate(prob)}
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        
        for name, p in sorted_probs:
            if p > 0.001: # Only show matches above 0.1%
                st.write(f"**{name}**: {p*100:.2f}%")
                st.progress(p)

st.markdown("---")
st.caption("Microfossil Identification System | Powered by Swin Transformer")
