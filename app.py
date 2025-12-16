import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import urllib.request
import os

# --- 1. CONFIGURATION ---
# The link to your model on Hugging Face
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["Diatoms", "Druppatractus_irregularis-bensoni", "Eucyrtidium_spp", "Fragments", "Others"]

st.set_page_config(page_title="Microfossil ID", page_icon="🔬")
st.title("🔬 Microfossil Identification System")

# --- 2. DOWNLOAD & LOAD MODEL ---
@st.cache_resource
def load_model():
    # Download weights if they aren't already on the Streamlit server
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Hugging Face (332MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # Initialize Swin-Base (Matches your 1024-dim weight file)
    model = models.swin_b(weights=None)
    num_features = model.head.in_features  # Should be 1024 for Swin-B
    model.head = nn.Linear(num_features, len(CLASS_NAMES))
    
    # Load the checkpoint file
    # We use map_location='cpu' because Streamlit Cloud runs on CPUs
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # --- FIX FOR "UNEXPECTED KEY" ERROR ---
    # Check if weights are inside a 'model_state_dict' wrapper
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Load the cleaned state_dict into the model
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Initialize the model
try:
    model = load_model()
    st.success("✅ Model Ready: Swin-Base Architecture Loaded")
except Exception as e:
    st.error(f"🚨 Setup Error: {e}")
    st.info("Try rebooting the app in the Streamlit Cloud dashboard.")
    st.stop()

# --- 3. PREDICTION UI ---
uploaded_file = st.file_uploader("Upload a microfossil image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Standard Swin Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        with st.spinner("Analyzing..."):
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            confidence, prediction = torch.max(prob, 0)
    
    # Display Result
    st.subheader(f"Prediction: {CLASS_NAMES[prediction]}")
    st.write(f"**Confidence Score:** {confidence.item()*100:.2f}%")
    
    # Detailed Probability Bars
    with st.expander("See probability breakdown"):
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {prob[i].item()*100:.1f}%")
            st.progress(float(prob[i]))
