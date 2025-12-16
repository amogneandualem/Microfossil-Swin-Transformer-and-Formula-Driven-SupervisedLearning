import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import urllib.request
import os

# --- 1. CONFIGURATION ---
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["Diatoms", "Druppatractus_irregularis-bensoni", "Eucyrtidium_spp", "Fragments", "Others"]

st.set_page_config(page_title="Microfossil ID", page_icon="🔬")
st.title("🔬 Microfossil Identification System")

# --- 2. DOWNLOAD & LOAD MODEL ---
@st.cache_resource
def load_model():
    # Download weights from Hugging Face if not already present
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading 332MB model weights from Hugging Face..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # Initialize Swin-Base (Matches your 1024-dim weights)
    model = models.swin_b(weights=None)
    model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))
    
    # Load weights onto CPU
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Swin-Base Model Loaded from Hugging Face!")
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# --- 3. PREDICTION UI ---
uploaded_file = st.file_uploader("Upload microfossil image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Swin Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred = torch.max(prob, 0)
    
    st.subheader(f"Prediction: {CLASS_NAMES[pred]}")
    st.write(f"Confidence: {conf.item()*100:.2f}%")
