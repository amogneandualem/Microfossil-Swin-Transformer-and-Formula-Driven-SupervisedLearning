import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import urllib.request
import os

# --- 1. CONFIGURATION (Matching your Training Script) ---
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"
MODEL_NAME = "swin_base_patch4_window7_224" # Matched to your Config
NUM_CLASSES = 32

# Mapping for your 32 classes
CLASS_NAMES = [
    "Diatoms", "Druppatractus_irregularis", "Eucyrtidium_spp", "Fragments", "Others",
    "Class 5", "Class 6", "Class 7", "Class 8", "Class 9", "Class 10",
    "Class 11", "Class 12", "Class 13", "Class 14", "Class 15", "Class 16",
    "Class 17", "Class 18", "Class 19", "Class 20", "Class 21", "Class 22",
    "Class 23", "Class 24", "Class 25", "Class 26", "Class 27", "Class 28",
    "Class 29", "Class 30", "Class 31"
]

st.set_page_config(page_title="Microfossil ID", page_icon="🔬")
st.title("🔬 Microfossil Identification System")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # Create the exact model structure from your training code
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    
    # Load the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # Extract the state dict using your trainer's specific key
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Clean the keys (handles the 'module.' prefix if you trained on Multi-GPU/DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('backbone.', '')
        new_state_dict[name] = v

    # Load weights with strict=False to ignore non-essential buffer mismatches
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

try:
    model = load_model()
    st.success(f"✅ {MODEL_NAME} Loaded Successfully!")
except Exception as e:
    st.error(f"🚨 Setup Error: {e}")
    st.stop()

# --- 2. PREDICTION UI ---
uploaded_file = st.file_uploader("Upload a microfossil image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Exact transforms from your EfficientMicrofossilDataset eval_transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        with st.spinner("Analyzing..."):
            outputs = model(input_tensor)
            prob = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, prediction = torch.max(prob, 0)
    
    label = CLASS_NAMES[prediction.item()]
    st.subheader(f"Result: {label}")
    st.progress(float(confidence.item()))
    st.write(f"**Confidence Score:** {confidence.item()*100:.2f}%")
