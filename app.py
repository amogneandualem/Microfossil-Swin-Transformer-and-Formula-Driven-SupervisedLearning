import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import urllib.request
import os

# --- 1. CONFIGURATION ---
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

# Your error shows 32 classes in the checkpoint. 
# Update this list with your actual 32 species names in the correct order!
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
        with st.spinner("Downloading weights from Hugging Face..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # 1. Create model using TIMM (matches your 'Unexpected keys')
    # Swin Base patch4 window7 224 is the standard for your file size
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=32)
    
    # 2. Load the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # 3. Extract weights if they are wrapped in a dictionary
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 4. Load into model
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Swin-Base (Timm) Model Loaded Successfully!")
except Exception as e:
    st.error(f"🚨 Setup Error: {e}")
    st.stop()

# --- PREDICTION UI ---
uploaded_file = st.file_uploader("Upload microfossil image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Input Image", use_container_width=True)
    
    # Standard Swin Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        with st.spinner("Classifying..."):
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(prob, 0)
    
    label = CLASS_NAMES[pred.item()] if pred.item() < len(CLASS_NAMES) else f"Unknown ({pred.item()})"
    st.subheader(f"Prediction: {label}")
    st.write(f"**Confidence:** {conf.item()*100:.2f}%")
