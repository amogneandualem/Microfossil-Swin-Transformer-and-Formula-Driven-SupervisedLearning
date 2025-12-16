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

# Updated species list - Put your 32 species here in the EXACT order they were trained
# The order must match your training data folders!
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
        with st.spinner("Downloading model weights from Hugging Face..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # 1. Create SWIN-LARGE model
    # Matches the 1024/2048 channel sizes from your error report
    model = timm.create_model('swin_large_patch4_window7_224', pretrained=False, num_classes=32)
    
    # 2. Load the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # 3. Handle nested state dict ('model_state_dict' vs direct)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 4. Load with strict=False
    # This ignores relative_position_index mismatch which is safe for Swin
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Initialize the model
try:
    model = load_model()
    st.success("✅ Swin-Large Model Loaded Successfully!")
except Exception as e:
    st.error(f"🚨 Setup Error: {e}")
    st.info("Check your internet connection or Hugging Face URL.")
    st.stop()

# --- 2. PREDICTION UI ---
uploaded_file = st.file_uploader("Upload a microfossil image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        with st.spinner("Analyzing..."):
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            confidence, prediction = torch.max(prob, 0)
    
    # Result Display
    res_label = CLASS_NAMES[prediction.item()]
    st.subheader(f"Prediction: {res_label}")
    st.write(f"**Confidence Score:** {confidence.item()*100:.2f}%")
    
    # Sidebar Probability View
    with st.expander("Show detailed probabilities"):
        for i, name in enumerate(CLASS_NAMES):
            if prob[i] > 0.01: # Only show classes > 1%
                st.write(f"{name}: {prob[i].item()*100:.1f}%")
