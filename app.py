import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# CONFIGURATION
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
CLASS_NAMES = ["Diatoms", "Druppatractus_irregularis-bensoni", "Eucyrtidium_spp", "Fragments", "Others"]

st.title("🔬 Microfossil Identification")

@st.cache_resource
def load_model():
    # Use swin_b to match your 1024-dimension weights
    model = models.swin_b(weights=None) 
    model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model
    return None

model = load_model()

if model:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, width=300)
        
        # Transform & Predict
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        batch = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(batch)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(prob, 0)
        
        st.success(f"Result: {CLASS_NAMES[pred]} ({conf*100:.1f}%)")
else:
    st.error("Model weights file not found on server.")
