import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- 1. CONFIGURATION ---
# We use swin_b because your 332MB weights are for the "Base" model (1024-dim)
MODEL_NAME = "swin_base_patch4_window7_224" 
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
CLASS_NAMES = [
    "Diatoms", 
    "Druppatractus_irregularis-bensoni", 
    "Eucyrtidium_spp", 
    "Fragments", 
    "Others"
]

st.set_page_config(page_title="Microfossil Classifier", page_icon="🔬")
st.title("🔬 Microfossil Identification System")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Initialize the BASE architecture (Must be swin_b)
    model = models.swin_b(weights=None)
    
    # Change the head to match your 5 classes
    # Swin-B has 1024 input features in the head
    num_features = model.head.in_features 
    model.head = nn.Linear(num_features, len(CLASS_NAMES))
    
    if os.path.exists(MODEL_PATH):
        try:
            # map_location='cpu' is required for Streamlit Cloud servers
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            return model, None
        except Exception as e:
            return None, f"Error loading weights: {e}"
    else:
        return None, f"Weight file not found at: {MODEL_PATH}"

model, error = load_model()

# --- 3. USER INTERFACE ---
if error:
    st.error(f"🚨 Configuration Error: {error}")
    st.info("Check if the model file is correctly uploaded to GitHub LFS.")
else:
    st.success("✅ Swin-Base Model Ready")

    uploaded_file = st.file_uploader("Upload a microscope image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Target Image', use_container_width=True)
        
        # Swin Transformer Standard Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        # Display Results
        st.subheader(f"Prediction: {CLASS_NAMES[predicted.item()]}")
        st.write(f"**Confidence Score:** {confidence.item()*100:.2f}%")
        
        # Show breakdown of all classes
        with st.expander("Show detailed probabilities"):
            for i, name in enumerate(CLASS_NAMES):
                st.write(f"{name}: {probabilities[i].item()*100:.1f}%")
                st.progress(float(probabilities[i]))
