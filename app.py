import streamlit as st
import torch
import timm
import os
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# --- 1. CONFIGURATION ---
# Fixed the string literal error and Repo ID format
REPO_ID = "amogneandualem/microfossil-classifier"
FILENAME = "best_model.pth"

st.set_page_config(page_title="Microfossil ID", page_icon="🔬", layout="centered")
st.title("🔬 Microfossil Identification System")
st.write("Upload a microfossil image to classify it using the Swin Transformer model.")

# --- 2. MODEL LOADING (Optimized for 1GB RAM) ---
@st.cache_resource
def load_model():
    try:
        # Download weights from Hugging Face
        with st.spinner("Downloading model weights from Hugging Face..."):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        
        # Initialize Swin-Base Architecture
        # Using the settings that match your specific channel/dimension error
        model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=32,
            embed_dim=128,           
            depths=(2, 2, 18, 2),    
            num_heads=(4, 8, 16, 32)
        )
        
        # Load weights with memory efficiency
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Extract state_dict if it's nested
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Clean potential prefixes from training (e.g., 'module.' from DataParallel)
        new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                          for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # Explicitly delete checkpoint to free RAM immediately
        del checkpoint
        del state_dict
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize the model
model = load_model()

if model:
    st.success("✅ Model loaded successfully!")
else:
    st.warning("⚠️ Model failed to load. Please check the logs.")
    st.stop()

# --- 3. PREDICTION INTERFACE ---
uploaded_file = st.file_uploader("Choose a microfossil image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Run Inference
    with st.spinner("Classifying..."):
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probabilities, dim=0)
            
    # Display Result
    st.markdown("---")
    st.subheader(f"Prediction: **Class {pred.item()}**")
    st.info(f"Confidence Score: {conf.item():.2%}")
