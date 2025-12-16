import streamlit as st
import torch
import timm
import os

# CONFIGURATION
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32

@st.cache_resource
def load_model():
    # SEARCH FOR FILE
    target = "best_model.pth"
    model_path = None
    for root, dirs, files in os.walk("."):
        if target in files:
            path = os.path.join(root, target)
            if os.path.getsize(path) > 100 * 1024 * 1024: # Check for 332MB
                model_path = path
                break
    
    if not model_path:
        return None, "best_model.pth not found or Git LFS sync incomplete."

    try:
        # INITIALIZE BASE MODEL
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # CLEAN KEYS
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"Loaded: {model_path}"
    except Exception as e:
        return None, str(e)
