import streamlit as st
import torch
import timm
import os

# 1. FORCE THE BASE ARCHITECTURE
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
MODEL_PATH = "swin_final_results_advanced/best_model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, "File not found."
    
    # Check if Git LFS actually downloaded the 332MB
    if os.path.getsize(MODEL_PATH) < 1000000: 
        return None, "LFS Error: File is only a 1KB pointer."

    try:
        # Initialize the LARGER base model
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Clean training prefixes
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # strict=False is KEY to bypassing small naming mismatches
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()
if error:
    st.error(f"🚨 Architecture Mismatch Fix Needed: {error}")
