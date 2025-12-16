import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# CONFIGURATION
MODEL_NAME = "swin_base_patch4_window7_224"  # Fixes the 1024-dim mismatch
NUM_CLASSES = 32
IMAGE_SIZE = 224

CLASSES = [
    'Acanthodesmia_micropora', 'Actinomma_leptoderma_boreale', 'Antarctissa_denticulata-cyrindrica', 
    'Antarctissa_juvenile', 'Antarctissa_longa-strelkovi', 'Botryocampe_antarctica', 
    'Botryocampe_inflatum-conithorax', 'Ceratocyrtis_historicosus', 'Cycladophora_bicornis', 
    'Cycladophora_cornutoides', 'Cycladophora_davisiana', 'Diatoms', 'Druppatractus_irregularis-bensoni', 
    'Eucyrtidium_spp', 'Fragments', 'Larcids_inner', 'Lithocampe_furcaspiculate', 
    'Lithocampe_platycephala', 'Lithomelissa_setosa-borealis', 'Lophophana_spp', 'Other_Nassellaria', 
    'Other_Spumellaria', 'Phormospyris_stabilis_antarctica', 'Phorticym_clevei-pylonium', 
    'Plectacantha_oikiskos', 'Pseudodictyophimus_gracilipes', 'Sethoconus_tablatus', 
    'Siphocampe_arachnea_group', 'Spongodiscus', 'Spongurus_pylomaticus', 'Sylodictya_spp', 'Zygocircus'
]

st.set_page_config(page_title="Microfossil AI", layout="centered")

@st.cache_resource
def load_model():
    # SEARCH FOR FILE
    target = "best_model.pth"
    model_path = None
    for root, dirs, files in os.walk("."):
        if target in files:
            path = os.path.join(root, target)
            if os.path.getsize(path) > 100 * 1024 * 1024:  # Verify 332MB
                model_path = path
                break
    
    if not model_path:
        return None, "best_model.pth not found. Check Git LFS status."

    try:
        # INITIALIZE BASE MODEL
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # STRIP TRAINING PREFIXES
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"Loaded: {model_path}"
    except Exception as e:
        return None, str(e)

model, status = load_model()

if not model:
    st.error(f"🚨 Setup Error: {status}")
else:
    st.success(f"✅ AI System Online | {status}")
    # ... (Prediction logic here) ...
