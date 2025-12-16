import streamlit as st
import torch
import timm
import os
import urllib.request
from torchvision import transforms
from PIL import Image

# --- 1. CONFIGURATION ---
# The URL you provided for your model weights
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"
MODEL_NAME = "swin_base_patch4_window7_224"

# Exact list of 32 classes from your training
CLASSES = [
    'Acanthodesmia_micropora', 'Actinomma_leptoderma_boreale',
    'Antarctissa_denticulata-cyrindrica', 'Antarctissa_juvenile',
    'Antarctissa_longa-strelkovi', 'Botryocampe_antarctica',
    'Botryocampe_inflatum-conithorax', 'Ceratocyrtis_historicosus',
    'Cycladophora_bicornis', 'Cycladophora_cornutoides',
    'Cycladophora_davisiana', 'Diatoms',
    'Druppatractus_irregularis-bensoni', 'Eucyrtidium_spp',
    'Fragments', 'Larcids_inner', 'Lithocampe_furcaspiculate',
    'Lithocampe_platycephala', 'Lithomelissa_setosa-borealis',
    'Lophophana_spp', 'Other_Nassellaria', 'Other_Spumellaria',
    'Phormospyris_stabilis_antarctica', 'Phorticym_clevei-pylonium',
    'Plectacantha_oikiskos', 'Pseudodictyophimus_gracilipes',
    'Sethoconus_tablatus', 'Siphocampe_arachnea_group',
    'Spongodiscus', 'Spongurus_pylomaticus',
    'Sylodictya_spp', 'Zygocircus'
]

st.set_page_config(page_title="Microfossil AI", layout="centered")
st.title("🔬 Microfossil Classification")

# --- 2. LOADING LOGIC ---
@st.cache_resource
def load_model():
    # If model is not found in the cloud workspace, download it from Hugging Face
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading 349MB model from Hugging Face..."):
            # This bypasses your local network upload issues
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("Download complete!")
    
    # Initialize the Swin Transformer
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    
    # Clean the state_dict (handles 'module.' or 'backbone.' prefixes)
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint
    cleaned_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                    for k, v in state
