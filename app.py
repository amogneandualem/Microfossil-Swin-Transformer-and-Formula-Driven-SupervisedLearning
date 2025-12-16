import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ==================== CONFIGURATION ====================
MODEL_NAME = "swin_base_patch4_window7_224" 
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

st.set_page_config(page_title="Microfossil PhD AI", layout="wide")
st.title("🔬 Microfossil Identification System")

# ==================== DEBUGGING SECTION ====================
with st.expander("📂 View Repository File Structure (Debug)"):
    all_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            rel_path = os.path.join(root, file)
            size_mb = os.path.getsize(rel_path) / (1024 * 1024)
            all_files.append(f"{rel_path} ({size_mb:.2f} MB)")
    st.write(all_files)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    target_file = "best_model.pth"
    model_path = None
    
    # Deep search for the file
    for root, dirs, files in os.walk("."):
        if target_file.lower() in [f.lower() for f in files]:
            # Handle case sensitivity
            actual_name = [f for f in files if f.lower() == target_file.lower()][0]
            temp_path = os.path.join(root, actual_name)
            
            # Check size: Must be > 100MB to be the real weight file
            if os.path.getsize(temp_path) > 100 * 1024 * 1024:
                model_path = temp_path
                break

    if not model_path:
        return None, f"CRITICAL: '{target_file}' not found or is too small (<100MB)."

    try:
        # Initialize Swin-Base to match 1024-dim weights
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Strip prefixes
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"✅ Successfully loaded: {model_path}"
    except Exception as e:
        return None, str(e)

model, status = load_model()

# ==================== USER INTERFACE ====================
if not model:
    st.error(status)
    st.info("Check the folder list above. If your file is not there, it was not pushed to GitHub.")
else:
    st.success(status)
    # ... rest of your prediction UI code ...
