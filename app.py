import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# CONFIGURATION
MODEL_NAME = "swin_base_patch4_window7_224" # FIXES: Size mismatch (1024-dim)
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

@st.cache_resource
def load_model():
    # Automatically find best_model.pth in any subfolder
    target = "best_model.pth"
    model_path = None
    for root, dirs, files in os.walk("."):
        if target in files:
            path = os.path.join(root, target)
            if os.path.getsize(path) > 100 * 1024 * 1024: # Check if > 100MB
                model_path = path
                break
    
    if not model_path:
        return None, "best_model.pth not found. Is it pushed to GitHub via LFS?"

    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Strip prefixes like 'module.' or 'backbone.'
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"✅ Model Loaded from: {model_path}"
    except Exception as e:
        return None, str(e)

model, status = load_model()

if not model:
    st.error(f"🚨 Setup Error: {status}")
    st.info("Check your GitHub repo to see if 'best_model.pth' is actually 332MB.")
else:
    st.success(status)
    # --- Prediction UI ---
    img_file = st.file_uploader("Upload Fossil Image", type=["jpg", "png", "jpeg"])
    if img_file:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption="Specimen", width=400)
        
        if st.button("🚀 Identify"):
            # Matches your training normalization
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                idx = torch.argmax(output, 1).item()
            
            st.header(f"Result: {CLASSES[idx]}")
