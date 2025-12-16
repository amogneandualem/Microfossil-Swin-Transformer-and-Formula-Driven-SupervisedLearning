import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import time

# --- CONFIGURATION (Matches your training script) ---
# This MUST be swin_base to handle the 1024/2048 dim weights
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
IMAGE_SIZE = 224
MODEL_PATH = "swin_final_results_advanced/best_model.pth"

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

st.set_page_config(page_title="Microfossil PhD AI", layout="centered")
st.title("🔬 Microfossil Identification System")

@st.cache_resource
def load_model():
    # Wait for Git LFS download if the file is just a 1KB pointer
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000000:
        st.info("⏳ Git LFS is still downloading the large model file. Please wait...")
        time.sleep(10)
        st.rerun()

    try:
        # 1. Create the BASE model structure
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # 2. Load the checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 3. Clean 'module.' prefixes
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 4. Load weights (strict=False handles minor timm version differences)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

if error:
    st.error(f"🚨 Architecture Mismatch or File Error: {error}")
    st.stop()

# --- UI LOGIC ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Specimen Preview", use_container_width=True)
    
    if st.button("🚀 Identify"):
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            idx = torch.argmax(output, 1).item()
            
        st.success(f"### Identification: **{CLASSES[idx]}**")
