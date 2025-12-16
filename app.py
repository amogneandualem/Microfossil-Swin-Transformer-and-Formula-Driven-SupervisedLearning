import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import time

# ==================== CONFIGURATION ====================
# MUST match your training script config
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
IMAGE_SIZE = 224
# Path from your GitHub screenshot
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

# ==================== ROBUST MODEL LOADING ====================
@st.cache_resource
def load_model():
    # 1. Wait for Git LFS if the file is too small (pointer file)
    max_retries = 5
    for i in range(max_retries):
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH)
            if file_size > 100000000: # Ensure file is > 100MB
                break
        st.warning(f"⏳ Waiting for model weights to download via Git LFS (Attempt {i+1}/{max_retries})...")
        time.sleep(10)
    
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000000:
        return None, "Model file not found or incomplete. Please wait 5 minutes and refresh."

    try:
        # 2. Initialize Swin-Base (Matches 1024-dim weights)
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 3. Clean keys from training wrappers
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

# ==================== UI LOGIC ====================
if error:
    st.error(f"🚨 {error}")
    st.stop()

st.success("✅ AI Model Loaded Successfully (Swin-Base)")

uploaded_file = st.file_uploader("Upload Microfossil Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Specimen", use_container_width=True)
    
    if st.button("🚀 Identify"):
        # Preprocessing matches your training script exactly
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, idx = torch.max(prob, 1)
            
        st.header(f"Result: {CLASSES[idx.item()]}")
        st.write(f"Confidence: {conf.item()*100:.2f}%")
