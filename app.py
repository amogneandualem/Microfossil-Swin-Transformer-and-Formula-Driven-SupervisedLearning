import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ==================== CONFIGURATION ====================
# Swin-Base is REQUIRED to match the 1024-dim weights in your error log
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
IMAGE_SIZE = 224

# The exact 32 classes from your dataset
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

# ==================== DEBUGGING SECTION ====================
with st.expander("📂 View Repository File Structure (Debug Mode)"):
    all_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            path = os.path.join(root, file)
            size = os.path.getsize(path) / (1024 * 1024)
            all_files.append(f"{path} ({size:.2f} MB)")
    st.write(all_files)

# ==================== ROBUST MODEL LOADING ====================
@st.cache_resource
def load_model():
    target_file = "best_model.pth"
    model_path = None
    
    # 1. Search every folder for the weights
    for root, dirs, files in os.walk("."):
        if target_file in files:
            temp_path = os.path.join(root, target_file)
            # Ensure it is the real ~332MB weight file, not a 1KB pointer
            if os.path.getsize(temp_path) > 100 * 1024 * 1024:
                model_path = temp_path
                break

    if not model_path:
        return None, "best_model.pth not found or is a small pointer file (<100MB). Check LFS status."

    try:
        # 2. Initialize the LARGER Swin-Base structure to fix size mismatch
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # 3. Load checkpoint onto CPU
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 4. Strip training prefixes (module. or backbone.)
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"Successfully loaded: {model_path}"
    except Exception as e:
        return None, str(e)

model, status = load_model()

# ==================== USER INTERFACE ====================
if not model:
    st.error(f"🚨 Setup Error: {status}")
    st.info("Large files can take 5+ minutes to sync on Streamlit Cloud. Please wait.")
else:
    st.success(f"✅ AI Online | {status}")
    
    uploaded_file = st.file_uploader("Choose a microfossil image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded specimen", use_container_width=True)
        
        if st.button("🚀 Identify Microfossil"):
            # Transformation exactly matches your training logic
            preprocess = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, idx = torch.max(probs, 1)
            
            st.header(f"Result: {CLASSES[idx.item()]}")
            st.progress(conf.item())
            st.write(f"Confidence Level: {conf.item()*100:.2f}%")
