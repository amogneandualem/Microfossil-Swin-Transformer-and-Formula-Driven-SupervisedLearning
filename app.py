import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import time

# ==================== CONFIGURATION ====================
# This MUST be swin_base to match your 1024-dim checkpoint
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
IMAGE_SIZE = 224

# Exact class list from your dataset
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

# ==================== ROBUST MODEL LOADING ====================
@st.cache_resource
def load_model():
    # 1. Search for best_model.pth in all subfolders
    target_file = "best_model.pth"
    model_path = None
    for root, dirs, files in os.walk("."):
        if target_file in files:
            temp_path = os.path.join(root, target_file)
            # Ensure it's the real 332MB file, not a 1KB LFS pointer
            if os.path.getsize(temp_path) > 100 * 1024 * 1024:
                model_path = temp_path
                break

    if not model_path:
        return None, "best_model.pth not found (or still downloading via Git LFS)."

    try:
        # 2. Initialize Swin-Base Architecture
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # 3. Load Checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 4. Clean keys from training (remove module. or backbone.)
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        # 5. Apply weights with strict=False to bypass minor naming variations
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"Loaded: {model_path}"
    except Exception as e:
        return None, str(e)

model, status = load_model()

# ==================== USER INTERFACE ====================
if not model:
    st.error(f"🚨 Model Error: {status}")
    st.info("Wait 5 minutes for Git LFS to finish downloading the large file.")
    if st.button("🔄 Check for file again"):
        st.rerun()
else:
    st.success(f"✅ AI System Online | {status}")

    col1, col2 = st.columns(2)
    
    with col1:
        source = st.radio("Input Source:", ("Upload File", "Camera"))
        img_file = st.file_uploader("Select Image", type=["jpg", "png"]) if source == "Upload File" else st.camera_input("Capture")

    with col2:
        if img_file:
            image = Image.open(img_file).convert('RGB')
            st.image(image, caption="Current Specimen", use_container_width=True)
            
            if st.button("🚀 Analyze Microfossil"):
                # Exact preprocessing from your training script
                preprocess = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = preprocess(image).unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    confidence, index = torch.max(probs, 1)
                
                st.markdown(f"## Result: **{CLASSES[index.item()]}**")
                st.progress(confidence.item())
                st.write(f"Confidence Level: {confidence.item()*100:.2f}%")
