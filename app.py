import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import time

# ==================== CONFIGURATION ====================
# Swin-Base is required for your 1024-dim weights
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
IMAGE_SIZE = 224

# The exact 32 classes from your training dataset
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
    target_file = "best_model.pth"
    model_path = None
    
    # Search every folder for the weights
    for root, dirs, files in os.walk("."):
        if target_file in files:
            path = os.path.join(root, target_file)
            # Verify it is the actual 332MB file and not an LFS stub
            if os.path.getsize(path) > 100 * 1024 * 1024:
                model_path = path
                break

    if not model_path:
        # Check if it exists but is just a tiny pointer file
        for root, dirs, files in os.walk("."):
            if target_file in files:
                size = os.path.getsize(os.path.join(root, target_file)) / 1024
                return None, f"LFS Error: Found file, but it's only {size:.2f}KB. Wait for Git LFS to sync."
        return None, "File 'best_model.pth' not found in any folder. Check GitHub."

    try:
        # Initialize Base structure to solve Size Mismatch
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Strip training prefixes
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"Loaded from: {model_path}"
    except Exception as e:
        return None, str(e)

model, status = load_model()

# ==================== UI LOGIC ====================
if not model:
    st.error(f"🚨 {status}")
    st.info("Tip: Large files take 5-10 minutes to sync on Streamlit. Try refreshing in a few minutes.")
    if st.button("🔄 Reload Model"):
        st.rerun()
else:
    st.success(f"✅ AI Online | {status}")
    
    img_file = st.file_uploader("Upload Microfossil Image", type=["jpg", "png", "jpeg"])
    if img_file:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption="Specimen", use_container_width=True)
        
        if st.button("🚀 Identify"):
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                conf, idx = torch.max(prob, 1)
            
            st.header(f"Result: {CLASSES[idx.item()]}")
            st.write(f"Confidence: {conf.item()*100:.2f}%")
