import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ==================== CONFIGURATION ====================
# REQUIRED: Swin-Base matches the 1024-dim layers in your error log
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

st.set_page_config(page_title="Microfossil Identification", layout="centered")
st.title("🔬 Microfossil Identification System")

# ==================== ROBUST MODEL LOADING ====================
@st.cache_resource
def load_model():
    target_file = "best_model.pth"
    model_path = None
    
    # 1. Search every subfolder for the file
    for root, dirs, files in os.walk("."):
        if target_file in files:
            temp_path = os.path.join(root, target_file)
            # 2. Verify file size: Must be >100MB to be real weights, not a pointer
            if os.path.getsize(temp_path) > 100 * 1024 * 1024:
                model_path = temp_path
                break

    if not model_path:
        # Check if the file exists but is just a tiny LFS pointer
        for root, dirs, files in os.walk("."):
            if target_file in files:
                size_kb = os.path.getsize(os.path.join(root, target_file)) / 1024
                return None, f"LFS Sync Error: Found {target_file} but it is only {size_kb:.2f}KB. Ensure you pushed the actual 332MB file from your PC terminal."
        return None, "best_model.pth not found in any folder. Check your GitHub repository."

    try:
        # 3. Initialize Swin-Base to fix the size mismatch error
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # 4. Load weights onto CPU
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 5. Clean training prefixes like 'module.' or 'backbone.'
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"Successfully loaded from: {model_path}"
    except Exception as e:
        return None, f"Error initializing model: {str(e)}"

model, status = load_model()

# ==================== USER INTERFACE ====================
if not model:
    st.error(f"🚨 Setup Error: {status}")
    st.info("Note: Large models (332MB) take 5-10 minutes to sync on Streamlit Cloud.")
    if st.button("🔄 Reload App"):
        st.rerun()
else:
    st.success(f"✅ AI Online | {status}")
    
    img_file = st.file_uploader("Upload Microfossil Image", type=["jpg", "png", "jpeg"])
    if img_file:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption="Uploaded Specimen", use_container_width=True)
        
        if st.button("🚀 Run Identification"):
            # Normalization parameters must match your training script
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
            st.write(f"**Confidence Level:** {conf.item()*100:.2f}%")
