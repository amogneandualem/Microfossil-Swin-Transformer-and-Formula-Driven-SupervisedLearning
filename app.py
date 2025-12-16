import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION ---
# This path must match your GitHub folder structure exactly
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
MODEL_NAME = "swin_base_patch4_window7_224" # Matches Swin-Base (1024-dim)
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

st.set_page_config(page_title="Microfossil PhD AI", layout="centered")
st.title("🔬 Microfossil Identification System")

# --- 2. DEBUGGING / FILE SEARCHER ---
# This helps find the file if the path is slightly wrong on the server
if not os.path.exists(MODEL_PATH):
    st.warning(f"Searching for model file... (Not found at {MODEL_PATH})")
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "best_model.pth":
                MODEL_PATH = os.path.join(root, file)
                st.success(f"✅ Found model at: {MODEL_PATH}")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model(path):
    try:
        # Initialize Swin-Base architecture
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu")
            
            # Extract state_dict (handles your specific trainer format)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Clean "module." prefix from multi-GPU training
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load weights (strict=False ignores non-essential metadata)
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            return model, None
        return None, "File still not found after search."
    except Exception as e:
        return None, str(e)

model, error = load_model(MODEL_PATH)

# --- 4. INTERFACE ---
if error:
    st.error(f"Model Error: {error}")
    st.info("Ensure you pushed the model via Git LFS and the path is correct.")

source = st.radio("Choose Input Method:", ("Upload Image File", "Use Camera"))

img_data = None
if source == "Upload Image File":
    img_data = st.file_uploader("Select microfossil image...", type=["jpg", "jpeg", "png"])
else:
    img_data = st.camera_input("Capture microscope specimen")

if img_data is not None:
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption='Specimen Preview', use_container_width=True)
    
    if st.button('🚀 Classify Specimen'):
        if model is not None:
            with st.spinner('AI analyzing...'):
                # Exact transforms from your training script
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, 1)
                
                label = CLASSES[idx.item()]
                st.success(f"### Identification: **{label}**")
                st.info(f"**Confidence Score:** {conf.item()*100:.2f}%")
        else:
            st.error("Model is not loaded. Classification unavailable.")
