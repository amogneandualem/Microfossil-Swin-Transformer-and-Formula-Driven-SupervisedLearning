import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION ---
# Path confirmed from your GitHub screenshot
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
# Base architecture required to match 1024/2048 dimensions
MODEL_NAME = "swin_base_patch4_window7_224" 

# Full 32-class list from your training configuration
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

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        # Create the Base model (matches your 332MB weight file)
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            # Extract weights from the trainer's dictionary
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Clean 'module.' prefix if it exists from multi-GPU training
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load with strict=False to handle naming variations in timm layers
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model, None
        else:
            return None, f"File not found at {MODEL_PATH}"
    except Exception as e:
        return None, str(e)

model, error_msg = load_model()

# --- 3. USER INTERFACE ---
if error_msg:
    st.error(f"Model Error: {error_msg}")

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
                # Standard ImageNet transforms used in training
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
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
            st.error("Model not loaded. Please check the error message above.")
