import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
# Corrected to Swin-Base to match your 1024-dimension weights
MODEL_NAME = "swin_base_patch4_window7_224" 

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
        # Initialize Base architecture to solve the Size Mismatch
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            # Extract state_dict from trainer wrapper
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            return model, None
    except Exception as e:
        return None, str(e)
    return None, "Model file not found."

model, error_msg = load_model()

# Show error only if loading failed
if error_msg and "not found" not in error_msg:
    st.error(f"Error loading model: {error_msg}")

# --- 3. INTERFACE ---
# Radio buttons must exist for the camera option to show
source = st.radio("Choose Input Method:", ("Upload Image File", "Use Camera"))

img_data = None
if source == "Upload Image File":
    img_data = st.file_uploader("Select a microfossil image...", type=["jpg", "jpeg", "png"])
else:
    # This triggers the browser camera request
    img_data = st.camera_input("Capture a microscope sample")

if img_data is not None:
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption='Sample Preview', use_container_width=True)
    
    if st.button('🚀 Classify Specimen'):
        if model is not None:
            with st.spinner('AI analysis in progress...'):
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
            st.error("Cannot classify: Model is not loaded correctly.")
