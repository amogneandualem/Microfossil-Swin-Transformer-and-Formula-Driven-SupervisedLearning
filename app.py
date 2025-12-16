import streamlit as st
import torch
import timm
import os
from torchvision import transforms
from PIL import Image

# ------------------ 1. CONFIGURATION ------------------
MODEL_NAME = "swin_base_patch4_window7_224"
# The app will check the subfolder AND the root folder to be safe
POSSIBLE_PATHS = [
    "swin_final_results_advanced/best_model.pth",
    "best_model.pth"
]

CLASSES = [
    'Acanthodesmia_micropora', 'Actinomma_leptoderma_boreale',
    'Antarctissa_denticulata-cyrindrica', 'Antarctissa_juvenile',
    'Antarctissa_longa-strelkovi', 'Botryocampe_antarctica',
    'Botryocampe_inflatum-conithorax', 'Ceratocyrtis_historicosus',
    'Cycladophora_bicornis', 'Cycladophora_cornutoides',
    'Cycladophora_davisiana', 'Diatoms',
    'Druppatractus_irregularis-bensoni', 'Eucyrtidium_spp',
    'Fragments', 'Larcids_inner', 'Lithocampe_furcaspiculate',
    'Lithocampe_platycephala', 'Lithomelissa_setosa-borealis',
    'Lophophana_spp', 'Other_Nassellaria', 'Other_Spumellaria',
    'Phormospyris_stabilis_antarctica', 'Phorticym_clevei-pylonium',
    'Plectacantha_oikiskos', 'Pseudodictyophimus_gracilipes',
    'Sethoconus_tablatus', 'Siphocampe_arachnea_group',
    'Spongodiscus', 'Spongurus_pylomaticus',
    'Sylodictya_spp', 'Zygocircus'
]

st.set_page_config(page_title="Microfossil AI", layout="centered")
st.title("🔬 Microfossil Species Identification")

# ------------------ 2. LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    
    final_path = None
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            final_path = path
            break
            
    if final_path is None:
        st.error("⚠️ Model file (.pth) not found. Check if swin_final_results_advanced is ignored in .gitignore!")
        return None

    checkpoint = torch.load(final_path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint
    cleaned_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                    for k, v in state_dict.items() if not k.startswith('head.')}
    
    model.load_state_dict(cleaned_dict, strict=False)
    model.eval()
    return model

model = load_model()

# ------------------ 3. INTERFACE ------------------
source = st.radio("Choose Input Method:", ("Upload File", "Take Photo"))

img_file = None
if source == "Upload File":
    img_file = st.file_uploader("Select microfossil image...", type=["jpg", "jpeg", "png"])
else:
    img_file = st.camera_input("Capture specimen")

# Button only appears if image is provided
if img_file is not None:
    image = Image.open(img_file).convert('RGB')
    
    # 🚀 THE CLASSIFY BUTTON
    if st.button('Classify Specimen'):
        if model is not None:
            with st.spinner('AI is analyzing...'):
                # Prep
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    out = model(img_tensor)
                    probs = torch.nn.functional.softmax(out, dim=1)
                    conf, idx = torch.max(probs, 1)
                
                label = CLASSES[idx.item()]
                st.success(f"### Prediction: **{label}**")
                st.info(f"**Confidence:** {conf.item()*100:.2f}%")
        else:
            st.error("Model not loaded.")
