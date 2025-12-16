import streamlit as st
import torch
import timm
import os
from torchvision import transforms
from PIL import Image

# --- 1. CONFIGURATION ---
# Based on your GitHub folder structure
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
MODEL_NAME = "swin_base_patch4_window7_224"

# Full list of 32 microfossil species
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

st.set_page_config(page_title="Microfossil AI Classifier", layout="centered")
st.title("🔬 Microfossil Species Identification")
st.write("Swin Transformer Base Model (Trained for PhD Research)")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Initialize the Swin Transformer architecture
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    
    # Check if the file exists in the folder
    if not os.path.exists(MODEL_PATH):
        st.error(f"File not found at {MODEL_PATH}. Please check your GitHub folder structure.")
        return None

    # Load the weights (pulled via Git LFS)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    
    # Extract state_dict and clean prefixes
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint
    cleaned_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                    for k, v in state_dict.items() if not k.startswith('head.')}
    
    model.load_state_dict(cleaned_dict, strict=False)
    model.eval()
    return model

model = load_model()

# --- 3. IMAGE PREPROCESSING ---
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, index = torch.max(probs, 1)
    
    return CLASSES[index.item()], confidence.item()

# --- 4. USER INTERFACE ---
uploaded_file = st.file_uploader("Upload a microscopic image of a fossil...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Sample', use_container_width=True)
    
    if st.button('Classify Specimen'):
        with st.spinner('Analyzing features...'):
            label, score = predict(image, model)
            
        st.success(f"### Prediction: **{label}**")
        st.info(f"**Confidence Level:** {score*100:.2f}%")
