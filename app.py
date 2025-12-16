import streamlit as st
import torch
import timm
import os
from torchvision import transforms
from PIL import Image

# ------------------ 1. CONFIGURATION ------------------
# Path matches your swin_final_results_advanced folder from GitHub
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
MODEL_NAME = "swin_base_patch4_window7_224"

# Full 32-species list for your PhD research
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

# Page config
st.set_page_config(page_title="Microfossil Species Classifier", layout="centered")
st.title("🔬 Microfossil AI Identification")
st.write("Swin Transformer Base (Swin-B) Model Architecture")

# ------------------ 2. LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model file missing! Expected path: {MODEL_PATH}")
        return None

    # Load weights (Auto-pulled via Git LFS on Streamlit Cloud)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint
    
    # Clean keys for consistency
    cleaned_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                    for k, v in state_dict.items() if not k.startswith('head.')}
    
    model.load_state_dict(cleaned_dict, strict=False)
    model.eval()
    return model

model = load_model()

# ------------------ 3. PREDICTION LOGIC ------------------
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

# ------------------ 4. INTERFACE ------------------
# Choice of input source
source = st.radio("Choose Input Method:", ("Upload Image File", "Use Camera"))

img_data = None
if source == "Upload Image File":
    img_data = st.file_uploader("Select a microfossil image...", type=["jpg", "jpeg", "png"])
else:
    img_data = st.camera_input("Capture a microscope sample")

# Logic to show Classify button ONLY after image is ready
if img_data is not None:
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption='Specimen to Classify', use_container_width=True)
    
    # Classify button appears now
    if st.button('🚀 Run AI Classification'):
        if model is not None:
            with st.spinner('Extracting features and classifying...'):
                label, score = predict(image, model)
            
            # Results display
            st.success(f"### Identification: **{label}**")
            st.metric(label="Confidence Level", value=f"{score*100:.2f}%")
        else:
            st.error("Model failed to initialize.")
