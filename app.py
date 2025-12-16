import streamlit as st
import torch
import timm
import os
from torchvision import transforms
from PIL import Image

# 1. CONFIGURATION
# Your .gitattributes shows the model is in this sub-folder
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

st.set_page_config(page_title="Microfossil Classifier", layout="centered")
st.title("🔬 Microfossil AI Identification")

# 2. MODEL LOADING (CACHED)
@st.cache_resource
def load_model():
    # Initialize architecture
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model not found at {MODEL_PATH}. Check folder names on GitHub.")
        return None

    # Load weights from the LFS file
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint
    
    # Clean prefixes if necessary
    cleaned_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                    for k, v in state_dict.items() if not k.startswith('head.')}
    
    model.load_state_dict(cleaned_dict, strict=False)
    model.eval()
    return model

# 3. INTERFACE & PREDICTION
model = load_model()

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
        conf, idx = torch.max(probs, 1)
    return CLASSES[idx.item()], conf.item()

uploaded_file = st.file_uploader("Upload microfossil image...", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Specimen', use_container_width=True)
    if st.button('Identify Species'):
        label, score = predict(image, model)
        st.success(f"**Result: {label}** ({score*100:.1f}% confidence)")
