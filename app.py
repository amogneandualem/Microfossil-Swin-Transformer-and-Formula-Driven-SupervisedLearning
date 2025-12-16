import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION ---
# Path matches where you just committed the model
MODEL_PATH = "Best results/exfractal/best_model.pth"
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

st.set_page_config(page_title="Microfossil PhD Research", layout="wide")
st.title("🔬 Microfossil Species Identification System")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=32)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading weights: {e}")
    return None

model = load_model()

# --- 3. INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    source = st.radio("Input Source:", ("Upload File", "Camera Capture"))
    if source == "Upload File":
        img_file = st.file_uploader("Upload microfossil image...", type=["jpg", "png", "jpeg"])
    else:
        img_file = st.camera_input("Take photo")

with col2:
    if img_file is not None:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption="Current Sample", use_container_width=True)
        
        # --- THE CLASSIFY BUTTON ---
        if st.button('🚀 IDENTIFY SPECIES'):
            if model:
                with st.spinner('AI analyzing specimen...'):
                    preprocess = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    input_tensor = preprocess(image).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        prob = torch.nn.functional.softmax(output, dim=1)
                        conf, idx = torch.max(prob, 1)
                    
                    st.success(f"### Identification: **{CLASSES[idx.item()]}**")
                    st.metric("Confidence Score", f"{conf.item()*100:.2f}%")
            else:
                st.error("Model file not found. Check GitHub LFS status.")
