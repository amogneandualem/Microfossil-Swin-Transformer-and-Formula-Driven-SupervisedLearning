import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
MODEL_NAME = "swin_base_patch4_window7_224"

CLASSES = [
    'Acanthodesmia_micropora', 'Actinomma_leptoderma_boreale', 'Antarctissa_denticulata-cyrindrica', 
    'Antarctissa_juvenile', 'Antarctissa_longa-strelkovi', 'Botryocampe_antarctica', 
    'Botryocampe_inflatum-conithorax', 'Ceratocyrtis_historicosus', 'Cycladophora_bicornis', 
    'Cycladophora_cornutoides', 'Cycladophora_davisiana', 'Diatoms', 'Druppatractus_irregularis-bensoni', 
    'Eucyrtidium_spp', 'Fragments', 'Larcids_inner', 'Lithocampe_furcaspiculate', 
    'Lithomelissa_setosa-borealis', 'Lophophana_spp', 'Other_Nassellaria', 'Other_Spumellaria', 
    'Phormospyris_stabilis_antarctica', 'Phorticym_clevei-pylonium', 'Plectacantha_oikiskos', 
    'Pseudodictyophimus_gracilipes', 'Sethoconus_tablatus', 'Siphocampe_arachnea_group', 
    'Spongodiscus', 'Spongurus_pylomaticus', 'Sylodictya_spp', 'Zygocircus'
]

st.set_page_config(page_title="Microfossil PhD AI", layout="centered")
st.title("🔬 Microfossil Species Identification System")

# --- DEBUGGING SECTION ---
with st.expander("🛠️ Debug Folder Structure"):
    st.write(f"Looking for model at: `{MODEL_PATH}`")
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model file FOUND!")
        st.write(f"File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    else:
        st.error("❌ Model file NOT found at this path.")
        st.write("Files available in current directory:")
        st.code(os.listdir('.'))
        if os.path.exists('swin_final_results_advanced'):
            st.write("Files in 'swin_final_results_advanced':")
            st.code(os.listdir('swin_final_results_advanced'))

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
    return None

model = load_model()

# --- 3. INTERFACE ---
source = st.radio("Choose Input Method:", ("Upload Image File", "Use Camera"))
img_data = st.file_uploader("Select image...", type=["jpg", "jpeg", "png"]) if source == "Upload Image File" else st.camera_input("Capture")

if img_data is not None:
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption='Sample Preview', use_container_width=True)
    
    if st.button('🚀 Classify Specimen'):
        if model is not None:
            with st.spinner('AI analyzing...'):
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
                
                st.success(f"### Identification: **{CLASSES[idx.item()]}**")
                st.info(f"**Confidence Score:** {conf.item()*100:.2f}%")
        else:
            st.error("Model weights are missing. Please check the 'Debug Folder' above.")