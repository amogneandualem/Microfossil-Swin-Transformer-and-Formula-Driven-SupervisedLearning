import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION ---
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
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

st.set_page_config(page_title="Microfossil Identification", layout="centered")
st.title("🔬 Microfossil Identification System")

# --- 2. DYNAMIC FILE SEARCHER ---
def find_model_file(filename="best_model.pth"):
    for root, dirs, files in os.walk("."):
        if filename in files:
            full_path = os.path.join(root, filename)
            # Check if it's the actual file or just a 1KB LFS pointer
            if os.path.getsize(full_path) > 1000000: # Larger than 1MB
                return full_path
    return None

actual_path = find_model_file()

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model(path):
    if not path:
        return None, "Checkpoint not found or file is still an LFS pointer (1KB)."
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model(actual_path)

# --- 4. USER INTERFACE ---
if error:
    st.error(f"🚨 Model Error: {error}")
    st.info("Wait 5 minutes for Git LFS to finish downloading the large file.")

source = st.radio("Input Method:", ("Upload File", "Use Camera"))
img_data = st.file_uploader("Select image...", type=["jpg", "png"]) if source == "Upload File" else st.camera_input("Capture")

if img_data:
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption='Preview', use_container_width=True)
    if st.button('🚀 Classify Specimen'):
        if model:
            with st.spinner('Analyzing...'):
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, 1)
                st.success(f"### Result: **{CLASSES[idx.item()]}**")
                st.info(f"**Confidence Score:** {conf.item()*100:.2f}%")
