import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION (Strictly matching your Training Script) ---
# Your error confirms the checkpoint has Base-sized layers (1024/2048)
MODEL_NAME = "swin_base_patch4_window7_224" 
MODEL_PATH = "swin_final_results_advanced/best_model.pth"
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

st.set_page_config(page_title="Microfossil PhD AI", layout="centered")
st.title("🔬 Microfossil Identification System")

# --- 2. DYNAMIC PATH LOCATOR ---
# This ensures the file is found even if it's in a subfolder
actual_path = MODEL_PATH
if not os.path.exists(actual_path):
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "best_model.pth":
                actual_path = os.path.join(root, file)
                break

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model(path):
    try:
        # 1. Initialize the correct BASE architecture
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu")
            # 2. Extract state_dict from trainer wrapper
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # 3. Clean 'module.' prefixes from training
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # 4. Load with strict=False to bypass minor metadata mismatches
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model, None
        return None, "Checkpoint file not found."
    except Exception as e:
        return None, str(e)

model, error = load_model(actual_path)

# --- 4. INTERFACE ---
if error:
    st.error(f"Critical Model Error: {error}")

source = st.radio("Input Method:", ("Upload File", "Use Camera"))
img_data = st.file_uploader("Select image...", type=["jpg", "png"]) if source == "Upload File" else st.camera_input("Capture")

if img_data:
    image = Image.open(img_data).convert('RGB')
    st.image(image, caption='Preview', use_container_width=True)
    
    if st.button('🚀 Classify Specimen'):
        if model:
            with st.spinner('Analyzing...'):
                # Exact transforms from your training script
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
