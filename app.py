import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- 1. CONFIGURATION ---
# We will use this as a fallback, but the code below will search for it
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

st.set_page_config(page_title="Microfossil PhD AI", layout="centered")
st.title("🔬 Microfossil Identification System")

# --- 2. DYNAMIC FILE SEARCHER ---
# This locates the file even if it's buried in subfolders
def find_model_file(filename="best_model.pth"):
    for root, dirs, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)
    return None

actual_model_path = find_model_file()

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model(path):
    if not path:
        return None, "File 'best_model.pth' was not found anywhere in the repository."
    
    try:
        # Initialize Swin-Base (1024-dim) to match your checkpoint
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        checkpoint = torch.load(path, map_location="cpu")
        
        # Extract state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Remove 'module.' prefix if it exists
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load weights with strict=False to handle metadata differences
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# Attempt to load
model, error = load_model(actual_model_path)

# --- 4. USER INTERFACE ---
if error:
    st.error(f"🚨 Model Error: {error}")
    if actual_model_path:
        st.info(f"File was found at: {actual_model_path}")
    else:
        st.warning("Double check that you have committed 'best_model.pth' to GitHub.")

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
        else:
            st.error("Model not ready. Check the errors above.")
