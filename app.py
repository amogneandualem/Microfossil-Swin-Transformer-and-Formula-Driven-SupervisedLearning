import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ==================== CONFIGURATION ====================
# Matches your provided training script
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
IMAGE_SIZE = 224

# The exact 32 classes from your dataset
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

st.set_page_config(page_title="Microfossil PhD AI", layout="wide")
st.title("🔬 Microfossil Identification System")

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    # Force Swin-Base architecture to match 1024-dim weights
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    
    # Confirmed path from your GitHub folder structure
    model_path = "swin_final_results_advanced/best_model.pth"

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            # Extract state_dict (handles trainer wrappers)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Remove 'module.' prefix if it exists from training
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Use strict=False to bypass minor naming variations in timm layers
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model, None
        except Exception as e:
            return None, str(e)
    return None, "Model file not found."

model, error = load_model()

# ==================== USER INTERFACE ====================
if error:
    st.error(f"🚨 System Error: {error}")

source = st.radio("Choose Input Method:", ("Upload Image File", "Use Camera"))
img_buffer = st.file_uploader("Select JPG/PNG", type=["jpg", "png", "jpeg"]) if source == "Upload Image File" else st.camera_input("Capture")

if img_buffer:
    image = Image.open(img_buffer).convert('RGB')
    st.image(image, caption="Specimen Preview", use_container_width=True)
    
    if st.button("🚀 Run AI Classification"):
        if model:
            with st.spinner('Analyzing specimen...'):
                # Matches your eval_transform in training
                preprocess = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = preprocess(image).unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    confidence, index = torch.max(probs, 1)
                
                st.success(f"### Identification: **{CLASSES[index.item()]}**")
                st.info(f"**Confidence Score:** {confidence.item()*100:.2f}%")
        else:
            st.error("Model not ready.")
