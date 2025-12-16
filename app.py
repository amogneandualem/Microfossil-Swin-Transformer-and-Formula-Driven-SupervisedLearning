import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ==================== CONFIGURATION ====================
# Force Swin-Base to match the 1024/2048-dim weights in your error log
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
IMAGE_SIZE = 224

# Exact path based on your GitHub folder structure
MODEL_PATH = "swin_final_results_advanced/best_model.pth"

# Exact class list for your microfossil dataset
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
    # 1. Check if file exists and isn't just an LFS pointer
    if not os.path.exists(MODEL_PATH):
        return None, f"File not found at {MODEL_PATH}. Check your folder structure."
    
    if os.path.getsize(MODEL_PATH) < 1000000:
        return None, "Git LFS Error: File is a 1KB pointer. Wait for download."

    try:
        # 2. Initialize the LARGER Swin-Base model structure
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # 3. Load weights onto CPU for Streamlit Cloud
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 4. Remove 'module.' prefixes from DataParallel training
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 5. Load with strict=False to handle minor naming differences
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, "Model Ready"
    except Exception as e:
        return None, str(e)

model, status = load_model()

# ==================== USER INTERFACE ====================
if model is None:
    st.error(f"🚨 Model Error: {status}")
    if st.button("🔄 Reload App"):
        st.rerun()
else:
    st.success("✅ AI Model Loaded (Swin-Base)")
    
    img_file = st.file_uploader("Upload Microfossil Image", type=["jpg", "png", "jpeg"])
    
    if img_file:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption="Current Specimen", use_container_width=True)
        
        if st.button("🚀 Run AI Classification"):
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
            
            st.markdown(f"## Identification: **{CLASSES[index.item()]}**")
            st.info(f"Confidence Score: {confidence.item()*100:.2f}%")
