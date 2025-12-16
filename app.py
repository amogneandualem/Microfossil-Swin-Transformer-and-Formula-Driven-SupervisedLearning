import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ==================== CONFIGURATION ====================
# Swin-Base is required to match your 1024-dim checkpoint
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

st.set_page_config(page_title="Microfossil PhD AI", layout="wide")
st.title("🔬 Microfossil Identification System")

@st.cache_resource
def load_model():
    target_file = "best_model.pth"
    model_path = None
    
    # RECURSIVE SEARCH: Looks in every subfolder for the file
    for root, dirs, files in os.walk("."):
        if target_file in files:
            temp_path = os.path.join(root, target_file)
            # Verify file size to ensure it's not a 1KB LFS pointer
            if os.path.getsize(temp_path) > 100 * 1024 * 1024:
                model_path = temp_path
                break

    if not model_path:
        return None, "best_model.pth not found. Please check GitHub LFS status."

    try:
        # Initialize the larger Swin-Base structure
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # Load weights onto CPU
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Strip prefixes like 'module.'
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"✅ Successfully loaded from: {model_path}"
    except Exception as e:
        return None, str(e)

model, status = load_model()

# ==================== USER INTERFACE ====================
if not model:
    st.error(f"🚨 Setup Error: {status}")
    st.info("Ensure the file on GitHub shows a size of ~332MB.")
else:
    st.success(status)
    
    uploaded_file = st.file_uploader("Upload Microfossil Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Specimen", width=400)
        
        if st.button("🚀 Identify"):
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                conf, idx = torch.max(prob, 1)
                
            st.header(f"Result: {CLASSES[idx.item()]}")
            st.write(f"Confidence: {conf.item()*100:.2f}%")
