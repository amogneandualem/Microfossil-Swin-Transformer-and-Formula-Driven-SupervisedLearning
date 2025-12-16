import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ==================== CONFIGURATION ====================
# Swin-Base matches the 1024-dim layers in your error log
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

# ==================== ROBUST MODEL LOADING ====================
@st.cache_resource
def load_model():
    target_file = "best_model.pth"
    model_path = None
    
    # 1. Search every folder for the file
    for root, dirs, files in os.walk("."):
        if target_file in files:
            temp_path = os.path.join(root, target_file)
            # 2. Check for real file size (>100MB)
            if os.path.getsize(temp_path) > 100 * 1024 * 1024:
                model_path = temp_path
                break

    if not model_path:
        return None, "File found but it is a 1KB LFS pointer. Re-upload 332MB file from your PC."

    try:
        # 3. Initialize Swin-Base to fix size mismatch
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 4. Clean training prefixes
        state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, f"Successfully loaded from: {model_path}"
    except Exception as e:
        return None, str(e)

model, status = load_model()

# ==================== USER INTERFACE ====================
if not model:
    st.error(f"🚨 Setup Error: {status}")
    st.info("Large files take 5-10 minutes to sync on Streamlit Cloud.")
else:
    st.success(f"✅ AI Online | {status}")
    
    uploaded_file = st.file_uploader("Upload Microfossil Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Specimen", use_container_width=True)
        
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
