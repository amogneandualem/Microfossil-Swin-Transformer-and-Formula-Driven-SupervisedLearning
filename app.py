import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# ==================== CONFIGURATION ====================
# This matches your Config class in the training script
MODEL_NAME = "swin_base_patch4_window7_224" 
NUM_CLASSES = 32
IMAGE_SIZE = 224

# The exact 32 classes from your dataset directories
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
st.markdown("---")

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    # 1. Initialize the Base model structure (1024-dim)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    
    # 2. Dynamic search for best_model.pth (checks Replicates folders)
    model_file = None
    for root, dirs, files in os.walk("."):
        if "best_model.pth" in files:
            full_path = os.path.join(root, "best_model.pth")
            # Ensure we pick the actual large weights file, not a small LFS pointer
            if os.path.getsize(full_path) > 100 * 1024 * 1024: 
                model_file = full_path
                break

    if model_file:
        try:
            # Load weights (map to CPU for Streamlit)
            checkpoint = torch.load(model_file, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Remove 'module.' prefix from DataParallel/Distributed training
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load into model
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            return model, f"Loaded: {model_file}"
        except Exception as e:
            return None, f"Error loading weights: {str(e)}"
    return None, "Model weights (best_model.pth) not found. Ensure Git LFS is synced."

model, status = load_model()

# ==================== USER INTERFACE ====================
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Input Specimen")
    source = st.radio("Choose source:", ("Upload Image", "Use Camera"))
    
    img_buffer = None
    if source == "Upload Image":
        img_buffer = st.file_uploader("Select JPG/PNG", type=["jpg", "png", "jpeg"])
    else:
        img_buffer = st.camera_input("Take photo")

with col2:
    st.header("2. Analysis Result")
    if img_buffer:
        image = Image.open(img_buffer).convert('RGB')
        st.image(image, caption="Current Specimen", use_container_width=True)
        
        if st.button("🚀 Run AI Classification"):
            if model:
                # Same normalization as Config.eval_transform in training
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
                
                result_class = CLASSES[index.item()]
                conf_score = confidence.item() * 100
                
                st.success(f"### Identification: **{result_class}**")
                st.metric("Confidence Score", f"{conf_score:.2f}%")
                
                # Show top 3 probabilities
                top3_prob, top3_idx = torch.topk(probs, 3)
                st.write("Top 3 Predictions:")
                for i in range(3):
                    st.write(f"- {CLASSES[top3_idx[0][i].item()]}: {top3_prob[0][i].item()*100:.1f}%")
            else:
                st.error(f"System Error: {status}")
    else:
        st.info("Awaiting specimen input...")

# Sidebar Info
st.sidebar.title("System Status")
if model:
    st.sidebar.success("✅ AI Model Online")
    st.sidebar.info(status)
else:
    st.sidebar.error("❌ AI Model Offline")
    st.sidebar.warning(status)
