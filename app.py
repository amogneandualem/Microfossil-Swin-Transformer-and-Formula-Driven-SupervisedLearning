import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# Updated to match the path you just added to git
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

st.set_page_config(page_title="Microfossil Identification", layout="wide")
st.title("🔬 Microfossil Species Identification System")

@st.cache_resource
def load_model():
    # Initialize Swin-B architecture
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=32)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    return None

model = load_model()

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    source = st.radio("Input Source:", ("Upload Image", "Camera Capture"))
    if source == "Upload Image":
        img_file = st.file_uploader("Upload a fossil image", type=["jpg", "png", "jpeg"])
    else:
        img_file = st.camera_input("Take a photo through the microscope")

with col2:
    if img_file is not None:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption="Uploaded Specimen", use_container_width=True)
        
        # --- THE CLASSIFY BUTTON ---
        # Only visible when an image is ready
        if st.button('🚀 RUN AI CLASSIFICATION'):
            if model:
                with st.spinner('Analyzing specimen...'):
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
                    
                    st.success(f"### Result: {CLASSES[idx.item()]}")
                    st.metric("Confidence", f"{conf.item()*100:.2f}%")
            else:
                st.error("Model not found. Please wait for the LFS upload to finish.")
