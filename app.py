import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
# Use your exact Hugging Face path here
REPO_ID = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth 
FILENAME = "best_model.pth"

st.set_page_config(page_title="Microfossil ID", page_icon="🔬")
st.title("🔬 Microfossil Identification System")

@st.cache_resource
def load_model_direct():
    # 1. Download only the weights file from HF to Streamlit's RAM
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # 2. Build the Swin architecture
    # We force embed_dim=128 to match your '1024/2048' checkpoint sizes
    model = timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=False,
        num_classes=32,
        embed_dim=128,           
        depths=(2, 2, 18, 2),    
        num_heads=(4, 8, 16, 32) 
    )
    
    # 3. Load the weights
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Clean keys (removes 'module.' or 'backbone.' prefixes)
    new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                      for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# Initialize
try:
    model = load_model_direct()
    st.success("✅ Connected to Hugging Face weights!")
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

# --- PREDICTION UI ---
uploaded_file = st.file_uploader("Upload a fossil image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, width=300)
    
    # Standard Swin Transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        input_tensor = tf(img).unsqueeze(0)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    st.header(f"Result: Class {prediction}")
