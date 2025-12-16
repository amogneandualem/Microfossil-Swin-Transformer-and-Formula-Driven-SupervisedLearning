import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
import os

# Setup
st.set_page_config(page_title="Microfossil Classifier", page_icon="🔬")

# Model URL
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/model.pth"
MODEL_PATH = "model.pth"

# Download model
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)

# Load model
@st.cache_resource
def load_model():
    import timm
    model = timm.create_model("swin_large_patch4_window7_224", pretrained=False, num_classes=32)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# Main app
st.title("🔬 Microfossil Classifier")
st.write("Upload a microfossil image for classification")

# Upload
uploaded_file = st.file_uploader("Choose image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Classify"):
        with st.spinner("Processing..."):
            # Load model
            model = load_model()
            
            # Preprocess
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_idx = torch.max(probs, 1)
            
            # Show result
            st.success(f"Confidence: {top_prob.item()*100:.1f}%")
            st.write(f"Class index: {top_idx.item()}")
