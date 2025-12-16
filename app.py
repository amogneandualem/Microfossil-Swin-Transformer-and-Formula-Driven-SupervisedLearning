import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
import os

# Model URL and path
MODEL_URL = "https://huggingface.co/spaces/amogneandualem/microfossil-classifier/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

# Class names (update with your actual class names)
CLASS_NAMES = [...]  # 32 class names

# Download model if not exists
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... This may take a few minutes."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("Model downloaded!")

# Load model
@st.cache_resource
def load_model():
    download_model()
    # Import timm here to avoid unnecessary import if not needed
    import timm
    
    # Create model (adjust architecture if needed)
    model = timm.create_model(
        "swin_large_patch4_window7_224",
        pretrained=False,
        num_classes=len(CLASS_NAMES)
    )
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    
    # Adjust keys if necessary
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Streamlit app
st.title("Microfossil Classifier")

# Load model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            # Preprocess and predict
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            # Display results
            st.subheader("Results")
            for i in range(5):
                st.write(f"{i+1}. {CLASS_NAMES[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")
