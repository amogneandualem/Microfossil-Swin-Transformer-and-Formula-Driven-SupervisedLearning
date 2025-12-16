"""
Microfossil Classifier - Streamlit Cloud Compatible
Simplified version that works on Streamlit Cloud
"""

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Microfossil Classifier",
    page_icon="🔬",
    layout="centered"
)

# ========== CONSTANTS ==========
CLASS_NAMES = [
    'Acanthodesmia_micropora', 'Actinomma_leptoderma_boreale',
    'Antarctissa_denticulata-cyrindrica', 'Antarctissa_juvenile',
    'Antarctissa_longa-strelkovi', 'Botryocampe_antarctica',
    'Botryocampe_inflatum-conithorax', 'Ceratocyrtis_historicosus',
    'Cycladophora_bicornis', 'Cycladophora_cornutoides',
    'Cycladophora_davisiana', 'Diatoms',
    'Druppatractus_irregularis-bensoni', 'Eucyrtidium_spp',
    'Fragments', 'Larcids_inner', 'Lithocampe_furcaspiculate',
    'Lithocampe_platycephala', 'Lithomelissa_setosa-borealis',
    'Lophophana_spp', 'Other_Nassellaria', 'Other_Spumellaria',
    'Phormospyris_stabilis_antarctica', 'Phorticym_clevei-pylonium',
    'Plectacantha_oikiskos', 'Pseudodictyophimus_gracilipes',
    'Sethoconus_tablatus', 'Siphocampe_arachnea_group',
    'Spongodiscus', 'Spongurus_pylomaticus',
    'Sylodictya_spp', 'Zygocircus'
]

# ========== SIMPLE MODEL LOADER ==========
@st.cache_resource
def load_model():
    """Load model with fallbacks for compatibility"""
    try:
        # Try to import timm
        import timm
        
        # Create model
        model = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load checkpoint
        checkpoint = torch.load("model.pth", map_location='cpu')
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint
        
        # Clean keys
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            cleaned_state_dict[k] = v
        
        # Load weights
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Model loading error: {str(e)[:200]}")
        return None

# ========== MAIN APP ==========
def main():
    st.title("🔬 Microfossil Classifier")
    st.markdown("**AI-powered classification of microfossil images**")
    
    # Check if model exists
    if os.path.exists("model.pth"):
        st.success("✅ Model file found")
    else:
        st.warning("⚠️ Model file not found - running in demo mode")
        st.info("Please upload model.pth to your GitHub repository")
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'png', 'jpeg']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your image", use_container_width=True)
            
            # Show file info
            with st.expander("📊 Image Details"):
                st.write(f"**Name:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                st.write(f"**Dimensions:** {image.size}")
                st.write(f"**Format:** {image.format}")
    
    with col2:
        st.subheader("🔍 Analysis")
        
        if uploaded_file:
            if st.button("🚀 Classify", type="primary"):
                with st.spinner("Processing..."):
                    # Load model
                    model = load_model()
                    
                    if model:
                        # Transform image
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], 
                                               [0.229, 0.224, 0.225])
                        ])
                        
                        img_tensor = transform(image).unsqueeze(0)
                        
                        # Predict
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            top_prob, top_idx = torch.max(probs, 1)
                        
                        # Display results
                        idx = top_idx.item()
                        if idx < len(CLASS_NAMES):
                            st.success(f"🎯 **{CLASS_NAMES[idx]}**")
                            st.metric("Confidence", f"{top_prob.item()*100:.1f}%")
                            
                            # Show top 5
                            top_probs, top_indices = torch.topk(probs, 5)
                            st.subheader("Top 5 Predictions:")
                            for i in range(5):
                                idx = top_indices[0][i].item()
                                if idx < len(CLASS_NAMES):
                                    conf = top_probs[0][i].item() * 100
                                    st.write(f"{i+1}. {CLASS_NAMES[idx]}: {conf:.1f}%")
                        else:
                            st.error("Prediction index out of range")
                    else:
                        st.warning("Model not loaded - showing demo results")
                        st.info("**Demo prediction:** Actinomma_leptoderma_boreale")
                        st.info("**Confidence:** 92.5%")
        else:
            st.info("👆 Upload an image to begin")
            
            # Quick guide
            with st.expander("📚 How to use"):
                st.write("""
                1. **Upload** a microfossil image
                2. Click **"Classify"** button
                3. View **AI prediction** with confidence
                4. See **top 5** possible classifications
                
                **Note:** Model must be uploaded as `model.pth` in the repository.
                """)
    
    # Footer
    st.markdown("---")
    st.caption("Deployed on Streamlit Cloud | Built with PyTorch")

if __name__ == "__main__":
    main()
