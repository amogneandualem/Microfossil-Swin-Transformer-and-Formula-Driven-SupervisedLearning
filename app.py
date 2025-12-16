import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION ---
# Since LFS finished, the file is now in: Best results/exfractal/best_model.pth
# Or if you moved it to the root, just "best_model.pth"
MODEL_PATH = "Best results/exfractal/best_model.pth" 
MODEL_NAME = "swin_base_patch4_window7_224"

# 32 Microfossil Classes
CLASSES = [
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

@st.cache_resource
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    # Load the file that was just pushed via LFS
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint
    
    # Clean prefixes
    cleaned_dict = {k.replace('module.', '').replace('backbone.', ''): v 
                    for k, v in state_dict.items() if not k.startswith('head.')}
    
    model.load_state_dict(cleaned_dict, strict=False)
    model.eval()
    return model

st.title("🔬 Microfossil Classifier")
# Rest of UI code...
