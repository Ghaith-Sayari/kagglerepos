import os
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from kaggle.api.kaggle_api_extended import KaggleApi

# -------------------------------
# 1. Download model from Kaggle (if not exists)
MODEL_PATH = "model.pth"
KAGGLE_DATASET = "ghaithsayari/classification"  # Your Kaggle dataset path

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Kaggle...")
    
    # Authenticate using secrets
    api = KaggleApi()
    api.authenticate()  # expects KAGGLE_USERNAME and KAGGLE_KEY in env
    
    # Download and unzip model file
    api.dataset_download_file(KAGGLE_DATASET, file_name="model.pth", path=".", unzip=True)

# -------------------------------
# 2. Load model
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

model = load_model()

# -------------------------------
# 3. Preprocessing & class names
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = [
    "Melanoma", "Melanocytic nevi", "Basal cell carcinoma",
    "Actinic keratosis", "Benign keratosis", "Dermatofibroma", "Vascular lesion"
]

# -------------------------------
# 4. Streamlit Interface
st.title("🩺 Skin Lesion Classifier (ISIC 2019)")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, 0)

    st.success(f"Prediction: **{class_names[pred.item()]}** ({conf.item()*100:.2f}% confidence)")

    # Show confidence for all classes as a bar chart
    st.subheader("Confidence per class")
    st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(class_names))})
