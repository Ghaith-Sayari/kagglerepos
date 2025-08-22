import os
import json
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from kaggle.api.kaggle_api_extended import KaggleApi

# -------------------------------
# 1. Setup Kaggle API using environment variables
kaggle_json_path = "kaggle.json"
if not os.path.exists(kaggle_json_path):
    with open(kaggle_json_path, "w") as f:
        json.dump({
            "username": os.environ["KAGGLE_USERNAME"],
            "key": os.environ["KAGGLE_KEY"]
        }, f)

os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()

api = KaggleApi()
api.authenticate()

# -------------------------------
# 2. Download model from Kaggle if not exists
MODEL_PATH = "model.pth"
KAGGLE_DATASET = "ghaithsayari/classification"  # Your Kaggle dataset path

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Kaggle..."):
        api.dataset_download_file(KAGGLE_DATASET, file_name="model.pth", path=".", unzip=True)

# -------------------------------
# 3. Load model
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

model = load_model()

# -------------------------------
# 4. Preprocessing & class names
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = [
    "Melanoma", "Melanocytic nevi", "Basal cell carcinoma",
    "Actinic keratosis", "Benign keratosis", "Dermatofibroma", "Vascular lesion"
]

# -------------------------------
# 5. Streamlit Interface
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

    # Show confidence for all classes
    st.subheader("Confidence per class")
    st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(class_names))})

