from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import joblib
import numpy as np
from utils.model_utils import create_model
import os

app = Flask(__name__)

# تحميل نموذج الجلد
skin_model = joblib.load("skin_classifier_model.pkl")

# تحميل نموذج التشخيص
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
disease_model = create_model(num_classes=10)
disease_model.load_state_dict(torch.load("my_best_model.pth", map_location=device))
disease_model.eval()

# أسماء الأمراض
disease_classes = [
    'Acne',
    'Actinic Keratosis',
    'Atopic Dermatitis',
    'Lichen Planus',
    'Nail Disease',
    'Nevus',
    'Skin Canser',
    'Squamous Cell Carcinoma',
    'Vascular Tumors',
    'Vitiligo'
]

# التحويلات للصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# تحويل صورة PIL إلى RGB array ثم BGR
def get_bgr_from_image(img):
    img = img.convert("RGB")
    rgb = np.array(img)
    bgr = rgb[:, :, ::-1]  # قلب القنوات
    avg = bgr.mean(axis=(0, 1))
    return avg.tolist()  # [B, G, R]

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    image = Image.open(img_file.stream)

    # مرحلة 1: كشف إذا كان جلد أو لا
    b, g, r = get_bgr_from_image(image)
    skin_pred = skin_model.predict([[b, g, r]])[0]
    is_skin = int(skin_pred) == 1

    result = {
        "is_skin": is_skin,
        "accuracy": 0.77,
        "prediction": "not_skin"
    }

    if is_skin:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = disease_model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            result["prediction"] = disease_classes[pred_class]

    return jsonify(result)