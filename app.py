from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import joblib
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\elwady\Downloads\skin_app (1)\skin_app"))

from utils.model_utils import create_model

app = Flask(__name__)

# محاولة تحميل نموذج الجلد
try:
    skin_model = joblib.load(r"C:\Users\elwady\Downloads\skin_app (1)\skin_app\skin_classifier_model.pkl")
    skin_model_available = True
except Exception as e:
    print(f"تعذر تحميل نموذج الجلد: {e}")
    skin_model = None
    skin_model_available = False

# تحميل نموذج التشخيص
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
disease_model = create_model(num_classes=10)
disease_model.load_state_dict(torch.load(r"C:\Users\elwady\Downloads\skin_app (1)\skin_app\my_best_model.pth", map_location=device))
disease_model.eval()

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
    if not is_skin:
            return jsonify({"prediction": "not_skin"})
        else:
            accuracy = 0.83



    # نكمل على التشخيص سواء استخدمنا كشف الجلد أو تخطيناه
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = disease_model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        prediction = disease_classes[pred_class]
        confidence = probs[0, pred_class].item()

    response = {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "skin_model_used": skin_model_available
    }

    if accuracy is not None:
        response["accuracy"] = accuracy

    return jsonify(response)
    
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # بياخد البورت من البيئة أو يستخدم 5000 كافتراضي
    app.run(host="0.0.0.0", port=port)
  