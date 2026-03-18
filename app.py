# -------------------------------------------------
# Force deterministic behavior (VERY IMPORTANT)
# -------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "0"

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# -------------------------------------------------
# Flask & ML imports
# -------------------------------------------------
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from werkzeug.utils import secure_filename

# -------------------------------------------------
# Flask App Setup
# -------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------------------------
# Load Model ONCE (DO NOT MOVE THIS)
# -------------------------------------------------
MODEL_PATH = "model/model_blood_group_detection_resnet.h5"
model = load_model(MODEL_PATH)

print("✅ Model loaded")
print("🔍 Model input shape:", model.input_shape)

# -------------------------------------------------
# CLASS LABELS (MUST MATCH TRAINING ORDER)
# -------------------------------------------------
labels = {
    0: "A+",
    1: "A-",
    2: "AB+",
    3: "AB-",
    4: "B+",
    5: "B-",
    6: "O+",
    7: "O-"
}

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # -------------------------------------------------
        # Image Preprocessing (MATCH TRAINING EXACTLY)
        # -------------------------------------------------
        img = image.load_img(
            filepath,
            target_size=(256, 256),   # 🔴 change if your model uses 256
            color_mode="rgb"
        )

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # -------------------------------------------------
        # Prediction (FORCE INFERENCE MODE)
        # -------------------------------------------------
        preds = model(x, training=False).numpy()

        class_index = int(np.argmax(preds))
        confidence = float(preds[0][class_index] * 100)
        predicted_label = labels[class_index]

        # Debug (optional)
        print("RAW OUTPUT:", np.round(preds[0], 5))
        print("PREDICTED:", predicted_label, confidence)

        # Cleanup
        os.remove(filepath)

        return jsonify({
            "label": predicted_label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": "Prediction failed"}), 500


# -------------------------------------------------
# Run Server
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
