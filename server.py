from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import os
import requests

app = Flask(__name__)

# --- GitHub raw base URL ---
RAW_BASE_URL = "https://raw.githubusercontent.com/YoussefPoulis/cancer_backend/0a14b687746299f92a4d61d4fa3b63194d268803/"

# --- Main model ---
MAIN_MODEL_FILENAME = "main_cancer_model.keras"
MAIN_MODEL_URL = RAW_BASE_URL + MAIN_MODEL_FILENAME

if not os.path.exists(MAIN_MODEL_FILENAME):
    print(f"[INFO] Downloading main model from GitHub...")
    r = requests.get(MAIN_MODEL_URL)
    with open(MAIN_MODEL_FILENAME, "wb") as f:
        f.write(r.content)
    print("[INFO] Main model downloaded successfully.")

print(f"[INFO] Loading main model: {MAIN_MODEL_FILENAME}")
main_model = load_model(MAIN_MODEL_FILENAME)
print("[INFO] Main model loaded successfully.\n")

# --- Class names ---
class_names = [
    "ALL", "Bladder Cancer", "Brain Cancer", "Breast Cancer", "Cervical Cancer",
    "Esophageal Cancer", "Kidney Cancer", "Lung and Colon Cancer", "Lymphoma",
    "Oral Cancer", "Ovarian Cancer", "Pancreatic Cancer", "Skin Cancer", "Thyroid Cancer"
]

# --- Subclasses dictionary ---
subclasses = {
    "ALL": ["all_benign", "all_early", "all_pre", "all_pro"],
    "Bladder Cancer": ["bladder_muscle_invasive", "bladder_non_muscle_invasive"],
    "Brain Cancer": ["brain_glioma", "brain_menin", "brain_tumor"],
    "Breast Cancer": ["breast_benign", "breast_malignant"],
    "Cervical Cancer": ["cervix_dyk", "cervix_koc", "cervix_mep", "cervix_pab", "cervix_sfi"],
    "Esophageal Cancer": ["esophagus_benign", "esophagus_malignant"],
    "Kidney Cancer": ["kidney_normal", "kidney_tumor"],
    "Lung and Colon Cancer": ["colon_aca", "colon_bnt", "lung_aca", "lung_bnt", "lung_scc"],
    "Lymphoma": ["lymph_cll", "lymph_fl", "lymph_mcl"],
    "Oral Cancer": ["oral_normal", "oral_scc"],
    "Ovarian Cancer": [
        "ovarian_clear_cell_carcinoma", "ovarian_endometrioid",
        "ovarian_low_grade_serous", "ovarian_mucinous_carcinoma",
        "ovarian_high_grade_serous_carcinoma"
    ],
    "Pancreatic Cancer": ["pancreatic_normal", "pancreatic_tumor"],
    "Skin Cancer": [
        "Skin_Acne", "Skin_Actinic Keratosis", "Skin_Basal Cell Carcinoma",
        "Skin_Chickenpox", "Skin_Dermato Fibroma", "Skin_Dyshidrotic Eczema",
        "Skin_Melanoma", "Skin_Nail Fungus", "Skin_Nevus", "Skin_Normal Skin",
        "Skin_Pigmented Benign Keratosis", "Skin_Ringworm", "Skin_Seborrheic Keratosis",
        "Skin_Squamous Cell Carcinoma", "Skin_Vascular Lesion"
    ],
    "Thyroid Cancer": ["thyroid_benign", "thyroid_malignant"]
}

# --- Cache for loaded submodels ---
loaded_submodels = {}

def preprocess_image(img):
    """Resize and preprocess image for EfficientNet model."""
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    print(f"[DEBUG] Image preprocessed to shape: {img_array.shape}")
    return img_array

def download_and_load_submodel(cancer_type):
    """Download submodel from GitHub raw URL if not exists, then load it."""
    if cancer_type not in loaded_submodels:
        submodel_filename = f"{cancer_type.lower().replace(' ', '_')}_model.keras"
        submodel_url = RAW_BASE_URL + submodel_filename

        # Download if not exists
        if not os.path.exists(submodel_filename):
            print(f"[INFO] Downloading submodel for {cancer_type}...")
            r = requests.get(submodel_url)
            if r.status_code == 200:
                with open(submodel_filename, "wb") as f:
                    f.write(r.content)
                print(f"[INFO] Submodel downloaded: {submodel_filename}")
            else:
                print(f"[WARNING] Submodel not found on GitHub: {submodel_filename}")
                loaded_submodels[cancer_type] = None
                return None

        # Load the model
        print(f"[INFO] Loading submodel: {submodel_filename}")
        loaded_submodels[cancer_type] = load_model(submodel_filename)
        print(f"[INFO] Submodel loaded successfully: {submodel_filename}")
    return loaded_submodels[cancer_type]

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img_array = preprocess_image(img)

    # --- Main prediction ---
    main_pred = main_model.predict(img_array)
    main_index = np.argmax(main_pred)
    cancer_type = class_names[main_index]
    main_confidence = float(main_pred[0][main_index])
    print(f"[DEBUG] Main prediction: {cancer_type} ({main_confidence:.4f})")

    # --- Subclass prediction ---
    submodel = download_and_load_submodel(cancer_type)
    if submodel:
        sub_pred = submodel.predict(img_array)
        if len(sub_pred[0]) == len(subclasses[cancer_type]):
            sub_index = np.argmax(sub_pred)
            subclass_name = subclasses[cancer_type][sub_index]
            subclass_confidence = float(sub_pred[0][sub_index])
        else:
            subclass_name = None
            subclass_confidence = None
            print(f"[DEBUG] Subclass output size mismatch for {cancer_type}")
    else:
        subclass_name = None
        subclass_confidence = None
        print(f"[DEBUG] No submodel found for {cancer_type}")

    return jsonify({
        "cancer_type": cancer_type,
        "confidence": main_confidence,
        "subclass": subclass_name,
        "subclass_confidence": subclass_confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
