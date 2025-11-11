from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# --- Main model ---
main_model_path = "main_cancer_model.keras"
print(f"Loading main model from: {main_model_path}")
main_model = load_model(main_model_path)
print("Main model loaded successfully.\n")

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

def load_submodel(cancer_type):
    """Load submodel for a given cancer type (without 'models/' path)."""
    if cancer_type not in loaded_submodels:
        submodel_name = f"{cancer_type.lower().replace(' ', '_')}_model.keras"
        print(f"[DEBUG] Looking for submodel: {submodel_name}")
        if os.path.exists(submodel_name):
            loaded_submodels[cancer_type] = load_model(submodel_name)
            print(f"[DEBUG] Submodel loaded successfully: {submodel_name}")
        else:
            loaded_submodels[cancer_type] = None
            print(f"[DEBUG] Submodel file NOT FOUND: {submodel_name}")
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
    print(f"[DEBUG] Main prediction raw array: {main_pred}")

    # --- Subclass prediction ---
    submodel = load_submodel(cancer_type)
    if submodel:
        sub_pred = submodel.predict(img_array)
        print(f"[DEBUG] Subclass raw output: {sub_pred}")
        if len(sub_pred[0]) == len(subclasses[cancer_type]):
            sub_index = np.argmax(sub_pred)
            subclass_name = subclasses[cancer_type][sub_index]
            subclass_confidence = float(sub_pred[0][sub_index])
            print(f"[DEBUG] Predicted subclass: {subclass_name} ({subclass_confidence:.4f})")
        else:
            subclass_name = None
            subclass_confidence = None
            print(f"[DEBUG] Subclass output size mismatch for {cancer_type}")
    else:
        subclass_name = None
        subclass_confidence = None
        print(f"[DEBUG] No submodel found for {cancer_type}")

    print("---------------------------------------------------\n")

    return jsonify({
        "cancer_type": cancer_type,
        "confidence": main_confidence,
        "subclass": subclass_name,
        "subclass_confidence": subclass_confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)

