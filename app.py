import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os 
from flask_cors import CORS
import pandas as pd
flask_app = Flask(__name__)
CORS(flask_app)

crop_model = pickle.load(open("crop_model.pkl", "rb"))
disease_data = pd.read_csv("Dataset/eggplant_diseases.csv", encoding='latin1')
crop_data = pd.read_csv("Dataset/eggplant_details.csv", encoding='latin1')

def get_disease_info(disease_name: str):
    """Fetch disease information from CSV dataset."""
    result = disease_data[disease_data["Disease Name"].str.lower() == disease_name.lower()]
    if result.empty:
        return None
    return result.to_dict(orient="records")[0] 

def get_crop_info(crop_name: str):
    crop_name_clean = crop_name.strip().lower()
    result = crop_data[crop_data["Crop Name"].astype(str).str.lower().str.strip() == crop_name_clean]
    if result.empty:
        return None
    return result.to_dict(orient="records")[0]

@flask_app.route("/")
def home():
    return jsonify({"status": "Server is running"})

@flask_app.route('/api/predict', methods=['GET','POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    try:
        float_features = [float(data[k]) for k in sorted(data.keys())]
        features = np.array([float_features])
        prediction = crop_model.predict(features)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route('/api/disease', methods=['GET','POST'])
def api_disease():
    data = request.get_json()
    disease_name = data.get("disease_name") if data else None
    if not disease_name:
        return jsonify({"error": "No disease_name provided"}), 400

    info = get_disease_info(disease_name.strip())
    if info is None:
        return jsonify({"error": f"No records found for: {disease_name}"}), 404
    return jsonify({"info": info})

@flask_app.route('/api/crop', methods=['POST'])
def api_crop():
    data = request.get_json()
    crop_name = data.get("crop_name") if data else None
    if not crop_name:
        return jsonify({"error": "No crop_name provided"}), 400
    info = get_crop_info(crop_name)
    if info is None:
        return jsonify({"error": f"No records found for: {crop_name}"}), 404
    return jsonify({"info": info})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)
