import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
crop_model = pickle.load(open("crop_model.pkl", "rb"))
# This is for crop prediction

# Disease info model: should be a lookup, not an ML model
# You may either load a DataFrame or a function as discussed earlier.
import pandas as pd
disease_data = pd.read_csv("Dataset/eggplant_diseases.csv")

# --- Utility function ---
def get_disease_info(disease_name: str):
    """Fetch disease information from CSV dataset."""
    result = disease_data[disease_data["Disease Name"].str.lower() == disease_name.lower()]
    if result.empty:
        return None
    return result.to_dict(orient="records")[0]  # Return first matching record

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route('/api/predict', methods=['POST'])
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


@flask_app.route('/api/disease', methods=['POST'])
def api_disease():
    data = request.get_json()
    disease_name = data.get("disease_name") if data else None
    if not disease_name:
        return jsonify({"error": "No disease_name provided"}), 400

    info = get_disease_info(disease_name.strip())
    if info is None:
        return jsonify({"error": f"No records found for: {disease_name}"}), 404
    return jsonify({"info": info})



if __name__ == "__main__":
    flask_app.run(debug=True)
