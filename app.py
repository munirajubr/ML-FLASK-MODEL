# app.py — cleaned, debugged, no timestamp handling
import os
import io
import base64
import json
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- load env ----
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME") or os.getenv("MONGO_DBNAME")
MODEL_API_URL = os.getenv("MODEL_API_URL") or os.getenv("PREDICTION_API_URL")

# model request settings (configurable via .env)
MODEL_API_TIMEOUT = float(os.getenv("MODEL_API_TIMEOUT", "60"))          # seconds
MODEL_API_MAX_RETRIES = int(os.getenv("MODEL_API_MAX_RETRIES", "3"))
MODEL_API_BACKOFF_FACTOR = float(os.getenv("MODEL_API_BACKOFF_FACTOR", "1.0"))

# ---- quick validation ----
if not MONGO_URI:
    raise RuntimeError("MONGODB_URI (or MONGO_URI) not set in environment/.env")
if not DATABASE_NAME:
    raise RuntimeError("DATABASE_NAME not set in environment/.env")
if not MODEL_API_URL:
    raise RuntimeError("MODEL_API_URL (or PREDICTION_API_URL) not set in environment/.env")

# ---- logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.info("Starting app (DB=%s, ModelURL=%s...)", DATABASE_NAME, MODEL_API_URL[:80] + ("..." if len(MODEL_API_URL) > 80 else ""))

# ---- Flask & Mongo ----
app = Flask(__name__)
CORS(app)

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[DATABASE_NAME]
except Exception as e:
    logger.exception("Failed to connect to MongoDB: %s", str(e))
    raise

# ---- requests session with retry ----
def create_retry_session(retries=3, backoff_factor=1.0, status_forcelist=(500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

prediction_session = create_retry_session(
    retries=MODEL_API_MAX_RETRIES,
    backoff_factor=MODEL_API_BACKOFF_FACTOR
)

# ---- helpers ----
def _convert_bson(o):
    """Convert ObjectId to str recursively so JSON is serializable."""
    if isinstance(o, ObjectId):
        return str(o)
    if isinstance(o, dict):
        return {k: _convert_bson(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_convert_bson(i) for i in o]
    return o

def _looks_like_base64(s: str) -> bool:
    """Quick heuristic for base64-like string."""
    if not isinstance(s, str) or len(s) < 20:
        return False
    try:
        base64.b64decode(s[:50], validate=True)
        return True
    except Exception:
        return False

def _extract_main_prediction(pred_json):
    """Try to pick the most likely main prediction from various response shapes."""
    if not isinstance(pred_json, dict):
        return pred_json
    keys_try = ("main_prediction", "main_pred", "prediction", "pred", "label", "result", "class")
    for k in keys_try:
        if k in pred_json and pred_json[k] not in (None, {}):
            return pred_json[k]
    # check nested dicts one level
    for v in pred_json.values():
        if isinstance(v, dict):
            for k in keys_try:
                if k in v and v[k] not in (None, {}):
                    return v[k]
    # list of predictions
    if "predictions" in pred_json and isinstance(pred_json["predictions"], list) and pred_json["predictions"]:
        return pred_json["predictions"][0]
    if "all_predictions" in pred_json and isinstance(pred_json["all_predictions"], list) and pred_json["all_predictions"]:
        return pred_json.get("prediction") or pred_json["all_predictions"][0]
    return pred_json

# ---- routes ----
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Server is running", "database": DATABASE_NAME})

@app.route("/api/iot_device", methods=["POST"])
def api_iot_device():
    """
    POST JSON:
      { "device_id": "<collection name>", "prediction_url": "<optional override>" }

    Behavior:
      - Fetch latest document for the collection named device_id (sorted by _id desc)
      - Extract image from common keys
      - Send to model API (image_url, multipart file, or json image)
      - Return device_data + model prediction
      - No timestamp processing is performed (per request)
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    device_id = payload.get("device_id")
    if not device_id:
        return jsonify({"error": "device_id is required"}), 400

    prediction_url = payload.get("prediction_url") or MODEL_API_URL

    # access collection
    try:
        coll = mongo_db[device_id]
    except Exception as e:
        logger.exception("Cannot access collection '%s': %s", device_id, str(e))
        return jsonify({"error": f"Cannot access collection '{device_id}': {str(e)}"}), 500

    # fetch most recent document (no timestamp logic)
    try:
        device_doc = coll.find_one(sort=[("_id", DESCENDING)])
    except Exception as e:
        logger.exception("Mongo find_one failed: %s", str(e))
        return jsonify({"error": "Failed to query MongoDB", "details": str(e)}), 500

    if not device_doc:
        return jsonify({"error": f"No data found for device {device_id}"}), 404

    device_doc_safe = _convert_bson(device_doc)

    # locate image in doc (top-level or one-level nested)
    image_data = None
    image_field = None
    candidates = ("image", "image_base64", "image_url", "img", "photo", "picture")

    for c in candidates:
        if c in device_doc_safe and device_doc_safe[c]:
            image_data = device_doc_safe[c]
            image_field = c
            break

    if not image_data:
        for k, v in device_doc_safe.items():
            if isinstance(v, dict):
                for c in candidates:
                    if c in v and v[c]:
                        image_data = v[c]
                        image_field = f"{k}.{c}"
                        break
            if image_data:
                break

    if not image_data:
        return jsonify({"error": "No image found in document", "device_id": device_id}), 400

    headers = {"Accept": "application/json"}
    pred_json = None
    main_prediction = None

    try:
        # if image_data is a URL
        if isinstance(image_data, str) and image_data.lower().startswith(("http://", "https://")):
            resp = prediction_session.post(prediction_url, json={"image_url": image_data}, headers=headers, timeout=MODEL_API_TIMEOUT)

        # if base64 (data URI or raw base64)
        elif isinstance(image_data, str) and (image_data.startswith("data:image/") or _looks_like_base64(image_data.split(",")[-1])):
            b64_part = image_data.split(",")[-1]
            try:
                img_bytes = base64.b64decode(b64_part)
                files = {"file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")}
                resp = prediction_session.post(prediction_url, files=files, headers=headers, timeout=MODEL_API_TIMEOUT)
            except Exception:
                # fallback to sending base64 in JSON body
                resp = prediction_session.post(prediction_url, json={"image": b64_part}, headers=headers, timeout=MODEL_API_TIMEOUT)

        # fallback: send whatever is under image as JSON
        else:
            resp = prediction_session.post(prediction_url, json={"image": image_data}, headers=headers, timeout=MODEL_API_TIMEOUT)

        resp.raise_for_status()

        try:
            pred_json = resp.json()
        except Exception:
            pred_json = {"raw_text": resp.text}

        main_prediction = _extract_main_prediction(pred_json)

    except requests.exceptions.RequestException as re:
        logger.warning("Prediction API failed for device %s: %s", device_id, str(re))

        # fallback: return saved prediction from DB if it exists
        saved = None
        for k in ("model_prediction", "model_prediction_raw", "prediction", "pred"):
            if k in device_doc_safe and device_doc_safe[k]:
                saved = device_doc_safe[k]
                break

        if saved:
            return jsonify({
                "warning": "Model API failed; returning saved prediction from DB as fallback",
                "device_id": device_id,
                "device_data": device_doc_safe,
                "model_prediction": saved
            }), 200

        return jsonify({
            "error": "Prediction API request failed",
            "details": str(re),
            "device_id": device_id,
            "device_data": device_doc_safe
        }), 502

    # success
    return jsonify({
        "device_id": device_id,
        "device_data": device_doc_safe,
        "image_field": image_field,
        "model_prediction": main_prediction,
        "prediction_api_response": pred_json
    }), 200

# ---- run app ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
