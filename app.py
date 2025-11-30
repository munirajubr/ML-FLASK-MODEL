import os
import io
import base64
import json
import logging
import pickle
import requests
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- load env ----
load_dotenv()

# ---- config (from .env) ----
MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME") or os.getenv("MONGO_DBNAME") or "AgriVision_IoT"

BACKEND_DEVICES_URL = os.getenv("BACKEND_DEVICES_URL", "https://agrivision-mobile-app-using-react-native.onrender.com")
MODEL_IOT_API_URL = os.getenv("MODEL_IOT_API_URL")
MODEL_API_URL = os.getenv("MODEL_API_URL") or os.getenv("PREDICTION_API_URL")
MODEL_IOT_CALL_URL = MODEL_IOT_API_URL or MODEL_API_URL

PORT = int(os.getenv("PORT", "10000"))

MODEL_API_TIMEOUT = float(os.getenv("MODEL_API_TIMEOUT", "60"))
MODEL_API_MAX_RETRIES = int(os.getenv("MODEL_API_MAX_RETRIES", "3"))
MODEL_API_BACKOFF_FACTOR = float(os.getenv("MODEL_API_BACKOFF_FACTOR", "1.0"))

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL_API_KEY_HEADER = os.getenv("MODEL_API_KEY_HEADER", "Authorization")

# ---- logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Flask app ----
app = Flask(__name__)
CORS(app)

# ---- Mongo client (optional) ----
mongo_db = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client[DATABASE_NAME]
        logger.info("Connected to MongoDB: %s", DATABASE_NAME)
    except Exception as e:
        logger.exception("Failed to connect to MongoDB: %s", e)
else:
    logger.warning("MONGO_URI not provided; IoT endpoints that use MongoDB will fail until set.")

# ---- requests session with retry ----
def create_retry_session(retries=3, backoff_factor=1.0, status_forcelist=(500,502,503,504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET","POST","PUT","DELETE","HEAD","OPTIONS"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

prediction_session = create_retry_session(retries=MODEL_API_MAX_RETRIES, backoff_factor=MODEL_API_BACKOFF_FACTOR)

# ---- helpers ----
def _convert_bson(o):
    if isinstance(o, ObjectId):
        return str(o)
    if isinstance(o, dict):
        return {k: _convert_bson(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_convert_bson(i) for i in o]
    return o

def _looks_like_base64(s: str) -> bool:
    if not isinstance(s, str) or len(s) < 20:
        return False
    try:
        candidate = s.split(",")[-1]
        base64.b64decode(candidate[:100], validate=True)
        return True
    except Exception:
        return False

def _extract_main_prediction(pred_json):
    if not isinstance(pred_json, dict):
        return pred_json
    keys_try = ("main_prediction","main_pred","prediction","pred","label","result","class")
    for k in keys_try:
        if k in pred_json and pred_json[k] not in (None, {}):
            return pred_json[k]
    for v in pred_json.values():
        if isinstance(v, dict):
            for k in keys_try:
                if k in v and v[k] not in (None, {}):
                    return v[k]
    if "predictions" in pred_json and isinstance(pred_json["predictions"], list) and pred_json["predictions"]:
        return pred_json["predictions"][0]
    if "all_predictions" in pred_json and isinstance(pred_json["all_predictions"], list) and pred_json["all_predictions"]:
        return pred_json.get("prediction") or pred_json["all_predictions"][0]
    return pred_json

def _build_model_headers():
    headers = {"Accept": "application/json"}
    if MODEL_API_KEY:
        headers[MODEL_API_KEY_HEADER] = MODEL_API_KEY
    return headers

def _pick_image_from_device_data(device_data):
    if not isinstance(device_data, dict):
        return None
    candidates = ("image", "image_base64", "image_url", "img", "photo", "picture", "saved_file")
    for k in candidates:
        if k in device_data and device_data[k]:
            return device_data[k]
    for k, v in device_data.items():
        if isinstance(v, dict):
            for c in candidates:
                if c in v and v[c]:
                    return v[c]
    return None

def _extract_prediction_from_raw(pred_raw):
    if not pred_raw:
        return None
    if isinstance(pred_raw, dict):
        if "prediction" in pred_raw and isinstance(pred_raw["prediction"], dict):
            return pred_raw["prediction"]
        if "class" in pred_raw and "confidence" in pred_raw:
            return {
                "class": pred_raw.get("class"),
                "confidence": pred_raw.get("confidence"),
                "confidence_percentage": pred_raw.get("confidence_percentage")
            }
        if "all_predictions" in pred_raw and isinstance(pred_raw["all_predictions"], list) and pred_raw["all_predictions"]:
            first = pred_raw["all_predictions"][0]
            if isinstance(first, dict):
                return first
        for v in pred_raw.values():
            if isinstance(v, dict):
                if "class" in v and "confidence" in v:
                    return {"class": v.get("class"), "confidence": v.get("confidence")}
                if "prediction" in v and isinstance(v["prediction"], dict):
                    return v["prediction"]
    if isinstance(pred_raw, list) and pred_raw:
        if isinstance(pred_raw[0], dict):
            return pred_raw[0]
    return None

def _extract_device_ids_from_response_json(j):
    if isinstance(j, list):
        return j
    if not isinstance(j, dict):
        return None
    if "devices" in j and isinstance(j["devices"], list):
        return j["devices"]
    for k in ("data", "result", "payload"):
        if k in j and isinstance(j[k], dict) and "devices" in j[k]:
            if isinstance(j[k]["devices"], list):
                return j[k]["devices"]
    for k, v in j.items():
        if isinstance(v, list) and all(isinstance(x, (str,int)) for x in v):
            return v
    return None

def _build_backend_candidate_urls(base):
    base = base.rstrip("/")
    candidates = [base, base + "/api/devices/get", base + "/api/devices", base + "/devices/get", base + "/devices"]
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _safe_str_from_bytes(maybe_bytes):
    if maybe_bytes is None:
        return None
    if isinstance(maybe_bytes, str):
        return maybe_bytes
    if isinstance(maybe_bytes, bytes):
        try:
            return maybe_bytes.decode("utf-8")
        except Exception:
            return "base64:" + base64.b64encode(maybe_bytes).decode("ascii")
    try:
        return str(maybe_bytes)
    except Exception:
        return None

def _make_json_safe(obj):
    # primitives ok
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # bytes -> base64
    if isinstance(obj, bytes):
        return "base64:" + base64.b64encode(obj).decode("ascii")

    # numpy scalar numbers
    try:
        if isinstance(obj, np.generic):
            if np.issubdtype(type(obj), np.integer):
                return int(obj)
            if np.issubdtype(type(obj), np.floating):
                return float(obj)
            if np.issubdtype(type(obj), np.bool_):
                return bool(obj)
            try:
                return obj.item()
            except Exception:
                return str(obj)
    except Exception:
        pass

    # numpy arrays -> lists
    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            return [_make_json_safe(x) for x in obj]

    # ObjectId -> str
    if isinstance(obj, ObjectId):
        return str(obj)

    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k)
            out[key] = _make_json_safe(v)
        return out

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(i) for i in obj]

    # requests.Response-like
    try:
        import requests as _requests
        if isinstance(obj, _requests.Response):
            safe = {
                "status_code": getattr(obj, "status_code", None),
                "url": _safe_str_from_bytes(getattr(obj, "url", None)),
                "text_snippet": _safe_str_from_bytes(getattr(obj, "text", None))[:800] if getattr(obj, "text", None) else None,
                "headers": dict(getattr(obj, "headers", {}) or {})
            }
            return _make_json_safe(safe)
    except Exception:
        pass

    # other objects -> try convert to str
    try:
        return str(obj)
    except Exception:
        try:
            return json.loads(json.dumps(obj))
        except Exception:
            return None

def _fetch_device_ids_robust(username, backend_base, timeout):
    debug = {"attempts": []}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    candidates = _build_backend_candidate_urls(backend_base)

    for url in candidates:
        # GET ?username=
        try:
            resp = prediction_session.get(url, params={"username": username}, headers=headers, timeout=timeout)
            s = getattr(resp, "status_code", None)
            text_snippet = _safe_str_from_bytes(getattr(resp, "text", None))
            debug["attempts"].append({"method": "GET?username", "url": _safe_str_from_bytes(getattr(resp.request, "url", url)), "status": s, "text_snippet": text_snippet})
            if s and 200 <= s < 300:
                try:
                    j = resp.json()
                except Exception:
                    j = {"raw_text": text_snippet}
                ids = _extract_device_ids_from_response_json(j)
                if ids:
                    return ids, debug
        except requests.RequestException as e:
            debug["attempts"].append({"method": "GET?username", "url": url, "error": _safe_str_from_bytes(str(e))})

        # GET with JSON body (some clients used this)
        try:
            resp = prediction_session.get(url, json={"username": username}, headers=headers, timeout=timeout)
            s = getattr(resp, "status_code", None)
            req_body = _safe_str_from_bytes(getattr(resp.request, "body", None))
            text_snippet = _safe_str_from_bytes(getattr(resp, "text", None))
            debug["attempts"].append({"method": "GET(json)", "url": _safe_str_from_bytes(getattr(resp.request, "url", url)), "status": s, "req_body": req_body, "text_snippet": text_snippet})
            if s and 200 <= s < 300:
                try:
                    j = resp.json()
                except Exception:
                    j = {"raw_text": text_snippet}
                ids = _extract_device_ids_from_response_json(j)
                if ids:
                    return ids, debug
        except requests.RequestException as e:
            debug["attempts"].append({"method": "GET(json)", "url": url, "error": _safe_str_from_bytes(str(e))})

        # POST JSON
        try:
            resp = prediction_session.post(url, json={"username": username}, headers=headers, timeout=timeout)
            s = getattr(resp, "status_code", None)
            req_body = _safe_str_from_bytes(getattr(resp.request, "body", None))
            text_snippet = _safe_str_from_bytes(getattr(resp, "text", None))
            debug["attempts"].append({"method": "POST", "url": _safe_str_from_bytes(getattr(resp.request, "url", url)), "status": s, "req_body": req_body, "text_snippet": text_snippet})
            if s and 200 <= s < 300:
                try:
                    j = resp.json()
                except Exception:
                    j = {"raw_text": text_snippet}
                ids = _extract_device_ids_from_response_json(j)
                if ids:
                    return ids, debug
        except requests.RequestException as e:
            debug["attempts"].append({"method": "POST", "url": url, "error": _safe_str_from_bytes(str(e))})

    return None, debug

# ---- Load local ML model & scaler and CSVs ----
crop_model = None
scaler = None
crop_data = None
disease_data = None

# try load model files if present
try:
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            crop_model = pickle.load(f)
            logger.info("Loaded crop model from model.pkl")
    else:
        logger.warning("model.pkl not found; /api/predict will be unavailable locally.")

    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            logger.info("Loaded scaler from scaler.pkl")
    else:
        logger.warning("scaler.pkl not found; inputs won't be scaled if missing.")
except Exception as e:
    logger.exception("Failed to load model/scaler: %s", e)
    crop_model = None
    scaler = None

# Try load CSV datasets for info endpoints
try:
    if os.path.exists("Dataset/eggplant_diseases.csv"):
        disease_data = pd.read_csv("Dataset/eggplant_diseases.csv", encoding="latin1")
        logger.info("Loaded disease dataset")
    else:
        logger.warning("Dataset/eggplant_diseases.csv not found.")

    if os.path.exists("Dataset/eggplant_details.csv"):
        crop_data = pd.read_csv("Dataset/eggplant_details.csv", encoding="latin1")
        logger.info("Loaded crop details dataset")
    else:
        logger.warning("Dataset/eggplant_details.csv not found.")
except Exception as e:
    logger.exception("Failed to load CSV datasets: %s", e)

def get_disease_info(disease_name: str):
    if disease_data is None:
        return None
    result = disease_data[disease_data.iloc[:,0].astype(str).str.lower() == disease_name.strip().lower()]
    if result.empty:
        return None
    return result.to_dict(orient="records")[0]

def get_crop_info(crop_name: str):
    if crop_data is None:
        return None
    crop_name_clean = crop_name.strip().lower()
    # attempt to find in a column that contains "Crop" or "Name"
    cols = [c for c in crop_data.columns if "crop" in c.lower() or "name" in c.lower()]
    if cols:
        col = cols[0]
        result = crop_data[crop_data[col].astype(str).str.lower().str.strip() == crop_name_clean]
    else:
        # fallback search across all string columns
        mask = None
        for c in crop_data.select_dtypes(include="object").columns:
            if mask is None:
                mask = crop_data[c].astype(str).str.lower().str.strip() == crop_name_clean
            else:
                mask = mask | (crop_data[c].astype(str).str.lower().str.strip() == crop_name_clean)
        result = crop_data[mask] if mask is not None else pd.DataFrame()
    if result.empty:
        return None
    return result.to_dict(orient="records")[0]

# ---- Routes ----
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Server is running"})

@app.route("/api/devices", methods=["POST"])
def api_devices():
    """
    Returns:
    {
      "username": "...",
      "devices": {
         "<device_id>": [ {record1}, {record2}, ... ],
         ...
      }
    }
    Each device_id maps to a list of records (useful if you want to show multiple entries per device).
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    username = data.get("username")
    if not username:
        return jsonify({"error": "username is required"}), 400

    backend_url = data.get("backend_devices_url") or BACKEND_DEVICES_URL
    model_url = data.get("model_api_url") or MODEL_IOT_CALL_URL
    headers = _build_model_headers()

    device_ids, _debug = _fetch_device_ids_robust(username, backend_url, MODEL_API_TIMEOUT)
    if not device_ids:
        # return empty mapping if none
        return jsonify(_make_json_safe({"username": username, "devices": {}})), 200

    devices_map = {}  # device_id -> list of records

    for device_id in device_ids:
        # default clean record structure (omit device_id inside record - device_id is already the key)
        record = {
            "soil_moisture": None,
            "temperature": None,
            "humidity": None,
            "image": None,
            "created_at": None,
            "prediction": None
        }

        model_json = None
        # try model API
        if model_url:
            try:
                resp = prediction_session.post(model_url, json={"device_id": device_id}, headers=headers, timeout=MODEL_API_TIMEOUT)
                resp.raise_for_status()
                model_json = resp.json()
            except Exception:
                model_json = None

        # prefer device_data from model response
        device_data = None
        if isinstance(model_json, dict):
            device_data = model_json.get("device_data") or model_json

        # fallback to MongoDB if no device_data from model
        if not device_data and mongo_db is not None:
            try:
                coll = mongo_db[device_id]
                doc = coll.find_one(sort=[("_id", DESCENDING)])
                if doc:
                    device_data = _convert_bson(doc)
            except Exception:
                device_data = None

        # if still no data, append empty record
        if not device_data or not isinstance(device_data, dict):
            # ensure key exists and append the empty record
            devices_map.setdefault(device_id, []).append(record)
            continue

        # extract fields
        record["soil_moisture"] = (
            device_data.get("soil_moisture")
            or device_data.get("soil")
            or device_data.get("moisture")
        )
        record["temperature"] = device_data.get("temperature") or device_data.get("temp")
        record["humidity"] = device_data.get("humidity") or device_data.get("humid")
        record["created_at"] = (
            device_data.get("created_at")
            or device_data.get("predicted_at")
            or device_data.get("_id")
        )
        record["image"] = _pick_image_from_device_data(device_data)

        pred_raw = (
            device_data.get("model_prediction_raw")
            or device_data.get("model_prediction")
            or device_data.get("prediction")
        )
        record["prediction"] = _extract_prediction_from_raw(pred_raw)

        devices_map.setdefault(device_id, []).append(record)

    response = {"username": username, "devices": devices_map}
    return jsonify(_make_json_safe(response)), 200

# IoT device endpoint (from your earlier code)
@app.route("/api/iot_device", methods=["POST"])
def api_iot_device():
    if mongo_db is None:
        return jsonify({"error": "MongoDB not configured on server"}), 500

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    device_id = payload.get("device_id")
    if not device_id:
        return jsonify({"error": "device_id is required"}), 400

    prediction_url = payload.get("prediction_url") or MODEL_API_URL
    if not prediction_url:
        return jsonify({"error": "MODEL_API_URL not configured and no prediction_url provided"}), 400

    try:
        coll = mongo_db[device_id]
    except Exception as e:
        logger.exception("Cannot access collection '%s': %s", device_id, e)
        return jsonify({"error": f"Cannot access collection '{device_id}': {str(e)}"}), 500

    try:
        device_doc = coll.find_one(sort=[("_id", DESCENDING)])
    except Exception as e:
        logger.exception("Mongo find_one failed: %s", e)
        return jsonify({"error": "Failed to query MongoDB", "details": str(e)}), 500

    if not device_doc:
        return jsonify({"error": f"No data found for device {device_id}"}), 404

    device_doc_safe = _convert_bson(device_doc)

    # find image
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

    headers = _build_model_headers()
    pred_json = None
    main_prediction = None

    try:
        # URL
        if isinstance(image_data, str) and image_data.lower().startswith(("http://","https://")):
            resp = prediction_session.post(prediction_url, json={"image_url": image_data}, headers=headers, timeout=MODEL_API_TIMEOUT)
        # base64 / data uri
        elif isinstance(image_data, str) and (image_data.startswith("data:image/") or _looks_like_base64(image_data)):
            b64_part = image_data.split(",")[-1]
            try:
                img_bytes = base64.b64decode(b64_part)
                files = {"file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")}
                resp = prediction_session.post(prediction_url, files=files, headers=headers, timeout=MODEL_API_TIMEOUT)
            except Exception:
                resp = prediction_session.post(prediction_url, json={"image": b64_part}, headers=headers, timeout=MODEL_API_TIMEOUT)
        else:
            resp = prediction_session.post(prediction_url, json={"image": image_data}, headers=headers, timeout=MODEL_API_TIMEOUT)

        resp.raise_for_status()
        try:
            pred_json = resp.json()
        except Exception:
            pred_json = {"raw_text": resp.text}
        main_prediction = _extract_main_prediction(pred_json)
    except requests.exceptions.RequestException as re:
        logger.warning("Prediction API failed for device %s: %s", device_id, re)
        # fallback to saved prediction in DB
        saved = None
        for k in ("model_prediction","model_prediction_raw","prediction","pred"):
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

    return jsonify({
        "device_id": device_id,
        "device_data": device_doc_safe,
        "image_field": image_field,
        "model_prediction": main_prediction,
        "prediction_api_response": pred_json
    }), 200

# Crop prediction endpoint (local model)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    required = ["N","P","K","temperature","humidity","ph","rainfall"]
    try:
        inputs = []
        for k in required:
            if k not in data:
                return jsonify({"error": f"Missing parameter: {k}"}), 400
            inputs.append(float(data[k]))
    except Exception as e:
        return jsonify({"error": "Invalid input types; all numeric expected", "details": str(e)}), 400

    if crop_model is None:
        # optionally forward to external MODEL_API_URL if configured
        if MODEL_API_URL:
            headers = _build_model_headers()
            try:
                resp = prediction_session.post(MODEL_API_URL, json=data, headers=headers, timeout=MODEL_API_TIMEOUT)
                resp.raise_for_status()
                return jsonify(resp.json()), 200
            except Exception as e:
                logger.exception("Local model missing and external MODEL_API_URL request failed: %s", e)
                return jsonify({"error": "Local model not available and external MODEL_API_URL request failed", "details": str(e)}), 502
        return jsonify({"error": "Local crop model not loaded"}), 500

    # scaling if scaler available
    try:
        arr = np.array([inputs], dtype=float)
        if scaler is not None:
            arr = scaler.transform(arr)
    except Exception as e:
        logger.exception("Scaling error: %s", e)
        return jsonify({"error": "Failed to scale inputs", "details": str(e)}), 500

    try:
        # prefer predict_proba for top-k if available
        if hasattr(crop_model, "predict_proba"):
            probs = crop_model.predict_proba(arr)[0]
            top_k = min(3, len(probs))
            top_idx = np.argsort(probs)[-top_k:][::-1]
            top_labels = [str(crop_model.classes_[i]) for i in top_idx]
            return jsonify({"predictions": top_labels}), 200
        else:
            pred = crop_model.predict(arr)
            return jsonify({"predictions": [str(p) for p in pred]}), 200
    except Exception as e:
        logger.exception("Model prediction failed: %s", e)
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

# Disease info endpoint (lookup CSV)
@app.route("/api/disease", methods=["POST"])
def api_disease():
    data = request.get_json(silent=True)
    disease_name = data.get("disease_name") if data else None
    if not disease_name:
        return jsonify({"error": "No disease_name provided"}), 400
    info = get_disease_info(disease_name)
    if info is None:
        return jsonify({"error": f"No records found for: {disease_name}"}), 404
    return jsonify({"info": info}), 200

# Crop info endpoint (lookup CSV)
@app.route("/api/crop", methods=["POST"])
def api_crop():
    data = request.get_json(silent=True)
    crop_name = data.get("crop_name") if data else None
    if not crop_name:
        return jsonify({"error": "No crop_name provided"}), 400
    info = get_crop_info(crop_name)
    if info is None:
        return jsonify({"error": f"No records found for: {crop_name}"}), 404
    return jsonify({"info": info}), 200

# utility: list collections (debug)
@app.route("/api/collections", methods=["GET"])
def list_collections():
    if mongo_db is None:
        return jsonify({"error": "MongoDB not configured"}), 500
    try:
        cols = mongo_db.list_collection_names()
        return jsonify({"collections": cols}), 200
    except Exception as e:
        logger.exception("Failed to list collections: %s", e)
        return jsonify({"error": "Failed to list collections", "details": str(e)}), 500

# varify if IoT device data exists
@app.route("/api/iot_device_exists", methods=["POST"])
def api_iot_device_exists():
    if mongo_db is None:
        return jsonify({"error": "MongoDB not configured on server"}), 500

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    device_id = payload.get("device_id")
    if not device_id:
        return jsonify({"error": "device_id is required"}), 400

    try:
        coll = mongo_db[device_id]
    except Exception as e:
        logger.exception("Cannot access collection '%s': %s", device_id, e)
        return jsonify({"error": f"Cannot access collection '{device_id}': {str(e)}"}), 500

    try:
        doc = coll.find_one()
    except Exception as e:
        logger.exception("Mongo find_one failed: %s", e)
        return jsonify({"error": "Failed to query MongoDB", "details": str(e)}), 500

    if not doc:
        return jsonify({"exists": False, "device_id": device_id}), 404

    return jsonify({"exists": True, "device_id": device_id}), 200

# ---- run ----
if __name__ == "__main__":
    logger.info("Starting Flask server on port %s", PORT)
    app.run(host="0.0.0.0", port=PORT)
