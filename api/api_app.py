import hmac
import logging
import os
import uuid

from flask import Flask, jsonify, request

from inference import ImageRejected, Predictor

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
MAX_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    raise RuntimeError("API_KEY is not set; refusing to start an unprotected API.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_MB * 1024 * 1024)

# send Flask's log lines to the same place as gunicorn's (Cloud Run reads stdout/stderr)
gunicorn_log = logging.getLogger("gunicorn.error")
if gunicorn_log.handlers:
    app.logger.handlers = gunicorn_log.handlers
    app.logger.setLevel(gunicorn_log.level)
else:
    app.logger.setLevel(logging.INFO)

# the model is loaded ONCE per container start, never per request
predictor = Predictor(os.getenv("MODEL_PATH", "models/champion.keras"),
                      os.getenv("MODEL_INFO_PATH", "models/champion.json"),
                      os.getenv("THRESHOLD"))
app.logger.info("model loaded: %s v%s (%s) threshold=%.2f",
                predictor.model_name, predictor.version, predictor.run_name, predictor.threshold)


def err(status, message):
    return jsonify({"error": message, "status": status}), status


@app.get("/")
def root():
    return jsonify({
        "service": "VisionGuard model API",
        "model": f"{predictor.model_name} v{predictor.version}",
        "endpoints": {"GET /health": "liveness, public",
                      "POST /predict": "multipart field 'image', header X-API-Key"},
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": predictor.model_name, "version": predictor.version,
                    "run_name": predictor.run_name, "threshold": predictor.threshold,
                    "img_size": predictor.img_size})


@app.post("/predict")
def predict():
    rid = uuid.uuid4().hex[:12]
    sent = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(sent.encode(), API_KEY.encode()):   # constant-time comparison
        app.logger.warning("rid=%s status=401 missing or wrong API key", rid)
        return err(401, "Missing or invalid API key.")

    f = request.files.get("image")
    if f is None or not f.filename:
        return err(400, "Send the image as multipart/form-data in a field named 'image'.")
    name = os.path.basename(f.filename)[:100]
    if os.path.splitext(name)[1].lower() not in ALLOWED_EXT:
        app.logger.info("rid=%s status=415 file=%s", rid, name)
        return err(415, "Only JPG and PNG images are accepted.")
    data = f.read()
    if not data:
        return err(400, "The uploaded file is empty.")

    try:
        res = predictor.predict(data)
    except ImageRejected as e:
        app.logger.info("rid=%s status=422 file=%s reason=%s", rid, name, e)
        return err(422, str(e))

    app.logger.info("rid=%s status=200 file=%s bytes=%d label=%s p_ai=%.4f confidence=%.2f latency_ms=%.0f",
                    rid, name, len(data), res["label"], res["p_ai"], res["confidence"], res["latency_ms"])
    return jsonify({
        "request_id": rid,
        "filename": name,
        "label": res["label"],
        "p_ai": round(res["p_ai"], 6),
        "confidence": round(res["confidence"], 4),
        "threshold": round(res["threshold"], 4),
        "latency_ms": round(res["latency_ms"], 1),
        "model": {"name": predictor.model_name, "version": predictor.version,
                  "run_name": predictor.run_name},
    })


@app.errorhandler(404)
def not_found(_):
    return err(404, "Not found.")


@app.errorhandler(405)
def bad_method(_):
    return err(405, "Method not allowed.")


@app.errorhandler(413)
def too_large(_):
    return err(413, f"Image larger than {MAX_MB:g} MB.")


@app.errorhandler(500)
def server_error(_):
    return err(500, "Unexpected server error.")