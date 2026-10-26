import base64
import hashlib
import io
import os
import time

from dotenv import load_dotenv
load_dotenv()

import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image
from sqlalchemy import func, inspect, text
from werkzeug.utils import secure_filename

from db import db, Prediction
from inference import ImageRejected, Predictor, check_image

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


class ModelServiceError(Exception):
    """The remote model API could not be used."""


def error_page(code, title, message):
    return render_template("error.html", code=code, title=title, message=message), code


def thumbnail_data_uri(data: bytes) -> str:
    """Small preview embedded in the page. Nothing is written to disk."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((320, 320))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ensure_columns():
    """Adds columns introduced after the first release (small project, no migration tool)."""
    existing = {c["name"] for c in inspect(db.engine).get_columns("predictions")}
    with db.engine.begin() as conn:
        if "source" not in existing:
            conn.execute(text("ALTER TABLE predictions ADD COLUMN source VARCHAR(20) NOT NULL DEFAULT 'local'"))


def remote_predict(url, key, data, filename, content_type):
    """Sends the image to the cloud model API; returns the same dict shape as Predictor.predict()."""
    endpoint = url.rstrip("/") + "/predict"
    response, t0 = None, time.perf_counter()
    for _ in range(2):          # one retry: a cold-starting function may drop the first connection
        try:
            t0 = time.perf_counter()
            response = requests.post(
                endpoint, headers={"X-API-Key": key},
                files={"image": (filename, data, content_type or "application/octet-stream")},
                timeout=(10, 120))
            break
        except (requests.ConnectionError, requests.Timeout):
            response = None
    if response is None:
        raise ModelServiceError("The model service did not respond. Please try again in a moment.")
    round_trip = (time.perf_counter() - t0) * 1000

    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code == 200:
        try:
            m = body.get("model", {})
            return {
                "label": body["label"], "p_ai": float(body["p_ai"]),
                "confidence": float(body["confidence"]), "threshold": float(body["threshold"]),
                "latency_ms": round_trip, "api_latency_ms": float(body.get("latency_ms", 0)),
                "model_name": m.get("name", "unknown"), "model_version": str(m.get("version", "unknown")),
                "run_name": m.get("run_name", "unknown"), "source": "cloud-api",
            }
        except (KeyError, TypeError, ValueError) as e:
            raise ModelServiceError("The model service sent an unexpected response.") from e
    if response.status_code in (401, 403):
        raise ModelServiceError("The model service rejected the API key. Check MODEL_API_KEY.")
    if response.status_code in (413, 415, 422):
        raise ImageRejected(body.get("error", "The model service rejected this image."))
    raise ModelServiceError(f"The model service returned an error (HTTP {response.status_code}).")


def create_app():
    app = Flask(__name__)
    max_mb = float(os.getenv("MAX_UPLOAD_MB", "5"))
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024)
    app.config["HISTORY_LIMIT"] = int(os.getenv("HISTORY_LIMIT", "100"))
    app.config["MODEL_API_URL"] = os.getenv("MODEL_API_URL", "").strip()
    app.config["MODEL_API_KEY"] = os.getenv("MODEL_API_KEY", "")
    remote = bool(app.config["MODEL_API_URL"])
    db.init_app(app)

    if remote:
        if not app.config["MODEL_API_KEY"]:
            raise RuntimeError("MODEL_API_URL is set but MODEL_API_KEY is empty")
        app.predictor = None                   # no local model: inference happens in the cloud
    else:
        app.predictor = Predictor(os.getenv("MODEL_PATH", "models/champion.keras"),
                                  os.getenv("MODEL_INFO_PATH", "models/champion.json"),
                                  os.getenv("THRESHOLD"))
    with app.app_context():
        db.create_all()
        ensure_columns()

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=max_mb)

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        if f is None or not f.filename:
            return error_page(400, "No file selected", "Please choose an image before pressing Analyse.")

        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return error_page(415, "Unsupported file type",
                              "Only JPG and PNG images are accepted. Please convert your file and try again.")
        filename = secure_filename(f.filename) or ("upload" + ext)   # sanitized: no paths, no odd characters

        data = f.read()                       # kept in memory only, never saved to disk
        if not data:
            return error_page(400, "Empty file", "The uploaded file is empty.")

        try:
            if remote:
                check_image(data)             # cheap local check, so bad files never use the network
                res = remote_predict(app.config["MODEL_API_URL"], app.config["MODEL_API_KEY"],
                                     data, filename, f.mimetype)
            else:
                res = app.predictor.predict(data)
                pr = app.predictor
                res.update(model_name=pr.model_name, model_version=pr.version,
                           run_name=pr.run_name, source="local")
        except ImageRejected as e:
            return error_page(422, "This is not a valid image",
                              f"{e} The file has an image extension, but its content is not a readable JPG or PNG.")
        except ModelServiceError as e:
            app.logger.error("model service error: %s", e)
            return error_page(502, "Model service unavailable", str(e))

        rec = Prediction(filename=filename[:255], file_size=len(data),
                         sha256=hashlib.sha256(data).hexdigest(),
                         label=res["label"], p_ai=res["p_ai"], confidence=res["confidence"],
                         threshold=res["threshold"], model_name=res["model_name"],
                         model_version=res["model_version"], run_name=res["run_name"],
                         latency_ms=res["latency_ms"], source=res["source"])
        saved = True
        try:
            db.session.add(rec)
            db.session.commit()
        except Exception:
            db.session.rollback()
            app.logger.exception("could not save prediction")
            saved = False

        return render_template("result.html", res=res, filename=filename, saved=saved,
                               thumb=thumbnail_data_uri(data))

    @app.get("/history")
    def history():
        rows = db.session.scalars(
            db.select(Prediction).order_by(Prediction.id.desc()).limit(app.config["HISTORY_LIMIT"])).all()
        counts = dict(db.session.execute(
            db.select(Prediction.label, func.count()).group_by(Prediction.label)).all())
        return render_template("history.html", rows=rows, counts=counts,
                               total=sum(counts.values()), limit=app.config["HISTORY_LIMIT"])

    @app.get("/health")
    def health():
        db_ok = True
        try:
            db.session.execute(text("SELECT 1"))
        except Exception:
            db_ok = False
        body = {"status": "ok" if db_ok else "degraded",
                "mode": "cloud-api" if remote else "local-model",
                "database": "ok" if db_ok else "unreachable"}
        if remote:
            body["model_api"] = app.config["MODEL_API_URL"]
        else:
            pr = app.predictor
            body.update(model_loaded=pr.model is not None, model=f"{pr.model_name} v{pr.version}",
                        threshold=pr.threshold)
        return jsonify(body), (200 if db_ok else 503)

    @app.errorhandler(404)
    def not_found(_):
        return error_page(404, "Page not found", "That page does not exist.")

    @app.errorhandler(413)
    def too_large(_):
        return error_page(413, "File too large", f"Images must be smaller than {max_mb:g} MB.")

    @app.errorhandler(500)
    def server_error(_):
        return error_page(500, "Something went wrong", "An unexpected error occurred. Please try again.")

    return app


app = create_app()

if __name__ == "__main__":
    # Flask's own server, debug OFF
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)