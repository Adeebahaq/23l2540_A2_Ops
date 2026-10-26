import base64, hashlib, io, os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, render_template, request
from PIL import Image
from sqlalchemy import func, text
from werkzeug.utils import secure_filename

from db import db, Prediction
from inference import ImageRejected, Predictor

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


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


def create_app():
    app = Flask(__name__)
    max_mb = float(os.getenv("MAX_UPLOAD_MB", "5"))
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024)
    app.config["HISTORY_LIMIT"] = int(os.getenv("HISTORY_LIMIT", "100"))
    db.init_app(app)

    # model loaded once at start-up, shared by every request
    app.predictor = Predictor(os.getenv("MODEL_PATH", "models/champion.keras"),
                              os.getenv("MODEL_INFO_PATH", "models/champion.json"),
                              os.getenv("THRESHOLD"))
    with app.app_context():
        db.create_all()

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
            res = app.predictor.predict(data)
        except ImageRejected as e:
            return error_page(422, "This is not a valid image",
                              f"{e} The file has an image extension, but its content is not a readable JPG or PNG.")

        pr = app.predictor
        rec = Prediction(filename=filename[:255], file_size=len(data),
                         sha256=hashlib.sha256(data).hexdigest(),
                         label=res["label"], p_ai=res["p_ai"], confidence=res["confidence"],
                         threshold=res["threshold"], model_name=pr.model_name,
                         model_version=pr.version, run_name=pr.run_name, latency_ms=res["latency_ms"])
        saved = True
        try:
            db.session.add(rec)
            db.session.commit()
        except Exception:
            db.session.rollback()
            app.logger.exception("could not save prediction")
            saved = False

        return render_template("result.html", res=res, rec=rec, filename=filename, saved=saved,
                               thumb=thumbnail_data_uri(data), pr=pr)

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
        pr = app.predictor
        body = {"status": "ok" if db_ok else "degraded", "model_loaded": pr.model is not None,
                "model": f"{pr.model_name} v{pr.version}", "threshold": pr.threshold,
                "database": "ok" if db_ok else "unreachable"}
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
    # Flask's own server, debug OFF (see report notes)
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)