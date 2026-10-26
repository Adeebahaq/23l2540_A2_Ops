from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Prediction(db.Model):
    """One row per request: what was decided, when, by which model, how confidently."""
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, index=True,
                           default=lambda: datetime.now(timezone.utc))
    filename = db.Column(db.String(255), nullable=False)      # sanitized name; the image itself is NOT stored
    file_size = db.Column(db.Integer, nullable=False)
    sha256 = db.Column(db.String(64), nullable=False)
    label = db.Column(db.String(20), nullable=False)          # decision
    p_ai = db.Column(db.Float, nullable=False)                # raw model probability of AI-generated
    confidence = db.Column(db.Float, nullable=False)          # margin-based confidence, 0.5-1.0
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    model_version = db.Column(db.String(20), nullable=False)
    run_name = db.Column(db.String(100), nullable=False)
    latency_ms = db.Column(db.Float, nullable=False)