import io, json, os, threading, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageRejected(Exception):
    """The bytes are not a readable JPEG/PNG image."""


def check_image(data: bytes) -> str:
    """Rejects files that merely *look* like images (right extension, wrong or broken content)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except Exception as e:
        raise ImageRejected("The file could not be read as an image.") from e
    if fmt not in ALLOWED_FORMATS:
        raise ImageRejected(f"Unsupported image format: {fmt}.")
    return fmt


class Predictor:
    """Loads the champion model ONCE; predict() is then cheap and uses the training preprocessing."""

    def __init__(self, model_path, info_path, threshold=None):
        info = {}
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
        self.threshold = float(threshold if threshold not in (None, "") else info.get("threshold", 0.5))
        self.img_size = int(info.get("img_size", 224))
        self.model_name = info.get("model_name", "VisionGuard")
        self.version = str(info.get("version", "unknown"))
        self.run_name = info.get("run_name", "unknown")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self._lock = threading.Lock()
        # warm-up call so the first user request is not slow
        self.model(np.zeros((1, self.img_size, self.img_size, 3), "float32"), training=False)

    def preprocess(self, data: bytes):
        """Identical to pipeline.py: decode to 3-channel RGB, antialiased resize to a square,
        keep raw 0-255 floats. Model-specific scaling is already inside the model."""
        try:
            img = tf.io.decode_image(data, channels=3, expand_animations=False)
            img = tf.image.resize(img, (self.img_size, self.img_size), antialias=True)
        except tf.errors.InvalidArgumentError as e:
            raise ImageRejected("The image data is corrupted and could not be decoded.") from e
        return tf.expand_dims(tf.cast(img, tf.float32), 0)

    def confidence(self, p):
        """Margin from the decision threshold, scaled to 0.5 (on the line) .. 1.0 (certain).
        Plain max(p, 1-p) would be misleading because the threshold is not 0.5."""
        t = self.threshold
        if p >= t:
            return 0.5 + 0.5 * (p - t) / (1 - t)
        return 0.5 + 0.5 * (t - p) / t

    def predict(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        check_image(data)
        x = self.preprocess(data)
        with self._lock:
            p = float(self.model(x, training=False).numpy().ravel()[0])
        return {
            "label": "AI-generated" if p >= self.threshold else "Real",
            "p_ai": p,
            "confidence": self.confidence(p),
            "threshold": self.threshold,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    pr = Predictor(os.getenv("MODEL_PATH", "models/champion.keras"),
                   os.getenv("MODEL_INFO_PATH", "models/champion.json"), os.getenv("THRESHOLD"))
    print(f"model {pr.model_name} v{pr.version} ({pr.run_name}), img_size={pr.img_size}, threshold={pr.threshold:.2f}")
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            r = pr.predict(f.read())
        print(f"{path}\n  -> {r['label']}  P(AI)={r['p_ai']:.4f}  confidence={r['confidence']:.2f}  ({r['latency_ms']:.0f} ms)")