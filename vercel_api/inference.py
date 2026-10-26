import io
import json
import os
import threading
import time

import numpy as np
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


def preprocess(data: bytes, size: int) -> np.ndarray:
    """Same steps as training: decode to RGB, antialiased bilinear resize to a square,
    raw 0-255 float32. The backbone's own scaling is inside the model."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
    except Exception as e:
        raise ImageRejected("The image data is corrupted and could not be decoded.") from e
    return arr[None, ...]


def _load_interpreter(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf            # local development only
            Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=model_path)


class Predictor:
    """Loads the TFLite champion ONCE; predict() is then cheap and matches the training preprocessing."""

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
        self.interp = _load_interpreter(model_path)
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]["index"]
        self._out = self.interp.get_output_details()[0]["index"]
        self._lock = threading.Lock()
        self._run(np.zeros((1, self.img_size, self.img_size, 3), np.float32))   # warm-up

    def _run(self, x):
        with self._lock:
            self.interp.set_tensor(self._in, x)
            self.interp.invoke()
            return float(self.interp.get_tensor(self._out).ravel()[0])

    def confidence(self, p):
        """Margin from the decision threshold, scaled to 0.5 (on the line) .. 1.0 (certain)."""
        t = self.threshold
        if p >= t:
            return 0.5 + 0.5 * (p - t) / (1 - t)
        return 0.5 + 0.5 * (t - p) / t

    def predict(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        check_image(data)
        p = self._run(preprocess(data, self.img_size))
        return {
            "label": "AI-generated" if p >= self.threshold else "Real",
            "p_ai": p,
            "confidence": self.confidence(p),
            "threshold": self.threshold,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }