import csv
import importlib.util
import json
import os
import random
import shutil
import sys

import numpy as np
import tensorflow as tf

SRC, INFO = "models/champion.keras", "models/champion.json"
OUT = "vercel_api/model"

for f in (SRC, INFO, "vercel_api/inference.py"):
    if not os.path.exists(f):
        sys.exit(f"missing {f}  (run python fetch_champion.py, and create vercel_api/inference.py)")

# load the exact preprocessing code the cloud API will run
spec = importlib.util.spec_from_file_location("inference_lite", "vercel_api/inference.py")
lite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lite)

info = json.load(open(INFO, encoding="utf-8"))
size, thr = int(info["img_size"]), float(info["threshold"])

model = tf.keras.models.load_model(SRC, compile=False)
try:
    model.export("tmp_saved_model")                       # Keras 3 route: SavedModel, then TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("tmp_saved_model")
except Exception as e:
    print("export route failed, using from_keras_model:", e)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
blob = converter.convert()

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/champion.tflite", "wb") as f:
    f.write(blob)
shutil.copy(INFO, f"{OUT}/champion.json")
shutil.rmtree("tmp_saved_model", ignore_errors=True)
print(f"\nTFLite model: {len(blob) / 1e6:.1f} MB   (Keras file: {os.path.getsize(SRC) / 1e6:.1f} MB)")

# ---- parity check: Keras + tf.image pipeline  vs  TFLite + Pillow pipeline ----
interp = tf.lite.Interpreter(model_content=blob)
interp.allocate_tensors()
i_in = interp.get_input_details()[0]["index"]
i_out = interp.get_output_details()[0]["index"]

rows = list(csv.DictReader(open("reports/splits/test.csv", encoding="utf-8")))
random.seed(1)
sample = []
for lab in ("0", "1"):
    sample += random.sample([r for r in rows if r["label"] == lab], 20)

diffs, flips, skipped = [], 0, 0
for r in sample:
    raw = open(r["path"], "rb").read()
    try:
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        x_pillow = lite.preprocess(raw, size)
    except Exception:
        skipped += 1
        continue
    img = tf.image.resize(img, (size, size), antialias=True)
    p_ref = float(model(tf.expand_dims(tf.cast(img, tf.float32), 0), training=False).numpy().ravel()[0])
    interp.set_tensor(i_in, x_pillow)
    interp.invoke()
    p_lite = float(interp.get_tensor(i_out).ravel()[0])
    diffs.append(abs(p_ref - p_lite))
    flips += (p_ref >= thr) != (p_lite >= thr)

print(f"\nimages compared: {len(diffs)} (skipped {skipped})")
print(f"max  |p_keras - p_tflite| = {max(diffs):.4f}")
print(f"mean |p_keras - p_tflite| = {float(np.mean(diffs)):.4f}")
print(f"decision flips at threshold {thr:.2f}: {flips}")