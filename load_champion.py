import sys
import numpy as np
import mlflow
import tensorflow as tf
from mlflow import MlflowClient
from dotenv import load_dotenv
load_dotenv()

if len(sys.argv) < 2:
    sys.exit("usage: python load_champion.py <path_to_image>")

NAME = "23L-2540_A2_Ops_VisionGuard"
client = MlflowClient()
mv = client.get_model_version_by_alias(NAME, "champion")
run = client.get_run(mv.run_id)
thr = run.data.metrics["threshold"]                       # threshold chosen on validation data
img_size = int(run.data.params.get("img_size", 224))

local = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_file/model.keras",
                                            dst_path="models/_download")
model = tf.keras.models.load_model(local, compile=False)

img = tf.keras.utils.load_img(sys.argv[1], target_size=(img_size, img_size)).convert("RGB")
x = np.expand_dims(tf.keras.utils.img_to_array(img), 0)   # raw 0-255; preprocessing is inside the model
p = float(model.predict(x, verbose=0)[0][0])
print(f"champion v{mv.version} (run {run.info.run_name})  P(AI-generated)={p:.4f}  threshold={thr:.2f}  ->  "
      f"{'AI-GENERATED' if p >= thr else 'REAL'}")