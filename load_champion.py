import sys
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

# exactly the training pipeline: decode to RGB, antialiased bilinear resize, raw 0-255 floats
img = tf.io.decode_image(tf.io.read_file(sys.argv[1]), channels=3, expand_animations=False)
img = tf.image.resize(img, (img_size, img_size), antialias=True)
x = tf.expand_dims(tf.cast(img, tf.float32), 0)
p = float(model(x, training=False).numpy().ravel()[0])
print(f"champion v{mv.version} (run {run.info.run_name})  P(AI-generated)={p:.4f}  threshold={thr:.2f}  ->  "
      f"{'AI-GENERATED' if p >= thr else 'REAL'}")