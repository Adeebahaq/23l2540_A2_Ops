import json, os, shutil
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
load_dotenv()

NAME = "23L-2540_A2_Ops_VisionGuard"
client = MlflowClient()
mv = client.get_model_version_by_alias(NAME, "champion")
run = client.get_run(mv.run_id)

os.makedirs("models", exist_ok=True)
local = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_file/model.keras",
                                            dst_path="models/_download")
shutil.copy(local, "models/champion.keras")

info = {
    "model_name": NAME,
    "version": mv.version,
    "run_name": run.info.run_name,
    "run_id": mv.run_id,
    "threshold": run.data.metrics["threshold"],
    "img_size": int(run.data.params.get("img_size", 224)),
    "label_map": {"0": "Real", "1": "AI-generated"},
}
with open("models/champion.json", "w") as f:
    json.dump(info, f, indent=2)
print("saved models/champion.keras and models/champion.json")
print(info)