import argparse
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
load_dotenv()

ap = argparse.ArgumentParser()
ap.add_argument("--champion", default="final_test", help="run name of the final chosen run")
a = ap.parse_args()

NAME = "23L-2540_A2_Ops_VisionGuard"
client = MlflowClient()
exp = client.get_experiment_by_name("23L-2540_A2_Ops")
runs = client.search_runs(
    [exp.experiment_id],
    filter_string="tags.logging_mode = 'manual' and attributes.status = 'FINISHED'",
    order_by=["metrics.val_final_f1 DESC"],
)
champ = next(r for r in runs if r.info.run_name == a.champion)
runner = next(r for r in runs if r.info.run_name not in (a.champion, "final_test"))


def register(run):
    # register the saved Keras file directly (works without an MLflow "logged model" record)
    return client.create_model_version(
        name=NAME,
        source=f"runs:/{run.info.run_id}/keras_file",
        run_id=run.info.run_id,
        description=f"run {run.info.run_name}, val F1 {run.data.metrics.get('val_final_f1', float('nan')):.4f}",
    )


v1 = register(runner)
v2 = register(champ)
client.set_registered_model_alias(NAME, "champion", v2.version)
print(f"v{v1.version} <- {runner.info.run_name}")
print(f"v{v2.version} <- {champ.info.run_name}")
print(f"alias 'champion' -> v{v2.version}")