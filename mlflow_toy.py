import os, sys, argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=0.001)
p.add_argument("--dagshub", action="store_true", help="log to DagsHub (3.2)")
a = p.parse_args()

if a.dagshub:
    from dotenv import load_dotenv
    load_dotenv()  # reads MLFLOW_TRACKING_URI / USERNAME / PASSWORD from .env
    exp = "23L-2540_A2_Ops_toy"
else:
    mlflow.set_tracking_uri("sqlite:///mlflow_toy.db")
    exp = "toy_local"
mlflow.set_experiment(exp)

tf.keras.utils.set_random_seed(42)
(xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()
xtr, xte = xtr / 255.0, xte / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(a.lr),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

class Log(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)

with mlflow.start_run(run_name=f"toy_lr_{a.lr}"):
    mlflow.set_tags({"stage": "toy", "dataset": "fashion_mnist"})
    mlflow.log_params({"lr": a.lr, "units": 64, "epochs": 5, "seed": 42})
    h = model.fit(xtr, ytr, epochs=5, validation_split=0.1, batch_size=128,
                  callbacks=[Log()], verbose=2)
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"lr={a.lr}")
    plt.savefig("toy_loss.png", dpi=120); plt.close()
    mlflow.log_artifact("toy_loss.png")
    mlflow.log_metric("test_acc", float(model.evaluate(xte, yte, verbose=0)[1]))
    mlflow.keras.log_model(model, "model")