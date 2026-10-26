import os, argparse, traceback
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
from dotenv import load_dotenv
from pipeline import make_dataset
from mlflow_utils import (git_commit, dvc_data_version, MLflowEpochLogger, collect_preds,
                          best_threshold_f1, metrics_at, save_cm_roc, log_environment)

SPLITS = ["reports/splits/train.csv", "reports/splits/val.csv", "reports/splits/test.csv"]
EXPERIMENT = "23L-2540_A2_Ops"

# each backbone with the preprocessing it expects (this is why they are paired)
BACKBONES = {
    "MobileNetV2": (tf.keras.applications.MobileNetV2,
                    tf.keras.applications.mobilenet_v2.preprocess_input),     # scales to [-1, 1]
    "EfficientNetB0": (tf.keras.applications.EfficientNetB0,
                       tf.keras.applications.efficientnet.preprocess_input),  # passthrough, rescales inside
    "ResNet50": (tf.keras.applications.ResNet50,
                 tf.keras.applications.resnet.preprocess_input),              # BGR + mean subtraction
}

p = argparse.ArgumentParser()
p.add_argument("--run-name", required=True)
p.add_argument("--mode", choices=["manual", "autolog"], default="manual")
p.add_argument("--backbone", default="MobileNetV2", choices=list(BACKBONES))
p.add_argument("--img-size", type=int, default=224)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--lr-ft", type=float, default=1e-5)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--unfreeze", type=int, default=20, help="last N backbone layers to unfreeze (BatchNorm stays frozen)")
p.add_argument("--augment", type=int, default=1)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--epochs-ft", type=int, default=15)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--eval-test", action="store_true", help="ONLY for the final chosen run")
p.add_argument("--local", action="store_true", help="log to a local sqlite file instead of DagsHub (smoke tests)")
a = p.parse_args()

load_dotenv()
if a.local:
    mlflow.set_tracking_uri("sqlite:///mlflow_smoke.db")
mlflow.set_experiment(EXPERIMENT)
tf.keras.utils.set_random_seed(a.seed)

train_ds = make_dataset(SPLITS[0], True, a.img_size, use_aug=bool(a.augment))
val_ds = make_dataset(SPLITS[1], False, a.img_size)
test_ds = make_dataset(SPLITS[2], False, a.img_size) if a.eval_test else None


def build_model():
    ctor, prep = BACKBONES[a.backbone]
    base = ctor(include_top=False, weights="imagenet", input_shape=(a.img_size, a.img_size, 3))
    base.trainable = False
    inp = layers.Input((a.img_size, a.img_size, 3))
    x = prep(inp)                       # preprocessing lives inside the model
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(a.dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name=f"{a.backbone}_tl"), base


model, base = build_model()


def compile_(lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )


def callbacks(prefix, offset):
    cb = [tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True),
          tf.keras.callbacks.ModelCheckpoint("best_tmp.keras", monitor="val_loss", save_best_only=True),
          tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2)]
    if a.mode == "manual":
        cb.append(MLflowEpochLogger(prefix, offset))
    return cb


if a.mode == "autolog":
    mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

with mlflow.start_run(run_name=a.run_name):
    mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_data_version(),
                     "seed": str(a.seed), "logging_mode": a.mode, "backbone": a.backbone})
    if a.mode == "manual":
        mlflow.log_params(vars(a))

    # ---- Stage 1: frozen backbone, train the head ----
    compile_(a.lr)
    mlflow.log_text("\n".join(l.name for l in model.layers if l.trainable), "stage1_trainable_layers.txt")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=callbacks("s1_", 0), verbose=2)

    # ---- Stage 2: unfreeze last N backbone layers (BatchNorm stays frozen), much smaller LR ----
    base.trainable = True
    cut = len(base.layers) - a.unfreeze
    for i, l in enumerate(base.layers):
        l.trainable = (i >= cut) and not isinstance(l, layers.BatchNormalization)
    names = [l.name for l in base.layers if l.trainable]
    print("Stage 2 trainable layers:", names)
    mlflow.log_text("\n".join(names), "stage2_trainable_layers.txt")
    compile_(a.lr_ft)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_ft, callbacks=callbacks("s2_", a.epochs), verbose=2)

    # ---- Threshold from VALIDATION; comparison metrics are validation metrics ----
    yv, pv = collect_preds(model, val_ds)
    thr = best_threshold_f1(yv, pv)
    mlflow.log_metric("threshold", thr)
    mlflow.log_metrics(metrics_at(yv, pv, thr, "val_final_"))
    mlflow.log_metric("best_val_loss", float(min(h1.history["val_loss"] + h2.history["val_loss"])))

    if a.eval_test:   # test set is touched only here, once
        yt, pt = collect_preds(model, test_ds)
        mlflow.log_metrics(metrics_at(yt, pt, thr, "test_"))
        cm, roc = save_cm_roc(yt, pt, thr, "test")
    else:
        cm, roc = save_cm_roc(yv, pv, thr, "val")
    mlflow.log_artifact(cm)
    mlflow.log_artifact(roc)

    for f in SPLITS:
        if os.path.exists(f):
            mlflow.log_artifact(f, "splits")
    log_environment()

    model.save("model.keras")
    mlflow.log_artifact("model.keras", "keras_file")
    try:
        mlflow.keras.log_model(model, name="model")
    except Exception:
        traceback.print_exc()
        print("WARNING: mlflow.keras.log_model failed; model.keras artifact was still saved.")