import os, sys, glob, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_fscore_support, accuracy_score)

CLASS_NAMES = ["Real", "AI-generated"]   # label 0 = Real, label 1 = AI-generated (positive class)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_version():
    """Reads md5 hashes from the .dvc pointer files in the project root."""
    hashes = []
    for f in glob.glob("*.dvc"):
        for line in open(f, encoding="utf-8"):
            if "md5:" in line:
                hashes.append(f"{os.path.basename(f)}:{line.split('md5:')[1].strip()}")
                break
    return ";".join(hashes) if hashes else "unknown"


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual approach: logs every epoch's metrics. Offset keeps stage 2 after stage 1."""
    def __init__(self, prefix="", offset=0):
        super().__init__()
        self.prefix, self.offset = prefix, offset

    def on_epoch_end(self, epoch, logs=None):
        d = {f"{self.prefix}{k}": float(v) for k, v in (logs or {}).items()}
        mlflow.log_metrics(d, step=self.offset + epoch)


def collect_preds(model, ds):
    """ds must NOT be shuffled, so labels line up with predictions."""
    y_true = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
    y_prob = model.predict(ds, verbose=0).ravel()
    return y_true, y_prob


def best_threshold_f1(y_true, y_prob):
    """Threshold chosen on VALIDATION data: maximises F1 of the AI-generated class."""
    ts = np.linspace(0.05, 0.95, 91)
    f1s = [precision_recall_fscore_support(y_true, (y_prob >= t).astype(int),
                                           average="binary", zero_division=0)[2] for t in ts]
    return float(ts[int(np.argmax(f1s))])


def metrics_at(y_true, y_prob, thr, prefix):
    pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {f"{prefix}accuracy": float(accuracy_score(y_true, pred)),
            f"{prefix}precision": float(p), f"{prefix}recall": float(r), f"{prefix}f1": float(f1),
            f"{prefix}roc_auc": float(roc_auc_score(y_true, y_prob))}


def save_cm_roc(y_true, y_prob, thr, tag, outdir="artifacts_tmp"):
    os.makedirs(outdir, exist_ok=True)
    cm = confusion_matrix(y_true, (y_prob >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion ({tag}, thr={thr:.2f})")
    cm_path = f"{outdir}/confusion_matrix_{tag}.png"
    fig.savefig(cm_path, dpi=120, bbox_inches="tight"); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(); ax.set_title(f"ROC ({tag})")
    roc_path = f"{outdir}/roc_curve_{tag}.png"
    fig.savefig(roc_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return cm_path, roc_path


def log_environment(outdir="artifacts_tmp"):
    os.makedirs(outdir, exist_ok=True)
    path = f"{outdir}/pip_freeze.txt"
    with open(path, "w") as f:
        f.write(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    mlflow.log_artifact(path)