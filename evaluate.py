import tensorflow as tf
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix,
                             ConfusionMatrixDisplay)
from train import make_ds   # uses your pipeline; no augmentation or shuffling for val/test

va = pd.read_csv("reports/splits/val.csv")
te = pd.read_csv("reports/splits/test.csv")
print("Label 1 =", te[te.label == 1]["class"].iloc[0], "| label 0 =", te[te.label == 0]["class"].iloc[0])

val_ds, test_ds = make_ds(va, False), make_ds(te, False)
models = {"cnn": "cnn_best.keras", "mobilenet": "tl_best.keras"}

rows, preds = [], {}
fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
for name, path in models.items():
    m = tf.keras.models.load_model(path)
    p_val = m.predict(val_ds, verbose=0).ravel()
    p_test = m.predict(test_ds, verbose=0).ravel()
    y_val, y_test = va.label.values, te.label.values

    # threshold chosen on VALIDATION only (Youden's J)
    fpr, tpr, thr = roc_curve(y_val, p_val)
    t = float(thr[np.argmax(tpr - fpr)])

    # single evaluation on the TEST set
    y_pred = (p_test >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, p_test)
    rows.append(dict(model=name, threshold=round(t, 3), accuracy=accuracy_score(y_test, y_pred),
                     precision=pr, recall=rc, f1=f1, roc_auc=auc))
    preds[name] = (p_test, y_pred)

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title(f"{name} confusion matrix (thr={t:.2f})")
    plt.savefig(f"{name}_confusion.png", dpi=150, bbox_inches="tight"); plt.close()

    f, tp, _ = roc_curve(y_test, p_test)
    ax_roc.plot(f, tp, label=f"{name} (AUC={auc:.3f})")

ax_roc.plot([0, 1], [0, 1], "k--"); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
ax_roc.set_title("Test ROC"); ax_roc.legend()
fig_roc.savefig("roc_test.png", dpi=150, bbox_inches="tight")

table = pd.DataFrame(rows).round(3)
print(table.to_string(index=False))
table.to_csv("test_metrics.csv", index=False)

# 2.6 error analysis: 10 most confident mistakes of MobileNetV2
p, yp = preds["mobilenet"]
wrong = te.assign(prob=p, pred=yp)
wrong = wrong[wrong.label != wrong.pred]
wrong["conf"] = np.where(wrong.pred == 1, wrong.prob, 1 - wrong.prob)
wrong = wrong.sort_values("conf", ascending=False).head(10)
print(wrong[["path", "class", "label", "pred", "conf"]].to_string(index=False))
fig, axs = plt.subplots(2, 5, figsize=(16, 7))
for ax, (_, r) in zip(axs.ravel(), wrong.iterrows()):
    img = Image.open(r.path).convert("RGB")
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"true={r['class']}\npred={'1' if r.pred else '0'} conf={r.conf:.2f}\n{img.size[0]}x{img.size[1]}", fontsize=8)
fig.savefig("misclassified.png", dpi=150, bbox_inches="tight")
wrong.to_csv("misclassified.csv", index=False)