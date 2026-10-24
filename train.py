import numpy as np, pandas as pd, tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

SMOKE = False          # True = 1 epoch per stage (quick test). Set False for the real run.
SEED = 42
IMG, BS = 224, 32
AUTOTUNE = tf.data.AUTOTUNE
tf.keras.utils.set_random_seed(SEED)

# ---------- 2.1 input pipeline ----------
aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

def load(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG, IMG))      # stretches to square
    return tf.cast(img, tf.float32), label      # raw 0-255 values

def make_ds(df, training):
    ds = tf.data.Dataset.from_tensor_slices(
        (df.path.values, df.label.values.astype("float32")))
    if training:
        ds = ds.shuffle(len(df), seed=SEED)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(BS).prefetch(AUTOTUNE)

def save_aug_batch(ds, fname="augmented_batch.png"):
    x, y = next(iter(ds))
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i].numpy().astype("uint8"))
        plt.title(f"label={int(y[i])}"); plt.axis("off")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()

# ---------- models ----------
METRICS = ["accuracy", tf.keras.metrics.AUC(name="auc"),
           tf.keras.metrics.Precision(name="precision"),
           tf.keras.metrics.Recall(name="recall")]

def build_cnn(dropout=0.4):
    def block(x, f):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return layers.MaxPooling2D()(x)
    inp = layers.Input((IMG, IMG, 3))
    x = layers.Rescaling(1 / 255.)(inp)
    for f in (32, 64, 128):
        x = block(x, f)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="baseline_cnn")

def build_transfer(dropout=0.3):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
    base.trainable = False
    inp = layers.Input((IMG, IMG, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)   # scales to [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="mobilenet_tl"), base

def callbacks(path):
    return [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2),
    ]

def compile_(model, lr):
    model.compile(tf.keras.optimizers.Adam(lr), "binary_crossentropy", metrics=METRICS)

def plot_hist(hist, name):
    best = int(np.argmin(hist.history["val_loss"])) + 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, m in zip(ax, ("loss", "accuracy")):
        a.plot(range(1, len(hist.history[m]) + 1), hist.history[m], label="train")
        a.plot(range(1, len(hist.history[m]) + 1), hist.history["val_" + m], label="val")
        a.axvline(best, ls="--", c="r", label=f"best epoch {best}")
        a.set_title(f"{name}: {m}"); a.set_xlabel("epoch"); a.legend()
    fig.savefig(f"{name}_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return best

def ep(n):
    return 1 if SMOKE else n

if __name__ == "__main__":
    print("GPU:", tf.config.list_physical_devices("GPU"))
    tr = pd.read_csv("reports/splits/train.csv")
    va = pd.read_csv("reports/splits/val.csv")
    train_ds, val_ds = make_ds(tr, True), make_ds(va, False)
    save_aug_batch(train_ds)

    # 2.2 baseline CNN
    cnn = build_cnn(); compile_(cnn, 1e-3); cnn.summary()
    h = cnn.fit(train_ds, validation_data=val_ds, epochs=ep(30), callbacks=callbacks("cnn_best.keras"))
    print("CNN best epoch:", plot_hist(h, "cnn"))

    # 2.3 stage 1: frozen backbone
    tl, base = build_transfer(); compile_(tl, 1e-3)
    print("Stage 1 trainable layers in backbone:", [l.name for l in base.layers if l.trainable])
    h1 = tl.fit(train_ds, validation_data=val_ds, epochs=ep(10), callbacks=callbacks("tl_stage1.keras"))
    print("Stage 1 best epoch:", plot_hist(h1, "tl_stage1"))

    # 2.3 stage 2: unfreeze only the last block (BatchNorm stays frozen)
    base.trainable = True
    for l in base.layers:
        l.trainable = l.name.startswith("block_16") and not isinstance(l, layers.BatchNormalization)
    print("Stage 2 trainable layers:", [l.name for l in base.layers if l.trainable])
    compile_(tl, 1e-5)
    h2 = tl.fit(train_ds, validation_data=val_ds, epochs=ep(15), callbacks=callbacks("tl_best.keras"))
    print("Stage 2 best epoch:", plot_hist(h2, "tl_stage2"))