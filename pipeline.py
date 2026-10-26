import keras
import pandas as pd
import tensorflow as tf

SEED = 42
IMG_SIZE = (224, 224)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

# random augmentation, used for the training set only
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal", seed=SEED),
    keras.layers.RandomRotation(0.05, seed=SEED),
    keras.layers.RandomZoom(0.1, seed=SEED),
    keras.layers.RandomContrast(0.1, seed=SEED),
], name="augmentation")


def _size(img_size):
    return (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)


def make_loader(img_size):
    size = _size(img_size)

    def load_image(path, label):
        raw = tf.io.read_file(path)                                         # 1. read bytes
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)  # 2. decode, force RGB
        img = tf.image.resize(img, size, antialias=True)                    # 3. resize to a fixed size
        img = tf.cast(img, tf.float32)                                      # values stay 0..255
        return img, tf.cast(label, tf.float32)

    return load_image


def make_dataset(csv_path, training, img_size=IMG_SIZE, batch=BATCH, use_aug=True):
    df = pd.read_csv(csv_path)
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(make_loader(img_size), num_parallel_calls=AUTOTUNE)
    if training and use_aug:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = make_dataset("reports/splits/train.csv", training=True)
    images, labels = next(iter(ds))
    print("batch:", images.shape, labels.shape, "pixel range:", float(tf.reduce_min(images)), float(tf.reduce_max(images)))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, img, lab in zip(axes.ravel(), images[:8], labels[:8]):
        ax.imshow(tf.clip_by_value(img, 0, 255).numpy().astype("uint8"))
        ax.set_title("AI-generated" if lab == 1 else "Real", fontsize=9)
        ax.axis("off")
    fig.suptitle("One augmented training batch  [23l2540_A2_Ops]")
    fig.tight_layout()
    fig.savefig("reports/figures/augmented_batch.png", dpi=120)