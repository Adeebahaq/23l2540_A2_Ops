import tensorflow as tf, keras, keras_tuner as kt, pandas as pd
from pipeline import make_dataset

tf.keras.utils.set_random_seed(42)
IMG, EP1, EP2 = 224, 3, 3          # epochs for stage 1 (head) and stage 2 (fine-tune)
train_ds = make_dataset("reports/splits/train.csv", True, IMG, use_aug=True)
val_ds = make_dataset("reports/splits/val.csv", False, IMG)   # validation only; test set is never used

class VisionGuardHM(kt.HyperModel):
    def build(self, hp):
        dropout = hp.Choice("dropout", [0.2, 0.3, 0.5])
        lr = hp.Choice("lr", [1e-3, 3e-4, 1e-4])
        hp.Choice("unfreeze", [10, 20, 40])
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                                 input_shape=(IMG, IMG, 3))
        base.trainable = False
        inp = keras.Input((IMG, IMG, 3))
        x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inp)   # same maths as mobilenet_v2.preprocess_input
        x = base(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(dropout)(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, hp, model, train, **kw):
        kw.pop("epochs", None)
        model.fit(train, epochs=EP1, verbose=2, **kw)            # stage 1: head only
        base = [l for l in model.layers if isinstance(l, keras.Model)][0]
        cut = len(base.layers) - hp.get("unfreeze")
        base.trainable = True
        for i, l in enumerate(base.layers):                      # stage 2: last N layers, BatchNorm frozen
            l.trainable = (i >= cut) and not isinstance(l, keras.layers.BatchNormalization)
        model.compile(optimizer=keras.optimizers.Adam(hp.get("lr") / 100),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model.fit(train, epochs=EP2, verbose=2, **kw)

tuner = kt.RandomSearch(VisionGuardHM(), objective="val_loss", max_trials=8, seed=42,
                        directory="tuner_vg", project_name="mobilenet", overwrite=True)
tuner.search(train_ds, validation_data=val_ds)

rows = []
for t in tuner.oracle.get_best_trials(8):
    r = dict(t.hyperparameters.values); r["val_loss"] = round(t.score, 4); rows.append(r)
df = pd.DataFrame(rows).sort_values("val_loss")
print("\nVISIONGUARD TUNER RESULTS (lower val_loss is better)")
print(df.to_string(index=False))
for c in ["lr", "dropout", "unfreeze"]:
    print("\n", df.groupby(c)["val_loss"].agg(["mean", "count"]).round(4))
