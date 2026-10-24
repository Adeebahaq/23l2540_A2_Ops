import tensorflow as tf
import keras_tuner as kt

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x = x / 255.0
xt, yt, xv, yv = x[:50000], y[:50000], x[50000:], y[50000:]

def build(hp):
    m = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
              "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6,
                        seed=42, directory="tuner_toy", project_name="fmnist", overwrite=True)
tuner.search(xt, yt, epochs=3, validation_data=(xv, yv))
print("BEST:", tuner.get_best_hyperparameters()[0].values)
for t in tuner.oracle.get_best_trials(6):
    print(t.hyperparameters.values, round(t.score, 4))