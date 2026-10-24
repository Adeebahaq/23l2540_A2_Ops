import sys, platform
import tensorflow as tf
import keras, numpy, pandas, PIL, sklearn, mlflow, flask

print("Python:", sys.version)
print("OS:", platform.platform())
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Pillow:", PIL.__version__)
print("scikit-learn:", sklearn.__version__)
print("MLflow:", mlflow.__version__)
print("Flask:", flask.__version__)
print("GPUs visible:", tf.config.list_physical_devices('GPU'))