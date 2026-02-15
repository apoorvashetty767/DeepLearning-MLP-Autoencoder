import numpy as np

# Load MNIST from keras (no training yet)
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Save as .npz so we can reuse without keras later
np.savez(
    "mnist.npz",
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test
)

print("Saved MNIST dataset successfully")
