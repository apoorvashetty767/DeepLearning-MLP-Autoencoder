import numpy as np

# Load MNIST data
data = np.load("mnist.npz")
X_train = data["x_train"]
y_train = data["y_train"]
X_test = data["x_test"]
y_test = data["y_test"]

print("Original X_train shape:", X_train.shape)  # (60000,28,28)
print("Original X_test shape:", X_test.shape)    # (10000,28,28)

# Flatten images: 28x28 -> 784
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values to 0-1
X_train_flat = X_train_flat / 255.0
X_test_flat = X_test_flat / 255.0

print("Flattened X_train shape:", X_train_flat.shape)  # (60000,784)
print("Flattened X_test shape:", X_test_flat.shape)    # (10000,784)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
