import numpy as np

# -------------------------------
# 1. Load flattened MNIST data
# -------------------------------
data = np.load("mnist.npz")
X_train = data["x_train"].reshape(data["x_train"].shape[0], -1) / 255.0
y_train = data["y_train"]
X_test = data["x_test"].reshape(data["x_test"].shape[0], -1) / 255.0
y_test = data["y_test"]

# -------------------------------
# 2. Helper functions
# -------------------------------
def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# -------------------------------
# 3. Network architecture
# -------------------------------
input_size = 784
hidden_size = 128
output_size = 10

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

print("Network initialized successfully!")

# -------------------------------
# 4. Forward pass on small batch
# -------------------------------
X_batch = X_train[:100]
y_batch = y_train[:100]

y_batch_onehot = one_hot(y_batch)

Z1 = np.dot(X_batch, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = softmax(Z2)

loss = -np.sum(y_batch_onehot * np.log(A2 + 1e-8)) / X_batch.shape[0]

print("Forward pass done!")
print("Loss on first 100 samples:", loss)
