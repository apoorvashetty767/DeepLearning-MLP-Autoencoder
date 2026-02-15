import numpy as np
import matplotlib.pyplot as plt

# 1. Load data (MNIST)
data = np.load("mnist.npz")
X_train = data["x_train"].reshape(data["x_train"].shape[0], -1) / 255.0
y_train = data["y_train"]
X_test = data["x_test"].reshape(data["x_test"].shape[0], -1) / 255.0
y_test = data["y_test"]

# 2. Helper functions
def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def accuracy(preds, labels):
    return np.mean(np.argmax(preds, axis=1) == labels)

# 3. Initialize network
input_size = 784
hidden_size = 128
output_size = 10

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

print("Network initialized successfully!")

# 4. Training parameters
learning_rate = 0.1
epochs = 5
batch_size = 128

num_samples = X_train.shape[0]

# 👉 ADDED: store metrics
loss_history = []
acc_history = []

# ---------------- Training loop starts ----------------
for epoch in range(epochs):
    # Shuffle data each epoch
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    epoch_loss = 0

    # Loop over batches
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]
        m = X_batch.shape[0]

        # One-hot encode labels
        y_batch_onehot = one_hot(y_batch)

        # Forward pass
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)

        # Loss
        loss = -np.sum(y_batch_onehot * np.log(A2 + 1e-8)) / m
        epoch_loss += loss * m

        # Backpropagation
        dZ2 = A2 - y_batch_onehot
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_batch.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Average loss for epoch
    epoch_loss /= num_samples

    # Accuracy on training data
    Z1_train = np.dot(X_train, W1) + b1
    A1_train = relu(Z1_train)
    Z2_train = np.dot(A1_train, W2) + b2
    A2_train = softmax(Z2_train)
    train_acc = accuracy(A2_train, y_train)

    # 👉 ADDED: save metrics
    loss_history.append(epoch_loss)
    acc_history.append(train_acc)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {train_acc:.4f}")

# Test accuracy
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)
test_acc = accuracy(A2_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# ---------------- PLOTS ----------------
epochs_range = range(1, epochs + 1)

# Loss curve
plt.figure()
plt.plot(epochs_range, loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.savefig("mlp_loss_curve.png")
plt.show()

# Accuracy curve
plt.figure()
plt.plot(epochs_range, acc_history)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Epochs")
plt.savefig("mlp_accuracy_curve.png")
plt.show()
