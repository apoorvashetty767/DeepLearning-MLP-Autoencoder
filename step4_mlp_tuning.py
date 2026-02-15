import numpy as np
import pandas as pd  # for saving results

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

# 3. Network parameters
input_size = 784
output_size = 10

# -------------------------------
# 4. Hyperparameter tuning values
# -------------------------------
learning_rates = [0.01, 0.05, 0.1]
hidden_sizes = [64, 128, 256]
batch_sizes = [64, 128, 256]

results = []

for lr in learning_rates:
    for hs in hidden_sizes:
        for bs in batch_sizes:
            print(f"Testing lr={lr}, hidden_size={hs}, batch_size={bs}")

            # Initialize network for this combination
            np.random.seed(42)
            W1 = np.random.randn(input_size, hs) * 0.01
            b1 = np.zeros((1, hs))
            W2 = np.random.randn(hs, output_size) * 0.01
            b2 = np.zeros((1, output_size))

            epochs = 1  # quick tuning, only 1 epoch
            num_samples = X_train.shape[0]

            # Shuffle data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0

            # Batch training
            for start in range(0, num_samples, bs):
                end = start + bs
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                m = X_batch.shape[0]

                y_batch_onehot = one_hot(y_batch)

                # Forward pass
                Z1 = np.dot(X_batch, W1) + b1
                A1 = relu(Z1)
                Z2 = np.dot(A1, W2) + b2
                A2 = softmax(Z2)

                # Loss
                loss = -np.sum(y_batch_onehot * np.log(A2 + 1e-8)) / m
                epoch_loss += loss * m

                # Backprop
                dZ2 = A2 - y_batch_onehot
                dW2 = np.dot(A1.T, dZ2) / m
                db2 = np.sum(dZ2, axis=0, keepdims=True) / m
                dA1 = np.dot(dZ2, W2.T)
                dZ1 = dA1 * relu_derivative(Z1)
                dW1 = np.dot(X_batch.T, dZ1) / m
                db1 = np.sum(dZ1, axis=0, keepdims=True) / m

                # Update weights
                W1 -= lr * dW1
                b1 -= lr * db1
                W2 -= lr * dW2
                b2 -= lr * db2

            epoch_loss /= num_samples

            # Training accuracy
            Z1_train = np.dot(X_train, W1) + b1
            A1_train = relu(Z1_train)
            Z2_train = np.dot(A1_train, W2) + b2
            A2_train = softmax(Z2_train)
            train_acc = accuracy(A2_train, y_train)

            results.append((lr, hs, bs, epoch_loss, train_acc))
            print(f"Loss={epoch_loss:.4f}, Train Acc={train_acc:.4f}\n")

# Save results for report
df = pd.DataFrame(results, columns=['Learning rate','Hidden size','Batch size','Loss','Train Acc'])
df.to_csv("hyperparam_tuning_results.csv", index=False)
print("Saved results to hyperparam_tuning_results.csv")
