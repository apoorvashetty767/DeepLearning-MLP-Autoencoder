import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# Configuration
# ==============================
input_size = 784
hidden_size = 128
latent_size = 64

learning_rate = 0.01
epochs = 30
batch_size = 64
lambda_l1 = 0.0001

# ==============================
# Activation Functions
# ==============================
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ==============================
# Load MNIST
# ==============================
data = np.load("mnist.npz")
X_train = data["x_train"]

X_train = X_train.astype(np.float32) / 255.0
X_train = X_train.reshape(X_train.shape[0], -1)
num_samples = X_train.shape[0]

# ==============================
# Initialize Weights
# ==============================
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)

W2 = np.random.randn(hidden_size, latent_size) * 0.01
b2 = np.zeros(latent_size)

W3 = np.random.randn(latent_size, hidden_size) * 0.01
b3 = np.zeros(hidden_size)

W4 = np.random.randn(hidden_size, input_size) * 0.01
b4 = np.zeros(input_size)

# ==============================
# Training
# ==============================
loss_history = []

for epoch in range(epochs):
    perm = np.random.permutation(num_samples)
    X_shuffled = X_train[perm]
    epoch_loss = 0

    for i in range(0, num_samples, batch_size):
        X = X_shuffled[i:i+batch_size]

        # -------- Forward pass --------
        z1 = X @ W1 + b1
        a1 = relu(z1)

        z2 = a1 @ W2 + b2
        latent = relu(z2)

        z3 = latent @ W3 + b3
        a3 = relu(z3)

        z4 = a3 @ W4 + b4
        output = sigmoid(z4)

        # -------- Loss --------
        mse_loss = np.mean((X - output) ** 2)
        l1_loss = lambda_l1 * np.mean(np.abs(latent))
        loss = mse_loss + l1_loss
        epoch_loss += loss

        # -------- Backprop --------
        d_out = (output - X) * sigmoid_derivative(z4)

        dW4 = a3.T @ d_out
        db4 = np.sum(d_out, axis=0)

        da3 = d_out @ W4.T
        dz3 = da3 * relu_derivative(z3)

        dW3 = latent.T @ dz3
        db3 = np.sum(dz3, axis=0)

        dlatent = dz3 @ W3.T
        dlatent += lambda_l1 * np.sign(latent)
        dz2 = dlatent * relu_derivative(z2)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # -------- Update --------
        W4 -= learning_rate * dW4
        b4 -= learning_rate * db4

        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    avg_loss = epoch_loss / (num_samples // batch_size)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

# ==============================
# Save Loss Plot
# ==============================
os.makedirs("plots", exist_ok=True)
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss")
plt.savefig("plots/autoencoder_loss.png")
plt.close()

# ==============================
# Save Reconstructions
# ==============================
os.makedirs("reconstructions", exist_ok=True)

samples = X_train[:10]

a1 = relu(samples @ W1 + b1)
latent = relu(a1 @ W2 + b2)
a3 = relu(latent @ W3 + b3)
recon = sigmoid(a3 @ W4 + b4)

for i in range(10):
    plt.figure(figsize=(4,2))

    plt.subplot(1,2,1)
    plt.imshow(samples[i].reshape(28,28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(recon[i].reshape(28,28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

    plt.savefig(f"reconstructions/recon_{i}.png")
    plt.close()