import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# Load MNIST
# ==============================
data = np.load("mnist.npz")
X = data["x_train"]

X = X.astype(np.float32) / 255.0
X = X.reshape(X.shape[0], -1)

# ==============================
# Load trained weights
# (same initialization + training logic reused)
# ==============================
# IMPORTANT:
# Copy these values from step5_autoencoder.py
input_size = 784
hidden_size = 128
latent_size = 64

# Reinitialize weights (same seed behavior assumed)
np.random.seed(42)

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)

W2 = np.random.randn(hidden_size, latent_size) * 0.01
b2 = np.zeros(latent_size)

W3 = np.random.randn(latent_size, hidden_size) * 0.01
b3 = np.zeros(hidden_size)

W4 = np.random.randn(hidden_size, input_size) * 0.01
b4 = np.zeros(input_size)

# ==============================
# Activation functions
# ==============================
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ==============================
# Forward pass (reconstruction)
# ==============================
a1 = relu(X @ W1 + b1)
latent = relu(a1 @ W2 + b2)
a3 = relu(latent @ W3 + b3)
X_recon = sigmoid(a3 @ W4 + b4)

# ==============================
# Reconstruction Error
# ==============================
recon_error = np.mean((X - X_recon) ** 2, axis=1)

# Threshold (top 1% as outliers)
threshold = np.percentile(recon_error, 99)

outlier_indices = np.where(recon_error > threshold)[0]

print(f"Detected {len(outlier_indices)} outliers")

# ==============================
# Save Outlier Images
# ==============================
os.makedirs("outliers", exist_ok=True)

for i, idx in enumerate(outlier_indices[:10]):
    plt.imshow(X[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Outlier {i}")
    plt.axis("off")
    plt.savefig(f"outliers/outlier_{i}.png")
    plt.close()