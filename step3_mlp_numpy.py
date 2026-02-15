# -------------------------------
# 4. Forward pass on small batch
# -------------------------------

# Use first 100 samples for testing
X_batch = X_train[:100]
y_batch = y_train[:100]

# One-hot encode labels
y_batch_onehot = one_hot(y_batch)

# Forward pass
Z1 = np.dot(X_batch, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = softmax(Z2)

# Cross-entropy loss
loss = -np.sum(y_batch_onehot * np.log(A2 + 1e-8)) / X_batch.shape[0]

print("Forward pass done!")
print("Loss on first 100 samples:", loss)
