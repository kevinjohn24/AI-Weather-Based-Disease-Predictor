import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

# 1. Load dataset
iris = load_iris()
X = iris.data.astype(np.float32)   # shape (150,4)
y = iris.target.reshape(-1,1)      # (150,1)

# Split into train/test
X_train, X_test, y_train_idx, y_test_idx = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Scale features (important for stable training)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode labels
enc = OneHotEncoder(sparse_output=False)
Y_train = enc.fit_transform(y_train_idx)
Y_test  = enc.transform(y_test_idx)

# 2. Define helpers
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def relu(z): return np.maximum(0, z)
def drelu(z): return (z > 0).astype(float)

def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

# 3. Initialize weights (4 → 8 → 3)
in_dim, h_dim, out_dim = 4, 8, 3
W1, b1 = np.random.randn(in_dim, h_dim)*0.1, np.zeros((1,h_dim))
W2, b2 = np.random.randn(h_dim, out_dim)*0.1, np.zeros((1,out_dim))

lr, epochs, batch_size = 0.1, 300, 16

# 4. Training loop
for epoch in range(epochs):
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        xb, yb = X_train[idx[i:i+batch_size]], Y_train[idx[i:i+batch_size]]

        # Forward
        z1 = xb @ W1 + b1; a1 = relu(z1)
        z2 = a1 @ W2 + b2; a2 = softmax(z2)
        loss = cross_entropy(yb, a2)

        # Backward
        m = len(xb)
        dz2 = (a2 - yb) / m
        dW2, db2 = a1.T @ dz2, np.sum(dz2, axis=0, keepdims=True)
        dz1 = (dz2 @ W2.T) * drelu(z1)
        dW1, db1 = xb.T @ dz1, np.sum(dz1, axis=0, keepdims=True)

        # Update
        W1 -= lr*dW1; b1 -= lr*db1
        W2 -= lr*dW2; b2 -= lr*db2

    if epoch % 50 == 0 or epoch == 299:
        print(f"Epoch {epoch+1}, Loss = {loss:.4f}")

# 5. Evaluate scratch model
a1 = relu(X_test @ W1 + b1)
probs = softmax(a1 @ W2 + b2)
pred_scratch = np.argmax(probs, axis=1)
acc_scratch = accuracy_score(y_test_idx, pred_scratch)
print(f"\nScratch model accuracy: {acc_scratch:.4f}")

# 6. Compare with sklearn MLP
clf = MLPClassifier(hidden_layer_sizes=(8,), activation="relu",
                    solver="adam", max_iter=500, random_state=42)
clf.fit(X_train, y_train_idx.ravel())
acc_lib = accuracy_score(y_test_idx, clf.predict(X_test))
print(f"Sklearn MLP accuracy  : {acc_lib:.4f}")
