import numpy as np

np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X + 2 + 0.1 * np.random.randn(100, 1)

learning_rate = 0.01
epochs = 1000
weights = np.random.rand(2, 1)

print("\nInitial weights:\n", weights)

for epoch in range(epochs):
    X_b = np.c_[np.ones((len(X), 1)), X]
    y_pred = X_b.dot(weights)
    error = y_pred - y
    gradient = 2 * X_b.T.dot(error) / len(X_b)
    weights -= learning_rate * gradient

    mse = np.mean(error ** 2)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.4f}")

print("\nFinal weights:\n", weights)