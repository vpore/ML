import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
X = data[:, [0, 2]]

y = iris.target

class MultipleLinearRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coefficients = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.coefficients = np.zeros(num_features)

        for _ in range(self.num_iterations):
            y_pred = np.dot(X, self.coefficients)
            error = y - y_pred
            gradient = - (2 / num_samples) * np.dot(X.T, error)
            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        return np.dot(X, self.coefficients)

    def calculate_r_squared(self, y, y_pred):
        ssr = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ssr / sst)
        return r_squared

    def calculate_rmse(self, y, y_pred):
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse

model = MultipleLinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)

y_pred = model.predict(X)

r_squared = model.calculate_r_squared(y, y_pred)
rmse = model.calculate_rmse(y, y_pred)

print(f"R-squared: {r_squared}")
print(f"RMSE: {rmse}")