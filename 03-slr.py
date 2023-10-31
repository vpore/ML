import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleLinearRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.slope = np.zeros(num_features)
        self.intercept = 0

        for _ in range(self.num_iterations):
            y_pred = np.dot(X, self.slope) + self.intercept
            grad_slope = - (2 / num_samples) * np.dot(X.T, y - y_pred)
            grad_intercept = - (2 / num_samples) * np.sum(y - y_pred)

            self.slope -= self.learning_rate * grad_slope
            self.intercept -= self.learning_rate * grad_intercept
    
    def predict(self, X):
        return np.dot(X, self.slope) + self.intercept

    def calculate_rmse(self, y, y_pred):
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse

data = pd.read_csv("Salary_dataset.csv")

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# X = data["YearsExperience"].values
# y = data["Salary"].values
# X = X.reshape(-1, 1)

model = SimpleLinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)
y_pred = model.predict(X)
rmse = model.calculate_rmse(y, y_pred)

print(f"RMSE: {rmse}")

plt.scatter(X, y, label="Actual data")
plt.plot(X, y_pred, color='pink', label="Fitted line")
plt.xlabel("Years_Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression - Salary dataset")
plt.legend()
plt.show()