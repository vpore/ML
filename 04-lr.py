import numpy as np

np.random.seed(0)
num_samples = 10
X = np.random.randn(num_samples, 2)
y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])

def sigmoid(z):
  return 1/(1+np.exp(-z))

def fit_LoR(X, y, learning_rate=0.01, num_iterations=1000):
  num_samples, num_features = X.shape
  weights = np.zeros(num_features)
  bias = 0
  for _ in range(num_iterations):
    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    dw = (1/num_samples)*np.dot(X.T, predictions-y)
    db = (1/num_samples)*np.sum(predictions-y)
    weights -= learning_rate*dw
    bias -= learning_rate*db
  return weights, bias

def predict_LoR(X, weights, bias):
  linear_model = np.dot(X, weights)+bias
  predictions = sigmoid(linear_model)
  predicted_labels = [1 if p > 0.5 else 0 for p in predictions]
  return predicted_labels

weights, bias = fit_LoR(X, y, learning_rate=0.01, num_iterations=1000)
predicted_labels = predict_LoR(X, weights, bias)
print(weights)
print(bias)
print(predicted_labels)