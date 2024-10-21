import numpy as np

# Example dataset (Size, Price)
X = np.array([2104, 1416, 1534, 852])
y = np.array([460, 232, 315, 178])

# Feature scaling (normalize the X values)
X_scaled = (X - np.mean(X)) / np.std(X)  # Standardize the data

# Initialize parameters
w, b = 0, 0
alpha = 0.00001  # Smaller learning rate to avoid overflow

# Batch Gradient Descent
def batch_gradient_descent(X, y, w, b, alpha, iterations):
    N = len(y)
    for _ in range(iterations):
        y_pred = w * X + b
        dw = -(2/N) * np.sum(X * (y - y_pred))
        db = -(2/N) * np.sum(y - y_pred)
        w -= alpha * dw
        b -= alpha * db
    return w, b

# Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, w, b, alpha, iterations):
    N = len(y)
    for _ in range(iterations):
        for i in range(N):
            y_pred = w * X[i] + b
            dw = -(2) * X[i] * (y[i] - y_pred)
            db = -(2) * (y[i] - y_pred)
            w -= alpha * dw
            b -= alpha * db
    return w, b

# Run both gradient descents
iterations = 10000  # Increase iterations since we're using a smaller learning rate
w_batch, b_batch = batch_gradient_descent(X_scaled, y, w, b, alpha, iterations)
w_sgd, b_sgd = stochastic_gradient_descent(X_scaled, y, w, b, alpha, iterations)

print("Batch Gradient Descent: w =", w_batch, ", b =", b_batch)
print("Stochastic Gradient Descent: w =", w_sgd, ", b =", b_sgd)
