import numpy as np

from loss import loss
from predict import predict


def sgd(X, y, learning_rate, epochs):
    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(epochs + 1):

        for i in range(len(y)):
            y_pred = np.matmul(X[i], w) + b

            w_gradient = -2 * X[i] * (y[i] - y_pred)
            b_gradient = -2 * (y[i] - y_pred)

            w -= learning_rate * w_gradient
            b -= learning_rate * b_gradient
        
        l = loss(y, predict(X, w, b))

        if (epoch + 1) % 1000 == 0:
            print(f'==== Epoch {epoch+1} Loss : {l:.4f} ====')

    return w, b
