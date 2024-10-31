import numpy as np

from predict import predict
from loss import loss

def bgd(X, y, learning_rate, epochs):

    w = np.zeros(X.shape[1])
    b = 0

    # print(f'w: {w}')

    for epoch in range(epochs + 1):

        y_pred = predict(X, w, b)

        l = loss(y, y_pred)

        if (epoch + 1) % 1000 == 0:
            print(f'==== Epoch {epoch+1} Loss : {l:.4f} ====')
        
        # w -= learning_rate  * (2/len(y)) * np.sum(y - y_pred)
        # b -= learning_rate  * (2/len(y)) * np.sum(y - y_pred)

        # Calculate gradients for weights and bias
        w_gradient = -(2 / len(y)) * np.dot(X.T, (y - y_pred))
        b_gradient = -(2 / len(y)) * np.sum(y - y_pred)

        # Update weights and bias
        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

    return w, b
