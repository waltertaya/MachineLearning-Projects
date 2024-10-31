import numpy as np

def predict(X, w, b):
    return (np.matmul(X, w) + b)
