import numpy as np


def loss(y, y_pred):
    return (1/len(y)) * np.sum((y - y_pred)**2)
