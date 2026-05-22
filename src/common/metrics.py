import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))
