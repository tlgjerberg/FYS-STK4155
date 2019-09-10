import numpy as np
from numpy import linalg
from random import random, seed
from designMatrix import CreateDesignMatrix_X


def CreateDesignMatrix_X(x, y, n=5):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = x**(i - k) * y**k

    return X


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4 + eps  # Added noise term eps


def MSE(y_data, y_predict):
    n = np.size(y_predict)
    return np.sum((y_data - y_predict)**2) / n


def R2(y_data, y_predict):
    return 1 - np.sum((y_data - y_predict)**2) / np.sum((y_data - np.mean(y_data))**2)
