import numpy as np
from numpy import linalg
from random import random, seed
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score


def CreateDesignMatrix_X(x, y, n=5):
    """
    Function for creating a design X-matrix with
    rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh,
    keyword agruments n is the degree of the polynomial you want to fit.
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


def FrankeFunction(x, y, eps):
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


def KFoldCrossValidation(x, y, z, k, p):
    """
    K-fold cross validation of data (x,y) and z with k folds and polynomial
    degree p. Returns the best R2 score.
    """

    # print(x)

    # KFold instance
    kfold = KFold(n_splits=k)

    MSE = np.zeros(k)
    R2 = np.zeros(k)
    beta = np.zeros((k, int((p + 1) * (p + 2) / 2)))
    index = 0

    for train_ind, test_ind in kfold.split(x):

        # Assigning train and test data
        x_train_cv = x[train_ind]
        x_test_cv = x[test_ind]

        y_train_cv = y[train_ind]
        y_test_cv = y[test_ind]

        z_train_cv = z[train_ind]
        z_test_cv = z[test_ind]

        # Raveling z data into 1D arrays
        z_train_cv_1d = np.ravel(z_train_cv)
        z_test_cv_1d = np.ravel(z_test_cv)

        # OLS to find a model for prediction using train data

        # Setting up the design matrices for training and test data
        XY_train_cv = CreateDesignMatrix_X(x_train_cv, y_train_cv, n=p)
        XY_test_cv = CreateDesignMatrix_X(x_test_cv, y_test_cv, n=p)

        # Inverting
        XY2_cv_inv = np.linalg.inv(XY_train_cv.T.dot(XY_train_cv))

        # Model parameters beta based on training data design matrix
        beta_cv = XY2_cv_inv.dot(XY_train_cv.T).dot(z_train_cv_1d)

        # Combining test design matrix with model parameters
        z_testModel_cv = XY_test_cv @ beta_cv

        MSE[index] = mean_squared_error(z_test_cv_1d, z_testModel_cv)
        R2[index] = r2_score(z_test_cv_1d, z_testModel_cv)
        beta[index, :] = beta_cv
        index += 1

        # print(np.argmax(R2))

    return beta[np.argmax(R2)]
