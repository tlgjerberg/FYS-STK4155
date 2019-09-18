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


def MatrixInvSVD(X):

    u, s, v = np.linalg.svd(X)

    uT = u.transpose()
    vT = v.transpose()

    XInv = np.dot(vT, np.dot(np.diag(s**-1), uT))

    return XInv


def KFoldCrossValidation(x, y, z, k, p):
    """
    K-fold cross validation of data (x,y) and z with k folds and polynomial
    degree p. Returns the best R2 score.
    """

    # print(x)

    # KFold instance
    kfold = KFold(n_splits=k, shuffle=True)

    MSE_test = np.zeros(k)
    MSE_train = np.zeros(k)
    R2_test = np.zeros(k)
    R2_train = np.zeros(k)
    beta = np.zeros((k, int((p + 1) * (p + 2) / 2)))
    tot_R2_estimate_test = tot_MSE_estimate_test = 0
    tot_R2_estimate_train = tot_MSE_estimate_train = 0
    index = 0
    jndex = 0

    for train_ind, test_ind in kfold.split(x):

        # Assigning train and test data
        x_train = x[train_ind]
        x_test = x[test_ind]

        y_train = y[train_ind]
        y_test = y[test_ind]

        z_train = z[train_ind]
        z_test = z[test_ind]

        # Raveling z data into 1D arrays
        z_train_1d = np.ravel(z_train)
        z_test_1d = np.ravel(z_test)

        # OLS to find a model for prediction using train data

        # Setting up the design matrices for training and test data
        XY_train = CreateDesignMatrix_X(x_train, y_train, n=p)
        XY_test = CreateDesignMatrix_X(x_test, y_test, n=p)

        # Inverting
        # XY2_inv = np.linalg.inv(XY_train.T.dot(XY_train))
        XY2_inv_SVD = MatrixInvSVD(XY_train.T.dot(XY_train))

        # XY2_inv = MatrixInvSVD(XY_train)

        # Model parameters beta based on training data design matrix
        beta = XY2_inv_SVD.dot(XY_train.T).dot(z_train_1d)

        # Combining test design matrix with model parameters
        z_testPred = XY_test @ beta
        z_trainPred = XY_train @ beta

        MSE_test[index] = mean_squared_error(z_test_1d, z_testPred)
        R2_test[index] = r2_score(z_test_1d, z_testPred)
        MSE_train[index] = mean_squared_error(z_train_1d, z_trainPred)
        R2_train[index] = r2_score(z_train_1d, z_trainPred)
        # beta[index, :] = beta
        # print(R2[index])

        tot_MSE_estimate_test += MSE_test[index]
        tot_R2_estimate_test += R2_test[index]
        tot_MSE_estimate_train += MSE_train[index]
        tot_R2_estimate_train += R2_train[index]
        # print(tot_MSE_estimate)

        index += 1

    # print(MSE)
    # print(tot_MSE_estimate)
    tot_MSE_estimate_test /= k
    tot_R2_estimate_test /= k
    tot_MSE_estimate_train /= k
    tot_MSE_estimate_train /= k
    # print(tot_MSE_estimate)
    # print(tot_R2_estimate)
    # print(np.argmax(R2))

    return tot_MSE_estimate_test, tot_MSE_estimate_train
