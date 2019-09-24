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


def OLS(X, z):

    XT = X.T
    X2 = XT.dot(X)
    X2inv = MatrixInvSVD((X2))
    beta = X2inv.dot(XT).dot(z)

    return beta


def Ridge(X, z, lmd):

    XT = X.T

    X2 = XT.dot(X)
    I = np.eye(np.size(X2, 0))
    # X2inv = MatrixInvSVD((X2 - lmd * I))
    X2inv = linalg.pinv((X2 - lmd * I))
    beta = X2inv.dot(XT).dot(z)
    return beta


# def Lasso(X, z, lmd):
#
#     clf_lasso = skl.Lasso(alpha=lmb).fit(X, z)
#     z_lasso = clf_lasso.predict(X)
#
#     return beta


def KFoldCrossValidation(x, y, z, k, p, lmd, method='OLS'):
    """
    K-fold cross validation of data (x,y) and z with k folds and polynomial
    degree p. Returns the best R2 score.
    """

    # KFold instance
    kfold = KFold(n_splits=k, shuffle=True)

    MSE_test = np.zeros(k)
    MSE_train = np.zeros(k)
    R2_test = np.zeros(k)
    R2_train = np.zeros(k)
    beta = np.zeros((k, int((p + 1) * (p + 2) / 2)))
    tot_R2_test = tot_MSE_test = 0
    tot_R2_train = tot_MSE_train = 0
    z_pred = []
    z_test_lst = []

    index = 0
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

        # Subtracting the mean from the training and test data to avoid
        # penalty to intercept
        x_train_mean = np.mean(x_train)
        y_train_mean = np.mean(y_train)
        z_train_mean = np.mean(z_train)

        # Setting up the design matrices for training and test data
        X_train = CreateDesignMatrix_X(x_train, y_train, n=p)
        X_test = CreateDesignMatrix_X(x_test, y_test, n=p)

        beta = Ridge(X_train, z_train_1d, lmd)

        # Computing modelfrom design matrix and model parameters
        z_testPred = X_test @ beta
        z_trainPred = X_train @ beta

        # Finding MSE and R2 scores with both training and test data
        MSE_test[index] = mean_squared_error(z_test_1d, z_testPred)
        R2_test[index] = r2_score(z_test_1d, z_testPred)
        MSE_train[index] = mean_squared_error(z_train_1d, z_trainPred)
        R2_train[index] = r2_score(z_train_1d, z_trainPred)

        z_pred.append(z_testPred)
        z_test_lst.append(z_test_1d)

        tot_MSE_test += MSE_test[index]
        tot_R2_test += R2_test[index]
        tot_MSE_train += MSE_train[index]
        tot_R2_train += R2_train[index]

        # print(tot_MSE_estimate)

        index += 1

    # print(MSE)
    # print(tot_MSE_estimate)
    bias_test = np.mean((z_test - np.mean(z_pred))**2)
    var_test = np.mean(np.var(z_pred))
    tot_MSE_test /= k
    tot_R2_test /= k
    tot_MSE_train /= k
    tot_MSE_train /= k

    return tot_MSE_test, tot_MSE_train, var_test, np.sqrt(bias_test)
