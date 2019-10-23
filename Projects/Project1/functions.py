import numpy as np
from numpy import linalg
from random import random, seed
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from skimage.io import imread, imshow


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


# def MatrixInvSVD(X):
#
#     u, s, v = np.linalg.svd(X)
#
#     uT = u.transpose()
#     vT = v.transpose()
#
#     XInv = np.dot(vT, np.dot(np.diag(s**-1), uT))
#
#     return XInv


def OLS_coeff(X, z):

    XT = X.T
    X2 = XT.dot(X)
    X2inv = linalg.pinv(X2)
    beta = X2inv.dot(XT).dot(z)

    return beta


def Ridge_coeff(X, z, lmd):

    if lmd != 0:
        X = X - np.mean(X)

    XT = X.T
    X2 = XT.dot(X)
    I = np.eye(np.size(X2, 0))
    X2inv = linalg.pinv((X2 + (lmd * I)))
    beta = X2inv.dot(XT).dot(z)
    # print(beta)

    return beta


def KFoldCrossValidation(x, y, z, k, p, Poly_max, lmd, method='Ridge'):
    """
    K-fold cross validation of data (x,y) and z with k folds and polynomial
    degree p
    """

    # KFold instance
    kfold = KFold(n_splits=k, shuffle=True)

    beta = np.zeros((k, int((p + 1) * (p + 2) / 2)))
    R2_test = MSE_test = 0
    R2_train = MSE_train = 0
    bias_test = var_test = 0

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

        # Setting up the design matrices for training and test data
        X_train = CreateDesignMatrix_X(x_train, y_train, n=p)
        X_test = CreateDesignMatrix_X(x_test, y_test, n=p)

        if method == 'Lasso':

            # Lasso regression
            clf_lasso = Lasso(alpha=lmd).fit(X_train, z_train_1d)
            z_testPred = clf_lasso.predict(X_test)
            z_trainPred = clf_lasso.predict(X_train)

        else:

            beta = Ridge_coeff(X_train, z_train_1d, lmd)

            # Computing modelfrom design matrix and model parameters
            z_testPred = X_test @ beta
            z_trainPred = X_train @ beta

            # padding_size = (Poly_max - p)
            # print(padding_size)

            # print(beta)
            # beta_padded = np.pad(beta, padding_size)
            # print(beta_padded)

        # Finding MSE and R2 scores with both training and test data
        MSE_test += mean_squared_error(z_test_1d, z_testPred)
        R2_test += r2_score(z_test_1d, z_testPred)
        MSE_train += mean_squared_error(z_train_1d, z_trainPred)
        R2_train += r2_score(z_train_1d, z_trainPred)
        bias_test += np.mean((z_test_1d - np.mean(z_testPred))**2)
        var_test += np.var(z_testPred)

    # Dividing the sum of MSE, bias and variance to find mean of all
    bias_test /= k
    var_test /= k

    MSE_test /= k
    R2_test /= k
    MSE_train /= k
    R2_train /= k

    return MSE_test, MSE_train, var_test, bias_test


def plot3D(x, y, z, z_predict):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    surf_pred = ax.plot_surface(x, y, z_predict, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('% .02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def DataImport(filename):
    """
    Imports terraindata...
    """

    # Load the terrain
    terrain1 = imread(filename)
    # Show the terrain

    downscaled = np.zeros(
        (int(len(terrain1[:, 0]) / 20), int(len(terrain1[0, :]) / 20)))
    downscaled = terrain1[0::20, 0::20]

    plt.figure()
    plt.imshow(terrain1, cmap='gray')
    plt.figure()

    plt.imshow(downscaled, cmap='gray')

    return terrain1, downscaled


def OLSPredict(x, y, z, poly):

    X = CreateDesignMatrix_X(x, y)

    beta = OLS_coeff(X, z)

    z_predict = X @ beta

    return z_predict
