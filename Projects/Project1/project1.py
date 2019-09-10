from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy import linalg
from random import random, seed
from functions import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

"""
Part a)
"""

np.random.seed(1)

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
# Noise matrix
eps = 0.1 * np.random.normal(0, 1, 400).reshape(np.size(x), np.size(y))
x, y = np.meshgrid(x, y)


z = FrankeFunction(x, y, eps)

X = CreateDesignMatrix_X(x, y)

# print(X)
# print(np.size(X, 0))
# print(np.size(X, 1))


# X2 = X.T.dot(X)
#
# u, s, v = linalg.svd(X2)
#
# X2inv = u.dot((s.dot(v)))
#
# beta = (X2inv).dot(X.T).dot(y)

# Inverting the matrix transpose matrix multiplication
X2inv = np.linalg.inv(X.T.dot(X))

# Finds model parameters
beta = X2inv.dot(X.T).dot(np.ravel(z))

# Setting up the model for comparing with real data
zpredict = X.dot(beta)

# Finding MSE and R**2 score
MSE = MSE(np.ravel(z), zpredict)
R2 = R2(np.ravel(z), zpredict)

# print(MSE, R2)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('% .02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


"""
Part b)
"""

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z)

"""
K-fold cross validation
"""


def KFoldCrossValidation(x, y, z, k, p):

    # KFold instance
    kfold = KFold(n_splits=k)

    MSE = np.zeros(k)
    R2 = np.zeros(k)
    index_count = 0

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
        XY_train_cv = CreateDesignMatrix_X(x_train_cv, y_train_cv, p)
        XY_test_cv = CreateDesignMatrix_X(x_test_cv, y_test_cv, p)

        # Inverting
        XY2_cv_inv = np.linalg.inv(XY_train_cv.T.dot(XY_train_cv))

        # Model parameters beta based on training data design matrix
        beta_cv = XY2_cv_inv.dot(XY_train_cv.T).dot(z_train_cv_1d)

        # Combining test design matrix with model parameters
        z_testModel_cv = XY_test_cv @ beta_cv

        MSE[index_count] = mean_squared_error(z_test_cv_1d, z_testModel_cv)
        R2[index_count] = r2_score(z_test_cv_1d, z_testModel_cv)
        index_count += 1

    return beta_cv[np.max(R2[index_count - 1])]


for p in range(20):
    print(KFoldCrossValidation(x, y, z, 20, p))

    # print(MSE_cv)
    # print(R2_cv)
