from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy import linalg
from random import random, seed
from functions import *  # Importing selfmade functions
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

"""
Part a)
"""

n = 100
noise_scale = 0.5
poly_deg = 20  # Choice of number of polynomial degrees to test for


np.random.seed(131)


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
# Noise matrix
eps = noise_scale * np.random.normal(0, 1, 400).reshape(np.size(x), np.size(y))
x, y = np.meshgrid(x, y)


z = FrankeFunction(x, y, eps)


zpredict = OLSPredict(x, y, np.ravel(z), 5)

# Finding MSE and R**2 score
MSE = MSE(np.ravel(z), zpredict)
var = np.var(zpredict)
R2 = R2(np.ravel(z), zpredict)

# plot3D(x, y, z, zpredict)

"""
Part b)
"""

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x, y, z, test_size=0.2)

k = 5  # Number of folds used in cross validation


# beta = np.zeros((poly_deg, int((poly_deg + 1) * (poly_deg + 2) / 2)))
MSE_test = np.zeros(poly_deg)
MSE_train = np.zeros(poly_deg)
Var_OLS = np.zeros(poly_deg)
Bias_OLS = np.zeros(poly_deg)
lmd = 0

# print(x)

for p in range(poly_deg):
    MSE_test[p], MSE_train[p], Var_OLS[p], Bias_OLS[p] = KFoldCrossValidation(
        x_train, y_train, z_train, k, p, poly_deg, lmd)


P_plt = np.linspace(0, poly_deg, poly_deg)

"""
Part c)
"""

plt.figure()
plt.plot(P_plt, MSE_test)
plt.plot(P_plt, MSE_train)
plt.plot(P_plt, Var_OLS)
plt.plot(P_plt, Bias_OLS)
plt.legend(['Test_MSE', 'Train_MSE', 'Var', 'Bias'])
plt.title('OLS Error')

"""
Part d
"""

# nlambdas = 100
# lambdas = np.linspace(-1, 0, nlambdas)
# lambdas = 0.001
# # lmd = 0.001
#
# MSE_ridge_test = np.zeros(poly_deg)
# MSE_ridge_train = np.zeros(poly_deg)
# Var_ridge = np.zeros(poly_deg)
# Bias_ridge = np.zeros(poly_deg)
#
# for p in range(poly_deg):
#     MSE_ridge_test[p], MSE_ridge_train[p], Bias_ridge[p], Var_ridge[p] = KFoldCrossValidation(
#         x, y, z, k, p, poly_deg, lambdas)
#
#
# # plt.figure()
# # plt.plot(P_plt, MSE_ridge_test)
# # plt.plot(P_plt, MSE_ridge_train)
# # plt.legend(['Test', 'Train'])
# #
# plt.figure()
# plt.plot(P_plt, MSE_ridge_test)
# plt.plot(P_plt, Var_ridge)
# plt.plot(P_plt, Bias_ridge)
# plt.legend(['MSE', 'Var', 'Bias'])
# plt.title('Ridge Error')

"""
Part e
"""

# gamma = 1e-4
#
# MSE_lasso_test = np.zeros(poly_deg)
# MSE_lasso_train = np.zeros(poly_deg)
# Var_lasso = np.zeros(poly_deg)
# Bias_lasso = np.zeros(poly_deg)
#
# for p in range(poly_deg):
#     MSE_lasso_test[p], MSE_lasso_train[p], Bias_lasso[p], Var_lasso[p] = KFoldCrossValidation(
#         x, y, z, k, p, gamma, method='Lasso')
#
# plt.figure()
# plt.plot(P_plt, MSE_lasso_test)
# plt.plot(P_plt, Var_lasso)
# plt.plot(P_plt, Bias_lasso)
# plt.legend(['MSE', 'Var', 'Bias'])
# plt.title('Lasso Error')


"""
Part f
"""

# terraindata = 'Norway_1arc.tif'
#
# full, downscaled = DataImport(terraindata)

"""
Part g
"""

# z = downscaled
#
# # print(z.dtype)
#
# x_dim_size = np.size(z, 1)
# y_dim_size = np.size(z, 0)
#
#
# x_dim = np.linspace(0, 1, x_dim_size)
# y_dim = np.linspace(0, 1, y_dim_size)
#
# x_dim, y_dim = np.meshgrid(x_dim, y_dim)
#
#
# MSE_terrain_test = np.zeros(poly_deg)
# MSE_terrain_train = np.zeros(poly_deg)
# Var_terrain = np.zeros(poly_deg)
# Bias_terrain = np.zeros(poly_deg)
#
# lmd = 0
#
# for p in range(poly_deg):
#
#     MSE_terrain_test[p], MSE_terrain_train[p], Var_terrain[p], Bias_terrain[p] = KFoldCrossValidation(
#         x_dim, y_dim, z, k, p, lmd)
#
#
# poly_best = np.argmax(MSE_terrain_test)
#
# print(poly_best)


plt.show()
