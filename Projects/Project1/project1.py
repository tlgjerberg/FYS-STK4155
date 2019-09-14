from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy import linalg
from random import random, seed
from functions import *  # Importing selfmade functions
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

"""
Part a)
"""

np.random.seed(111)

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
print(np.size(X2inv))

# Finds model parameters
beta = X2inv.dot(X.T).dot(np.ravel(z))
# print(beta)

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

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    x, y, z, test_size=0.2)

k = 5
P = 20

# beta = np.zeros((P, int((P + 1) * (P + 2) / 2)))
MSE_cv = np.zeros(P)
R2_cv = np.zeros(P)

for p in range(P):
    MSE_cv[p], R2_cv[p] = KFoldCrossValidation(x_train, y_train, z_train, k, p)

# print(MSE_cv)
# print(R2_cv)

P_plt = np.linspace(0, P, P)
plt.figure()
plt.plot(P_plt, MSE_cv)
plt.plot(P_plt, R2_cv)
plt.legend(['MSE', 'R2'])
plt.show()

"""
Part c)
"""
