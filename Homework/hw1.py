import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 2a)

x = np.random.rand(100, 1)
y = 5 * x * x + 0.1 * np.random.randn(100, 1)

# X = np.zeros((100, 3))
# X[:, 0] = 1
# X[:, 1] = x
# X[:, 2] = x*x
X = x * x


beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


print(beta)
xnew = np.linspace(0, 1, 100)

plt.plot(x, y, 'ro')
plt.plot(xnew, beta[0] * xnew**2, 'y')
# print(beta*X)

# 2b)

# print('break')

Xnew = xnew[:, np.newaxis]**2

linreg = LinearRegression()
linreg.fit(X, y)
ypredict = linreg.predict(Xnew)

print(linreg.coef_)

# print(ypredict)
plt.plot(xnew, ypredict, 'b')
plt.legend(['', 'numpy', 'sklearn'])

# 2c)


plt.show()
