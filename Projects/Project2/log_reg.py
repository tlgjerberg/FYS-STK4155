import numpy as np
from sklearn import datasets


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(20, noise=0.20)
    return X, y


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def WMatrix(p):

    w = p * (1 - p)
    W = np.diag(w)

    return W


def delta_cross_entropy(X, y, p):

    C = X.T.dot(y - p)

    return C


def cross_entropy():


def StocGradDescent(X, y, M, n_epochs):
    n = y.shape
    m = n // M
    B = np.zeros((m, n))
    By = np.zeros(m)

    j = 0
    for epoch in range(1, n_epochs + 1):

        for i in range(m):
            k = np.random.randint(m)
            B[i] = X[k]
            By[i] = y[k]

        p = softmax(By)
        C = delta_cross_entropy(B, By, p)

        j += 1


def LogisticRegression(X, y):
    p = softmax(X)
    eps = 1e-14
    while np.abs(beta_new - beta) > eps:
        beta_new = beta - delta_cross_entropy(X, y, p)


def main():
    X, y = generate_data()
    # print(X)
    # print(y)
    p = sigmoid(X)
    # print(p)
    W = WMatrix(p)
    print(W)


if __name__ == "__main__":
    main()
