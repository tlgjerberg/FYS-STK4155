import numpy as np


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def delta_cross_entropy(X, y, p):

    C = X.T.dot(y - p)

    return C


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


def main():
    X, y = data
    p = softmax(X)


if __name__ == "__main__":
    main()
