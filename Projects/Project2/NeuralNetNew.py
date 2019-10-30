import numpy as np


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        exps = np.exp(z)
        return exps / np.sum(exps)

    def FeedForward(self):

        self.act = []
        self.zList = []
        self.targets = []

        for b, w in zip(self.biases, self.weights):
            z = np.matmul(self.X @ self.weights) + self.biases
            a = _sigmoid(z)
            self.zList.append(z)
            self.act.append(a)
            selt.targets.append(_softmax(z))
        # Find error delta
          # Target t

    def BackPropagation(self):

        # feedforward
        FeedForward()

        # Output layer
        delta_out = (self.act[-1] - self.y) * \
            self.act[-1] * (1 - self.act[-1])

        grad_b_out = np.sum(delta_out, axis=0)
        grad_w_out = np.matmul(self.act[-l].T, delta_out)

        for l in range(2, self.num_layers):
            z = zList[-l]
            delta_hidden = np.matmul(self.weights[-l + 1].T, delta_out)
            * self.act[-l] * (1 - self.act[-l])

            grad_b_hidden = np.sum(delta_hidden, axis=0)
            grad_w_hidden = np.matmul(delta_hidden, self.act[-l - 1].T)

    def SDG(self, X, y, eta=0.01, epochs=10):
        self.X = X
        self.y = y

        train_data = np.hstack(X, y)

        for i in range(epochs):

            BackPropagation()
