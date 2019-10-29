import numpy as np


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        exps = np.exp(z)
        return exps / np.sum(exps)

    def FeedForward(self):

        self.activations = []
        self.zList = []
        self.targets = []

        for b, w in zip(self.biases, self.weights):
            z = np.matmul(self.X @ self.weights) + self.biases
            a = _sigmoid(z)
            self.zList.append(z)
            self.activations.append(a)
            selt.targets.append(_softmax(z))
        # Find error delta
          # Target t

    def BackPropagation(self):

        # feedforward
        FeedForward()

        # Output layer
        delta_out = (self.activation[-1] - self) * \
            self.activation[-1] * (1 - self.activation[-1])

        nabla_b_out = np.sum(delta_out, axis=0)
        nabla_w_out = np.matmul(self.a.T, delta_out)

        delta_hidden = np.matmul(self.a.T, delta_out) * self.a * (1 - self.a)

        nabla_b_hidden = np.sum(delta_hidden, axis=0)
        nabla_w_hidden = np.matmul(self.X.T, delta_hidden)

    def SDG(self, X, y, eta=0.01, epochs=10):
        self.X = X
        self.y = y

        train_data = np.hstack(X, y)

        for i in range(epochs):

            BackPropagation()
