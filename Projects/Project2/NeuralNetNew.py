import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, seed=0):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = np.empty(self.num_layers, dtype=np.ndarray)
        self.biases = np.empty(self.num_layers, dtype=np.ndarray)

        np.random.seed(seed)
        self._architecture()

    def _architecture(self):
        # self.weights[0] = np.random.randn(self.X.shape[1], self.layer_sizes[0])
        # self.biases[0] = np.zeros(sizes[0]) + 0.01

        for l in range(1, self.num_layers):
            self.weights[l] = np.random.randn(
                self.layer_sizes[l - 1], self.layer_sizes[l])
            self.biases = np.random.randn(1, self.layer_sizes[l])

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        exps = np.exp(z)
        return exps / np.sum(exps)

    def activation_function():
        pass

    def FeedForward(self):

        self.act = [self.X]
        self.zList = []
        self.targets = []

        for b, w in zip(self.biases, self.weights):
            # print("weights", w.T.shape)
            # print("X", self.X.shape)
            # print("biases", b.shape)
            z = np.matmul(self.X, w) + b
            print(z.shape)
            a = self._sigmoid(z)
            self.zList.append(z)
            self.act.append(a)

    def FeedForwardOut(self):
        pass

    def BackPropagation(self):

        # feedforward
        self.FeedForward()

        # Output layer
        delta_out = (self.act[-1] - self.y) * \
            self.act[-1] * (1 - self.act[-1])

        grad_b_out = np.sum(delta_out, axis=0)
        grad_w_out = np.matmul(self.act[-l].T, delta_out)

        for l in range(2, self.num_layers):
            z = zList[-l]
            delta_hidden = (np.matmul(
                self.weights[-l + 1].T, delta_out)
                * self.act[-l] * (1 - self.act[-l]))

            grad_b_hidden = np.sum(delta_hidden, axis=0)
            grad_w_hidden = np.matmul(delta_hidden, self.act[-l - 1].T)

    def getMiniBatches(self, data_indices, batch_size):

        random_indcs = np.random.choice(
            data_indices, size=batch_size, replace=False)

        return random_indcs

    def MBSDG(self, X, y, n_iters=100, eta=1e-4, epochs=10, batch_size=10):
        self.X_full = X
        self.y_full = y
        print("full x", self.X_full.shape)

        data_indices = np.arange(len(self.y_full))

        for i in range(epochs):
            for j in range(n_iters):

                random_indcs = self.getMiniBatches(data_indices, batch_size)

                self.X = self.X_full[random_indcs]
                self.y = self.y_full[random_indcs]

                self.BackPropagation()
