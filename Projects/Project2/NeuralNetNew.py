import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, act_funcs, X, y, seed=0):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.act_funcs = act_funcs
        self.X_data = X
        self.y_data = y
        self.act = np.empty(self.num_layers + 2, dtype=np.ndarray)
        self.weights = np.empty(self.num_layers + 1, dtype=np.ndarray)
        self.biases = np.empty(self.num_layers + 1, dtype=np.ndarray)

        np.random.seed(seed)
        self._architecture()
        print(self.weights.shape)

    def _architecture(self):
        self.weights[0] = np.random.randn(
            self.X_data.shape[1], self.layer_sizes[0])
        self.biases[0] = np.random.randn(self.X_data.shape[1])

        for l in range(1, self.num_layers + 1):
            self.weights[l] = np.random.randn(
                self.layer_sizes[l - 1], self.layer_sizes[l])
            self.biases = np.random.randn(self.layer_sizes[l])

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        exps = np.exp(z)
        return exps / np.sum(exps)

    def activation_function(self, z, activation="sigmoid"):
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-z))

        if activation == "tanh":
            return np.tanh(z)

    def activation_grad(self, a, activation="sigmoid"):
        if activation == "sigmoid":
            return a * (1 - a)

        if activation == "tanh":
            return 1 - a**2

    def FeedForward(self):

        activation = self.X
        self.act = [self.X]
        self.zList = []
        self.targets = []

        for b, w in zip(self.biases, self.weights):
            z = np.matmul(activation, w) + b
            activation = self._sigmoid(z)
            self.zList.append(z)
            self.act.append(activation)

    def BackPropagation(self):

        # feedforward
        self.FeedForward()

        # Output layer
        delta = (self.act[-1] - self.y)

        grad_b_out = np.sum(delta, axis=0)
        grad_w_out = np.matmul(self.act[-1].T, delta)

        # Hidden layers
        grad_w_hidden = np.empty(self.num_layers)
        grad_b_hidden = np.empty(self.num_layers)

        for l in range(2, self.num_layers):
            z = self.zList[-l]
            delta = (np.matmul(
                self.weights[-l + 1], delta)
                * self.act[-l] * (1 - self.act[-l]))

            grad_b_hidden[-l] = np.sum(delta, axis=0)
            grad_w_hidden[-l] = np.matmul(delta, self.act[-l - 1].T)

            # if self.lmda > 0.0:
            #     grad_w_out += self.lmda *

        # Gradient descent
        for l in range(self.num_layers):
            self.weights[l] -= self.eta * grad_w_hidden[l]
            self.biases[l] -= self.eta * grad_b_hidden[l]

    def getMiniBatches(self, data_indices, batch_size):

        random_indcs = np.random.choice(
            data_indices, size=batch_size, replace=False)

        return random_indcs

    def MBGD(self, n_iters=100, eta=1e-4, lmda=0.01, epochs=10, batch_size=10):
        self.eta = eta
        self.lmda = lmda

        data_indices = np.arange(len(self.y_data))

        for i in range(epochs):
            for j in range(n_iters):

                random_indcs = self.getMiniBatches(data_indices, batch_size)
                self.X = self.X_data[random_indcs]
                self.y = self.y_data[random_indcs]

                self.BackPropagation()

    def predict(self, X):
        self.X = X
        self.FeedForward()
        return self.act[-1]

    def accuracy_score(self, Y_test, Y_pred):
        return np.sum(Y_test == Y_pred) / len(Y_test)
