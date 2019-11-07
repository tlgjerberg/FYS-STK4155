import numpy as np
from sklearn.metrics import accuracy_score


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

    def _architecture(self):
        self.weights[0] = np.random.randn(
            self.X_data.shape[1], self.layer_sizes[0])
        self.biases[0] = np.random.randn(self.layer_sizes[0])

        for l in range(1, self.num_layers + 1):
            self.weights[l] = np.random.randn(
                self.layer_sizes[l - 1], self.layer_sizes[l])
            self.biases[l] = 0.1 * np.random.randn(self.layer_sizes[l])

    def activation_function(self, z, activation="sigmoid"):
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-z))

        if activation == "tanh":
            return np.tanh(z)

        if activation == "softmax":
            exps = np.exp(z)
            return exps / np.sum(exps)

    def activation_grad(self, a, activation="sigmoid"):
        if activation == "sigmoid":
            return a * (1 - a)

        if activation == "tanh":
            return 1 - a**2

        if activation == "softmax":
            return 1

    def FeedForward(self):

        activation = self.X
        self.act = [self.X]
        # self.zList = []
        self.targets = []
        for l in range(self.num_layers + 1):
            z = np.matmul(activation, self.weights[l]) + self.biases[l]
            # print(activation)
            activation = self.activation_function(z, self.act_funcs[l])
            # self.zList.append(z)
            self.act.append(activation)

        # for b, w in zip(self.biases, self.weights):
        #     z = np.matmul(activation, w) + b
        #     activation = self._sigmoid(z)
        #     self.zList.append(z)
        #     self.act.append(activation)

    def BackPropagation(self):

        # feedforward
        self.FeedForward()

        # Gradient arrays
        grad_w = np.empty(self.num_layers, dtype=np.ndarray)
        grad_b = np.empty(self.num_layers, dtype=np.ndarray)
        # Output layer
        delta = (self.act[-1] - self.y) * self.activation_grad(self.act[-1])

        grad_b[-1] = np.sum(delta, axis=0)
        grad_w[-1] = np.matmul(self.act[-2].T, delta)

        # Hidden layers
        for l in range(2, self.num_layers + 1):
            # z = self.zList[-l]
            delta = (np.matmul(delta, self.weights[-l + 1].T)
                     * self.activation_grad(self.act[-l]))

            grad_b[-l] = np.sum(delta, axis=0)
            grad_w[-l] = np.matmul(self.act[-l - 1].T, delta)

        # Gradient descent
        for l in range(self.num_layers):
            if self.lmda > 0.0:
                self.weights[l + 1] += self.lmda * grad_w[l]
                self.biases[l + 1] += self.lmda * grad_b[l]

            self.weights[l + 1] -= self.eta * (grad_w[l] / self.batch_size)
            self.biases[l + 1] -= self.eta * (grad_b[l] / self.batch_size)

    def getMiniBatches(self, data_indices, batch_size):

        random_indcs = np.random.choice(
            data_indices, size=batch_size, replace=True)

        return random_indcs

    def train(self, n_iters=100, eta=1e-4, lmda=0.01, epochs=20, batch_size=100):
        self.eta = eta
        self.lmda = lmda
        self.batch_size = batch_size

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
        y = np.zeros_like(self.act[-1])
        y[np.arange(len(self.act[-1])), self.act[-1].argmax(1)] = 1.0
        # y[[np.where(a == np.max(a))] = 1]
        return y

    def accuracy_score_self(self, Y_test, Y_pred):
        print(len(Y_test))
        print(np.sum(Y_test == Y_pred))
        return np.sum(Y_test == Y_pred) / len(Y_test)

    def accuracy_score(self, Y_test, Y_pred):
        return accuracy_score(Y_test, Y_pred)
