import numpy as np


class NeuralNetwork:
    def __init__(self, eta=0.01, sizes):
        self.eta = eta
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        exps = np.exp(z)
        return exps / np.sum(exps)

    def FeedForward(self, a):

        z_h = X @ self.weights + self.biases
        a_h = _sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        self.probabilities = self._softmax(z_o)

    def _hiddenError(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(
            error_output, self.output_weights.T)
        * self.a_h * (1 - self.a_h)

        return error_hidden

    def BackPropagation(self):
        # error_hidden = self._outputError()
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(
            error_output, self.output_weights.T)
        * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradiant = np.sum(error_output, axis=0)

        if self.lmbd > 0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def SGD(self, X, y, epochs, mini_batch_size):

        train_data = np.hstack((X, y))
        n = len(train_data)

        for j in range(epochs):
            np.random.shuffle(train_data)
            mini_batches = [
                train_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla
