import numpy as np


class NeuralNetwork:
    def __init__(self, sizes, eta=0.01):
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

    # def BackPropagation(self):
    #     # error_hidden = self._outputError()
    #     error_output = self.probabilities - self.Y_data
    #     error_hidden = np.matmul(
    #         error_output, self.output_weights.T)
    #     * self.a_h * (1 - self.a_h)
    #
    #     self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
    #     self.output_bias_gradiant = np.sum(error_output, axis=0)
    #
    #     if self.lmbd > 0:
    #         self.output_weights_gradient += self.lmbd * self.output_weights
    #         self.hidden_weights_gradient += self.lmbd * self.hidden_weights
    #
    #     self.output_weights -= self.eta * self.output_weights_gradient
    #     self.output_bias -= self.eta * self.output_bias_gradient
    #     self.hidden_weights -= self.eta * self.hidden_weights_gradient
    #     self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def FeedForward(self, a):

        z_h = X @ self.weights + self.biases
        a_h = _sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        self.probabilities = self._softmax(z_o)

    def feedforwardOut(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, X, y, epochs, mini_batch_size):

        train_data = np.hstack((X, y))
        n = len(train_data)

        for i in range(epochs):
            np.random.shuffle(train_data)
            mini_batches = [
                train_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                print(mini_batch)
                self.update_mini_batch(mini_batch, self.eta)

            # if test_data:
            #     print "Epoch {0}: {1} / {2}".format(
            #         j, self.evaluate(test_data), n_test)
            # else:
            #     print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

        def backprop(self, x, y):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            # list to store all the activations, layer by layer
            activations = [x]
            zs = []  # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            # backward pass
            delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())

            for l in xrange(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            return (nabla_b, nabla_w)

    def evaluate(self, X, y):

        test_data = np.hstack(X, y)

        test_results = [(np.argmax(self.feedforwardOut(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):

        return (output_activations - y)
