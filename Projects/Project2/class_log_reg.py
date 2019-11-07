

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000,
                 fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, Xb):
        return 1 / (1 + np.exp(-Xb))

    def _loss(self, p, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def _gradient(self, X, y, p):
        return X.T.dot(y - p)

    # def GradientDescent(self, X, y, p):
    #     grad = X.T.dot(y - p)
    #     return lr * grad

    def StochasitcGradDescent():
        pass

    def create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        return

    def GD(self, X, y):

        if self.fit_intercept:
            X = self._add_intercept(X)

        # Initialize weights
        self.theta = theta = np.random.randn(X.shape[1], 1)

        for i range(self.num_iter):
            Xb = np.dot(X, self.theta)
            p = self._sigmoid(Xb)
            gradient = _gradient(X, y, p)
            self.theta = self.lr * gradient

            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

    def SGD(self, X, y, batch_size=1):

        if self.fit_intercept:
            X = self._add_intercept(X)

        # Initialize weights
        self.theta = np.random.randn(X.shape[1], 1)

        for i range(self.num_iter):
            Xb = np.dot(X, self.theta)
            p = self._sigmoid(Xb)

            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

        return
