
class Regression(object):
    """docstring for ."""

    def __init__(self, x, y, z):
        super(, self).__init__()
        self.x = x
        self.y = y
        self.z = z

    def CreateDesignMatrix_X():
        """
        Function for creating a design X-matrix with
        rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh,
        keyword agruments n is the degree of the polynomial you want to fit.
        """
        if len(self.x.shape) > 1:
            x = np.ravel(self.x)
            y = np.ravel(self.y)

        N = len(x)
        l = int((n + 1) * (n + 2) / 2)		# Number of elements in beta
        X = np.ones((N, l))

        for i in range(1, n + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x**(i - k) * y**k

        return X

    def KFoldCrossValidationOLS(x, y, z, k, p):
        """
        K-fold cross validation of data (x,y) and z with k folds and polynomial
        degree p. Returns the best R2 score.
        """

        # KFold instance
        kfold = KFold(n_splits=k, shuffle=True)

        MSE_test = np.zeros(k)
        MSE_train = np.zeros(k)
        R2_test = np.zeros(k)
        R2_train = np.zeros(k)
        beta = np.zeros((k, int((p + 1) * (p + 2) / 2)))
        tot_R2_estimate_test = tot_MSE_estimate_test = 0
        tot_R2_estimate_train = tot_MSE_estimate_train = 0
        index = 0

        for train_ind, test_ind in kfold.split(x):

            # Assigning train and test data
            x_train = x[train_ind]
            x_test = x[test_ind]

            y_train = y[train_ind]
            y_test = y[test_ind]

            z_train = z[train_ind]
            z_test = z[test_ind]

            # Raveling z data into 1D arrays
            z_train_1d = np.ravel(z_train)
            z_test_1d = np.ravel(z_test)

            # Setting up the design matrices for training and test data
            XY_train = CreateDesignMatrix_X(x_train, y_train, n=p)
            XY_test = CreateDesignMatrix_X(x_test, y_test, n=p)

            # Computing beta from the design matrix and
            beta = OLS(XY_train, z_train_1d)

            # Computing modelfrom design matrix and model parameters
            z_testPred = XY_test @ beta
            z_trainPred = XY_train @ beta

            # Finding MSE and R2 scores with both training and test data
            MSE_test[index] = mean_squared_error(z_test_1d, z_testPred)
            R2_test[index] = r2_score(z_test_1d, z_testPred)
            MSE_train[index] = mean_squared_error(z_train_1d, z_trainPred)
            R2_train[index] = r2_score(z_train_1d, z_trainPred)
            # beta[index, :] = beta
            # print(R2[index])

            tot_MSE_estimate_test += MSE_test[index]
            tot_R2_estimate_test += R2_test[index]
            tot_MSE_estimate_train += MSE_train[index]
            tot_R2_estimate_train += R2_train[index]
            # print(tot_MSE_estimate)

            index += 1

        # print(MSE)
        # print(tot_MSE_estimate)
        tot_MSE_estimate_test /= k
        tot_R2_estimate_test /= k
        tot_MSE_estimate_train /= k
        tot_MSE_estimate_train /= k

        return tot_MSE_estimate_test, tot_MSE_estimate_train

        def KFoldCrossValidationOLS(x, y, z, k, p):
            """
            K-fold cross validation of data (x,y) and z with k folds and polynomial
            degree p. Returns the best R2 score.
            """

            # KFold instance
            kfold = KFold(n_splits=k, shuffle=True)

            MSE_test = np.zeros(k)
            MSE_train = np.zeros(k)
            R2_test = np.zeros(k)
            R2_train = np.zeros(k)
            beta = np.zeros((k, int((p + 1) * (p + 2) / 2)))
            tot_R2_test = tot_MSE_test = 0
            tot_R2_train = tot_MSE_train = 0
            z_pred = []
            z_test = []
            index = 0

            for train_ind, test_ind in kfold.split(x):

                # Assigning train and test data
                x_train = x[train_ind]
                x_test = x[test_ind]

                y_train = y[train_ind]
                y_test = y[test_ind]

                z_train = z[train_ind]
                z_test = z[test_ind]

                # Raveling z data into 1D arrays
                z_train_1d = np.ravel(z_train)
                z_test_1d = np.ravel(z_test)

                # Setting up the design matrices for training and test data
                XY_train = CreateDesignMatrix_X(x_train, y_train, n=p)
                XY_test = CreateDesignMatrix_X(x_test, y_test, n=p)

                # Computing beta from the design matrix and
                beta = OLS(XY_train, z_train_1d)

                # Computing modelfrom design matrix and model parameters
                z_testPred = XY_test @ beta
                z_trainPred = XY_train @ beta

                # Finding MSE and R2 scores with both training and test data
                MSE_test[index] = mean_squared_error(z_test_1d, z_testPred)
                R2_test[index] = r2_score(z_test_1d, z_testPred)
                MSE_train[index] = mean_squared_error(z_train_1d, z_trainPred)
                R2_train[index] = r2_score(z_train_1d, z_trainPred)
                # beta[index, :] = beta
                # print(R2[index])

                tot_MSE_test += MSE_test[index]
                tot_R2_test += R2_test[index]
                tot_MSE_train += MSE_train[index]
                tot_R2_train += R2_train[index]

            bias_test = np.mean(z_test - np.mean(z_pred))
            var_test = np.mean(np.var(z_pred))
            tot_MSE_test /= k
            tot_R2_test /= k
            tot_MSE_train /= k
            tot_R2_train /= k

            return tot_MSE_test, tot_MSE_train
