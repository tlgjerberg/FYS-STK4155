from NeuralNetNew import *
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# X, y = load_breast_cancer(return_X_y=True)
data = load_digits()
X, y = load_digits(return_X_y=True)

y = y.reshape(len(y), 1)

# Train-test split
trainingShare = 0.7
seed = 1
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y,
    train_size=trainingShare,
    test_size=1 - trainingShare,
    random_state=seed)


# Normalizing data
XTrainmax = np.max(XTrain)
XTrain /= XTrainmax

XTestmax = np.max(XTest)
XTest /= XTestmax

onehotencoder = OneHotEncoder(categories="auto", sparse=False)

yTrain_onehot = onehotencoder.fit_transform(yTrain)
yTest_onehot = onehotencoder.transform(yTest)


NN = NeuralNetwork([9, 5, 8, 10], ["sigmoid", "sigmoid", "sigmoid", "softmax"],
                   XTrain, yTrain_onehot)

# print(NN.num_layers)
# print("biases", NN.biases)
# print("weights", NN.weights[0].shape, NN.weights[1].shape,
#       NN.weights[2].shape, NN.weights[3].shape)
# print(NN.layer_sizes)


NN.MBGD(300)

Y_pred = NN.predict(XTest)
# print(Y_pred)
# print(yTest_onehot)

acc = NN.accuracy_score(yTest_onehot, Y_pred)

# print(acc)
