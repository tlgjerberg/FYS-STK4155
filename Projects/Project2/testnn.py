from NeuralNetNew import *
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

# X, y = load_breast_cancer(return_X_y=True)
X, y = load_digits(return_X_y=True)

# Train-test split
trainingShare = 0.7
seed = 1
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y,
    train_size=trainingShare,
    test_size=1 - trainingShare,
    random_state=seed)

# print(X.shape[1])
# print(y.shape)

NN = NeuralNetwork([9, 5, 10], XTrain, yTrain)

# print(NN.num_layers)
# print("biases", NN.biases)
# print("weights", NN.weights)
# print(NN.layer_sizes)


NN.MBSDG(XTrain, yTrain)

NN.predict(XTest)
