from NeuralNetNew import *
from sklearn.datasets import load_breast_cancer


X, y = load_breast_cancer(return_X_y=True)

# print(X.shape[1])
# print(y.shape)

NN = NeuralNetwork([y.size, 10, 5], X.shape[1], X.shape[0])


# print("biases", NN.biases.shape)
# print("weights", NN.weights.shape)
# print(NN.layer_sizes)


NN.MBSDG(X, y)
