from NeuralNetNew import *
from sklearn.datasets import load_breast_cancer, load_digits


# X, y = load_breast_cancer(return_X_y=True)
X, y = load_digits(return_X_y=True)

# print(X.shape[1])
# print(y.shape)

NN = NeuralNetwork([9, 5, 10], X, y)

# print(NN.num_layers)
# print("biases", NN.biases)
# print("weights", NN.weights)
# print(NN.layer_sizes)


NN.MBSDG(X, y)
