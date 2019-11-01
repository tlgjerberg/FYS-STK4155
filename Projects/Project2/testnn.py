from NeuralNetNew import *
from sklearn.datasets import load_breast_cancer


X, y = load_breast_cancer(return_X_y=True)

# print(X.shape)
# print(y.shape)

NN = NeuralNetwork([y.size, 10, 5])


print("biases", NN.biases)
print("weights", NN.weights)
print(NN.layer_sizes)


NN.MBSDG(X, y)
