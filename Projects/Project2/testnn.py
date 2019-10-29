from NeuralNetNew import *
from sklearn.datasets import load_breast_cancer

NN = NeuralNetwork([7, 3, 2])

print(NN.biases)
print(NN.weights)
print(NN.sizes)

data = load_breast_cancer(return_X_y=True)

print(data)
