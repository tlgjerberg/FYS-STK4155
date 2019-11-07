from NeuralNetNew import *
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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


NN = NeuralNetwork(
    [100, 50, 10],
    ["sigmoid", "sigmoid", "softmax"],
    XTrain,
    yTrain_onehot)

eta_vals = np.logspace(-5, 1, 7)
lmda_vals = np.logspace(-5, 1, 7)

# grid search
for eta in eta_vals:
    for lmda in lmda_vals:
        NN.train(100, eta=eta, lmda=lmda)

        test_predict = NN.predict(XTest)

        print("Learning rate  = ", eta)
        print("Lambda = ", lmda)
        print("Accuracy score on test set: ",
              accuracy_score(yTest_onehot, test_predict))

sns.set()

train_accuracy = np.zeros((len(eta_vals), len(lmda_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmda_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmda_vals)):

        train_pred = NN.predict(XTrain)
        test_pred = NN.predict(XTest)

        train_accuracy[i][j] = accuracy_score(yTrain_onehot, train_pred)
        test_accuracy[i][j] = accuracy_score(yTest_onehot, test_pred)


fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
