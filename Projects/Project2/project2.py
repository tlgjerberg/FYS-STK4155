import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
from NeuralNet import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

np.random.seed(0)

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0,
                   index_col=0, na_values=nanDict)

df.rename(index=str,
          columns={"default payment next month": "defaultPaymentNextMonth"},
          inplace=True)


# Remove instances with zeros only for past bill statements or paid amounts
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0) &
                (df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

#Features and targets
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

# print(X)
# print(y)

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto", sparse=False)


X = ColumnTransformer(
    [("", onehotencoder, [1, 2, 3, 5, 6, 7, 8, 9, 10]), ],
    remainder="passthrough"
).fit_transform(X)

# Train-test split
trainingShare = 0.7
seed = 1
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y,
    train_size=trainingShare,
    test_size=1 - trainingShare,
    random_state=seed)

# print(XTrain)

# Input Scaling
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

# One-hot's of the target vector
Y_train_onehot = onehotencoder.fit_transform(yTrain)
Y_test_onehot = onehotencoder.transform(yTest)


NN = NeuralNetwork([2, 10, 5])

NN.SGD(XTrain, Y_train_onehot, 10, 4)

NN.evaluate(XTest, Y_test_onehot)
