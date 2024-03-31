import numpy as np


x = np.load('data/iris/iris_features.npy')
y = np.load('data/iris/iris_labels.npy')
N = 120
x_train = x[:N]
x_test = x[N:]
y_train = y[N:]
y_test = y[:N]

print(x_train.shape)
print(y_train.shape)