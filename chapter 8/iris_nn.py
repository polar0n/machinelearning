import numpy as np
import pickle
import sys


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def evaluate(x, y, w):
    w12, b1, w23, b2, w34, b3 = w
    nc = nw = 0
    prob = np.zeros(len(y))
    for i in range(len(y)):
        a1 = sigmoid(np.dot(x[i], w12) + b1)
        a2 = sigmoid(np.dot(a1, w23) + b2)
        prob[i] = sigmoid(np.dot(a2, w34) + b3)
        z = 0 if prob[i] > 0.5 else 1
        if z == y[i]:
            nc += 1
        else:
            nw += 1
    return [float(nc) / float(nc + nw), prob]


xtest = np.load('data/iris/iris2_test.npy')
ytest = np.load('data/iris/iris2_test_labels.npy')

weights = pickle.load(open('data/iris/iris2_weights_best.pkl', 'rb'))
score, prob = evaluate(xtest, ytest, weights)
print()

for i in range(len(prob)):
    print(f'{i:3d}: actual: {ytest[i]:d} predict: {0 if (prob[i] < 0.5) else 1:d} prob: {prob[i]:.7f}')
print(f'Score = {(1-score):.4f}')
