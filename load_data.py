import numpy as np

with open('iris.data', 'r') as f:
    lines = [i[:-1] for i in f.readlines()]

n = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
x = [n.index(i.split(',')[-1]) for i in lines if i != '']
x = np.array(x, dtype='uint8')

y = [[float(j) for j in i.split(',')[:-1]] for i in lines if i != '']
y = np.array(y)

i = np.argsort(np.random.random(x.shape[0]))
x = x[i]
y = y[i]

np.save('iris_features.npy', y)
np.save('iris_labels.npy', x)
