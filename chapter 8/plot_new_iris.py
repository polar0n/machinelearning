import numpy as np
import matplotlib.pyplot as plt


x = np.load('data/iris/iris2_train.npy')
y = np.load('data/iris/iris2_train_labels.npy')

plt.scatter(
    x[np.where(y == 0)][:,0],
    x[np.where(y == 0)][:,1],
    c='r',
    marker='x'
)
plt.scatter(
    x[np.where(y == 1)][:,0],
    x[np.where(y == 1)][:,1],
    c='b',
    marker='+'
)
plt.show()
