import numpy as np
import keras
from keras.datasets import cifar10

(xtrn, ytrn), (xtst, ytst) = cifar10.load_data()
idx = np.argsort(np.random.random(ytrn.shape[0]))
xtrn = xtrn[idx]
ytrn = ytrn[idx]
idx = np.argsort(np.random.random(ytst.shape[0]))
xtst = xtst[idx]
ytst = ytst[idx]

np.save('cifar10_train_images.npy', xtrn)
np.save('cifar10_train_labels.npy', ytrn)
np.save('cifar10_test_images.npy', xtst)
np.save('cifar10_test_labels.npy', ytst)

xtrnv = xtrn.reshape((50000, 32*32*3))
xtstv = xtst.reshape((10000, 32*32*3))
np.save('cifar10_train_vectors.npy', xtrn)
np.save('cifar10_test_vectors.npy', xtst)
