import numpy as np
import matplotlib.pyplot as plt

x = np.load('iris_features.npy')

plt.boxplot(x)
plt.show()
