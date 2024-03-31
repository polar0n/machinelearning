import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle


clf = pickle.load(open('data/iris/best_mlp.sav', 'rb'))

x_test = np.load('data/iris/iris2_test.npy')
y_test = np.load('data/iris/iris2_test_labels.npy')

prob = clf.predict_proba(x_test)
score = clf.score(x_test, y_test)

print()
print('Test results:')
print()
for i in range(len(y_test)):
    p = 0 if prob[i, 1] < 0.5 else 1
    print(f'{i:03d}: {y_test[i]:d} - {p:d}, {prob[i, 1]:.7f}')
print(f'    Overall score: {score:.7f}')
print()
