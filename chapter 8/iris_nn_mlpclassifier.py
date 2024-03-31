import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier


x_train = np.load('data/iris/iris2_train.npy')
y_train = np.load('data/iris/iris2_train_labels.npy')
x_test = np.load('data/iris/iris2_test.npy')
y_test = np.load('data/iris/iris2_test_labels.npy')

clf = MLPClassifier(
    hidden_layer_sizes=(3, 2),
    activation='relu',
    solver='lbfgs',
    tol=1e-9,
    max_iter=5000
)
clf.fit(x_train, y_train)
prob = clf.predict_proba(x_test)
score = clf.score(x_test, y_test)

w12 = clf.coefs_[0]
w23 = clf.coefs_[1]
w34 = clf.coefs_[2]

b1 = clf.intercepts_[0]
b2 = clf.intercepts_[1]
b3 = clf.intercepts_[2]
weights = [w12, b1, w23, b2, w34, b3]
pickle.dump(weights, open('data/iris/iris2_weights.pkl', 'wb'))

print()
print('Test results:')
print()
for i in range(len(y_test)):
    p = 0 if prob[i, 1] < 0.5 else 1
    print(f'{i:03d}: {y_test[i]:d} - {p:d}, {prob[i, 1]:.7f}')
print(f'    Overall score: {score:.7f}')
print()