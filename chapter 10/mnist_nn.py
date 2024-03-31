import numpy as np
from sklearn.neural_network import MLPClassifier


x_train = np.load('data/mnist/super_augmented/train_vectors.npy')
y_train = np.load('data/mnist/super_augmented/train_labels.npy')
x_test = np.load('data/mnist/super_augmented/test_vectors.npy')
y_test = np.load('data/mnist/super_augmented/test_labels.npy')

clf = MLPClassifier(
    hidden_layer_sizes=(500),
    activation='logistic',
    solver='sgd',
    tol=1e-6,
    max_iter=50,
    nesterovs_momentum=False,
    verbose=True
)

clf.fit(x_train, y_train)
prob = clf.predict_proba(x_test)
score = clf.score(x_test, y_test)

print()
print(f'Overall score: {score:.7f}')
print()
