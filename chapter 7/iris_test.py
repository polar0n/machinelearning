import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    print(f'    predictions:   {clf.predict(x_test)}')
    print(f'    actual labels: {y_test}')
    print(f'    score = {clf.score(x_test, y_test):.4f}')


def main():
    x = np.load('data/iris/iris_features.npy')
    y = np.load('data/iris/iris_labels.npy')
    N = 120
    x_train = x[:N]
    x_test = x[N:]
    y_train = y[:N]
    y_test = y[N:]
    xa_train = np.load('data/iris/iris_train_features_augmented.npy')
    ya_train = np.load('data/iris/iris_train_labels_augmented.npy')
    xa_test = np.load('data/iris/iris_test_features_augmented.npy')
    ya_test = np.load('data/iris/iris_test_labels_augmented.npy')

    print('Nearest Centroid:')
    run(x_train, y_train, x_test, y_test, NearestCentroid())
    print('k-NN classifier (k=3):')
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
    print('Naïve Bayes classifier (Gaussian):')
    run(x_train, y_train, x_test, y_test, GaussianNB())
    print('Naïve Bayes (Multinomial):')
    run(x_train, y_train, x_test, y_test, MultinomialNB())
    print('Decision Tree Classifier:')
    run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())
    print('Random Forest Classifier:')
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=5))

    print('SVM (linear, C=1.0):')
    run(xa_train, ya_train, xa_test, ya_test, SVC(kernel='linear', C=1.0))
    print('SVM (RBF, C=1.0, γ=0.25):')
    run(xa_train, ya_train, xa_test, ya_test, SVC(kernel='rbf', C=1.0, gamma=0.25))
    print('SVM (RBF, C=1.0, γ=0.001 augmented):')
    run(xa_train, ya_train, xa_test, ya_test, SVC(kernel='rbf', C=1.0, gamma=0.001))
    print('SVM (RBF, C=1.0, γ=0.001, original):')
    run(x_train, y_train, x_test, y_test, SVC(kernel='rbf', C=1.0, gamma=0.001))


main()
