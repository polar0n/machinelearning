import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    print(f'    score = {clf.score(x_test, y_test):.4f}')
    print()


def main():
    x = np.load('data/breast_cancer/bc_features_standard.npy')
    y = np.load('data/breast_cancer/bc_labels.npy')

    # ---- Random Permutation [Start] ----
    np.random.seed(12345)
    idx = np.argsort(np.random.random(y.shape[0]))
    x = x[idx]
    y = y[idx]
    # sklearn tree classifiers also use np.random thus we need to set a new random value
    np.random.seed()
    # ---- Random Permutation [End] ----

    N = 455
    x_train = x[:N]
    x_test = x[N:]
    y_train = y[:N]
    y_test = y[N:]

    print('Nearest Centroid:')
    run(x_train, y_train, x_test, y_test, NearestCentroid())
    print('k-NN classifier (k=3):')
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
    print('k-NN Classifier (k=7):')
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))
    print('Naïve Bayes classifier (Gaussian):')
    run(x_train, y_train, x_test, y_test, GaussianNB())
    print('Decision Tree classifier:')
    run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())
    print('Random Forest classifier (estimators=5):')
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=5))
    print('Random Forest classifier (estimators=50):')
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=50))
    print('SVM (linear, C=1.0)')
    run(x_train, y_train, x_test, y_test, SVC(kernel='linear', C=1.0))
    print('SVM (RBF, C=1.0, γ=0.03333):')
    run(x_train, y_train, x_test, y_test, SVC(kernel='rbf', C=1.0, gamma=0.03333))


main()
