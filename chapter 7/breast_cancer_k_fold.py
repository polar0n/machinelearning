import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys



def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)


def split(x, y, k, m):
    ns = int(y.shape[0]/m)
    s = []
    for i in range(m):
        s.append([x[(ns*i):(ns*i+ns)],
                 y[(ns*i):(ns*i+ns)]])
    x_test, y_test = s[k]
    x_train = []
    y_train = []

    for i in range(m):
        if i == k:
            continue
        else:
            a, b = s[i]
            x_train.append(a)
            y_train.append(b)

    x_train = np.array(x_train).reshape(((m-1)*ns, 30))
    y_train = np.array(y_train).reshape((m-1)*ns)
    return [x_train, y_train, x_test, y_test]


def pp(z, k, s):
    m = z.shape[1]
    print(f'{s:21s}: {z[k].mean():.6f} ± {z[k].std()/np.sqrt(m):.4f}|', end='')
    for i in range(m):
        print(f' {z[k, i]:.4f}', end='')
    print()


def main():
    x = np.load('data/breast_cancer/bc_features_standard.npy')
    y = np.load('data/breast_cancer/bc_labels.npy')
    idx = np.argsort(np.random.random(y.shape[0]))
    x = x[idx]
    y = y[idx]
    args = sys.argv
    if len(args) == 1:
        m = 5
    else:
        m = int(args[1])
    z = np.zeros((8, m))

    # Fine tuning k-NN classifier
    # for k in range(m):
    #     args = split(x, y, k, m)
    #     z[0,k] = run(*args, KNeighborsClassifier(n_neighbors=k*2+1))
    #     z[1,k] = run(*args, KNeighborsClassifier(n_neighbors=k*2+1))
    #     z[2,k] = run(*args, KNeighborsClassifier(n_neighbors=k*2+1))
    #     z[3,k] = run(*args, KNeighborsClassifier(n_neighbors=k*2+1))
    #     z[4,k] = run(*args, KNeighborsClassifier(n_neighbors=k*2+1))
    #     z[5,k] = run(*args, KNeighborsClassifier(n_neighbors=k*2+1))
    #     z[6,k] = run(*args, KNeighborsClassifier(n_neighbors=k*2+1))
    #     z[7,k] = run(*args, KNeighborsClassifier(n_neighbors=k*2+1))

    # pp(z, 0, '1-NN')
    # pp(z, 1, '3-NN')
    # pp(z, 2, '5-NN')
    # pp(z, 3, '7-NN')
    # pp(z, 4, '9-NN')
    # pp(z, 5, '11-NN')
    # pp(z, 6, '13-NN')
    # pp(z, 7, '15-NN')
    
    # Fine tuning Random Forest Classifier
    # z = np.zeros((8, m))
    # nt = [5, 20, 50, 100, 200, 500, 1000, 5000]
    # for k in range(m):
    #     args = split(x, y, k, m)
    #     z[0,k] = run(*args, RandomForestClassifier(n_estimators=nt[0]))
    #     z[1,k] = run(*args, RandomForestClassifier(n_estimators=nt[1]))
    #     z[2,k] = run(*args, RandomForestClassifier(n_estimators=nt[2]))
    #     z[3,k] = run(*args, RandomForestClassifier(n_estimators=nt[3]))
    #     z[4,k] = run(*args, RandomForestClassifier(n_estimators=nt[4]))
    #     z[5,k] = run(*args, RandomForestClassifier(n_estimators=nt[5]))
    #     z[6,k] = run(*args, RandomForestClassifier(n_estimators=nt[6]))
    #     z[7,k] = run(*args, RandomForestClassifier(n_estimators=nt[7]))

    # pp(z, 0, 'RFTC (5)')
    # pp(z, 1, 'RFTC (20)')
    # pp(z, 2, 'RFTC (50)')
    # pp(z, 3, 'RFTC (100)')
    # pp(z, 4, 'RFTC (200)')
    # pp(z, 5, 'RFTC (500)')
    # pp(z, 6, 'RFTC (1000)')
    # pp(z, 7, 'RFTC (5000)')

    # Fine tuning linear SVM
    # z = np.zeros((8, m))
    # cs = [0.001, 0.01 ,0.1, 1.0, 2.0, 10.0, 50.0, 100.0]
    # for k in range(m):
    #     args = split(x, y, k, m)
    #     z[0,k] = run(*args, SVC(kernel='linear', C=cs[0]))
    #     z[1,k] = run(*args, SVC(kernel='linear', C=cs[1]))
    #     z[2,k] = run(*args, SVC(kernel='linear', C=cs[2]))
    #     z[3,k] = run(*args, SVC(kernel='linear', C=cs[3]))
    #     z[4,k] = run(*args, SVC(kernel='linear', C=cs[4]))
    #     z[5,k] = run(*args, SVC(kernel='linear', C=cs[5]))
    #     z[6,k] = run(*args, SVC(kernel='linear', C=cs[6]))
    #     z[7,k] = run(*args, SVC(kernel='linear', C=cs[7]))

    # pp(z, 0, 'SVM (linear, C=0.001)')
    # pp(z, 1, 'SVM (linear, C=0.01)')
    # pp(z, 2, 'SVM (linear, C=0.1)')
    # pp(z, 3, 'SVM (linear, C=1.0)')
    # pp(z, 4, 'SVM (linear, C=2.0)')
    # pp(z, 5, 'SVM (linear, C=10.0)')
    # pp(z, 6, 'SVM (linear, C=50.0)')
    # pp(z, 7, 'SVM (linear, C=100.0)')

    # for k in range(m):
    #     args = split(x, y, k, m)
    #     z[0,k] = run(*args, NearestCentroid())
    #     z[1,k] = run(*args, KNeighborsClassifier(n_neighbors=3))
    #     z[2,k] = run(*args, KNeighborsClassifier(n_neighbors=7))
    #     z[3,k] = run(*args, GaussianNB())
    #     z[4,k] = run(*args, DecisionTreeClassifier())
    #     z[5,k] = run(*args, RandomForestClassifier(n_estimators=5))
    #     z[6,k] = run(*args, RandomForestClassifier(n_estimators=50))
    #     z[7,k] = run(*args, SVC(kernel='linear', C=1.0))
    
    # pp(z, 0, 'Nearest Centroid')
    # pp(z, 1, 'Naïve Bayes')
    # pp(z, 2, 'Decision Tree')
    # pp(z, 3, 'Random Forest (5)')
    # pp(z, 4, 'Random Forest (50)')
    # pp(z, 5, 'SVM (linear)')


main()
