import numpy as np
import matplotlib.pyplot as plt
import array as arr

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


iris = load_iris()

print("iris.keys(): \n{}".format(iris.keys()))
print("Shape of iris data: {}".format(iris.data.shape))
print("Sample counts per class:\n{}".format(
{n: v for n, v in zip(iris.target_names, np.bincount(iris.target))}))
print("Feature names:\n{}".format(iris.feature_names))

X_train, X_test, y_train, y_test = train_test_split(
	iris['data'], iris['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

confusion = confusion_matrix(y_test, pred_knn)
print("Confusion matrix:\n{}".format(confusion))
print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))

ypoints = arr.array('d');       ##array de float

for case in [20,40,60,80,100]:
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_train, y_train, test_size=case, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train1, y_train1)
    pred_knn = knn.predict(X_test1)

    confusion = confusion_matrix(y_test1, pred_knn)
    print("Confusion matrix:\n{}".format(confusion))
    print("Test set accuracy: {:.2f}".format(knn.score(X_test1, y_test1)))
    ypoints.append(knn.score(X_test1, y_test1));

xpoints = arr.array('i',[20,40,60,80,100])      ## array de inteiros

plt.plot(xpoints, ypoints)
plt.show()
