# exercicio 1 & 2

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

breast_cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(breast_cancer['data'], breast_cancer['target'])
print(len(X_train),len(y_train))

# exercicio 3

from sklearn import tree

X, y = breast_cancer.data, breast_cancer.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
tree.plot_tree(clf)