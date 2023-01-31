# exercicio 1 & 2

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn import svm

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

breast_cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(breast_cancer['data'], breast_cancer['target'])
print(len(X_train),len(y_train))

# exercicio 3
clf=svm.SVC(kernel='rbf',gamma='auto')
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

# exercicio 4 (gráfico)

plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")
plt.show();

# exercicio 5

sc=MinMaxScaler(feature_range=(0,1))   # default se nao tiver argumentos
sc.fit(X_train)
x_train_scaled=sc.transform(X_train)

# exercicio 6

x_test_scaled=sc.transform(X_test)

# exercicio 7

clf2=svm.SVC(kernel='rbf',gamma='auto')
clf2.fit(x_train_scaled,y_train)

print(clf2.score(x_test_scaled,y_test))

# exercicio 8.2

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
gsc = GridSearchCV(svc, parameters)
gsc.fit(x_train_scaled, y_train)

print(gsc.best_params_)