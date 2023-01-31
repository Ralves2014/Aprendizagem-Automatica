from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

breast_cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(breast_cancer['data'], breast_cancer['target'])
print(len(X_train),len(y_train))

clf= RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)