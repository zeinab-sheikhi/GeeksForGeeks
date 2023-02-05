import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_predict = gnb.predict(X_test)
print("Accuracy of Gaussian Naive Bayes: ", accuracy_score(y_test, gnb_predict))

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_predict = dtc.predict(X_test)
print("Accuracy of Decision Tree Classifier: ", accuracy_score(y_test, dtc_predict))

svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
svm_predict = svm_clf.predict(X_test)
print("Accuracy of Support Vector Machine: ",
      accuracy_score(y_test, svm_predict))