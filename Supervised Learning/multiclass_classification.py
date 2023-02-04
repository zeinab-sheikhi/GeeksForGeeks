from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(max_depth=2)
decision_tree_model.fit(X_train, y_train)
decision_tree_predict = decision_tree_model.predict(X_test)
decision_tree_accuracy = confusion_matrix(y_test, decision_tree_predict)
print(decision_tree_accuracy)

# SVM 
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predict = svm_model.predict(X_test)
accuracy = svm_model.score(X_test, y_test)
svm_accuracy = confusion_matrix(y_test, svm_predict)
print(accuracy)
print(svm_accuracy)

# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
accuracy = knn_model.score(X_test, y_test)
knn_predict = knn_model.predict(X_test)
knn_accuarcy = confusion_matrix(y_test, knn_predict)
print(knn_accuarcy)
print(accuracy)


# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gaussian_nb_model = GaussianNB().fit(X_train, y_train)
gaussian_predict = gaussian_nb_model.predict(X_test)
accuracy = gaussian_nb_model.score(X_test, y_test)
gaussian_accuracy = confusion_matrix(y_test, gaussian_predict)
print(gaussian_accuracy)
print(accuracy)
