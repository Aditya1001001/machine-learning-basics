import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#loading hte dataset
cancer = datasets.load_breast_cancer()

#data exploration
#print(cancer.feature_names)
#print(cancer.target_names)

X = cancer.data
Y = cancer.target

#splitting the dataset
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size= 0.15)

classes = ['malignant', 'benign']

#creating the svm classfier and training it
#default kernel fuction is rbf
#default soft margin is one
SVM_classifier = svm.SVC(kernel='linear', C=2)
SVM_classifier.fit(X_train, Y_train)

SVM_predictions = SVM_classifier.predict(X_test)
SVM_accuracy = metrics.accuracy_score(Y_test, SVM_predictions)


KNN_classifier = KNeighborsClassifier(n_neighbors= 13)
KNN_classifier.fit(X_train, Y_train)

KNN_predictions = KNN_classifier.predict(X_test)
KNN_accuracy = metrics.accuracy_score(Y_test, KNN_predictions)

print("SVM-  ",SVM_accuracy,"\nKNN-   ",KNN_accuracy)