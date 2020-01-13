import sklearn 
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

#reading data
data = pd.read_csv("car.data")
#print(data.head())

#data preprocessing
#converting nonminal and ordinal data to numeric data
label_encoder = preprocessing.LabelEncoder()
buying = label_encoder.fit_transform(list(data["buying"]))
maint = label_encoder.fit_transform(list(data["maint"]))
persons = label_encoder.fit_transform(list(data["persons"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
classes = label_encoder.fit_transform(list(data["class"]))
door = label_encoder.fit_transform(list(data["door"]))
#print(safety)

predict = "class"

#putting it all back together
X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(classes)

#splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

#creating a classifier with k=5
model = KNeighborsClassifier(n_neighbors=5)

#fitting and testing the model
model.fit(X_train, Y_train)
accuracy = model.score(X_test,Y_test)
print(accuracy)

#label names
names = ["unacc", "acc", "good", "vgood"]

#making predictions
predicted = model.predict(X_test)
for i in range(len(X_test)):
    print("Predicted:  ", names[predicted[i]], "  Data:  ",X_test[i], "  Actual:  ", names[Y_test[i]])
    #getting the indices and distance of 9 nearest neighrbours
    neighbors = model.kneighbors([X_test[i]], 9, True)
    print(neighbors)