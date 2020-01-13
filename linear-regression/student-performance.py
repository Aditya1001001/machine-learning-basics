import pandas as pd 
import numpy as np  
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot 
from matplotlib import style

# we have to use the 'sep' argument because data columns are seprated by semi-colons instead of the deafault commas
data = pd.read_csv("student-mat.csv", sep=";")
#print(data.head())

# trimming the dataset
selected_attributes = ["G1", "G2", "G3", "studytime", "failures", "absences"]
data = data[selected_attributes]
#print(data.head())
#print(data.size)

predict = "G3"

#setting style for plots
style.use("ggplot")

#some visualisation
for x in selected_attributes:
    if x != "G3":

        pyplot.scatter(data[x], data[predict])
        pyplot.xlabel(x)
        pyplot.ylabel("Final Grade")
        pyplot.show()

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

#splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

best = 0 
for i in range(30):

    #this selection is done randomly so that's why we get different models for different iterations
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    model = linear_model.LinearRegression()

    #training the model and evaluating the efficiency
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test,Y_test)
    print(accuracy)

    #saving the model
    if accuracy > best:
        best = accuracy
        with open("linear_model.pickle","wb") as f:
            pickle.dump(model, f)

#loading the model
with open("linear_model.pickle","rb") as f:
    model = pickle.load(f) 

#printing the model 'parameters'
print("Coeff: ", model.coef_)
print("Intercept: ", model.intercept_)

#using the model to make predictions for given atrribut values
predictions = model.predict(X_test) 
for i in range(len(predictions)):
    print(predictions[i], X_test[i], Y_test[i], sep="   " )