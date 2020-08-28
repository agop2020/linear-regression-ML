import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())
predict = "G3" #label: what you're trying to get
X = np.array(data.drop([predict], 1)) #drops G3 from attributes
y = np.array(data[predict]) #Represents actual values of G3(what we want to predict)
#Training the model with half of x and y and testing it with the other half(this code just splits the arrays and assigns to variables)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
best = 0
'''
for _ in range(30):
    
    #Training the model & printing the accuracy of the model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    #Saves best model based on 30(for loop range, can be any larger number) accuracy scores.
    if(acc>best):
        best = acc
    #Saves the model in a pickle file and loads model into "linear" variable
        with open("studentmodel.pickle", 'wb') as f:
            pickle.dump(linear, f)
'''
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#Printing coefficients for each attribute + the intercept
print("co", linear.coef_)
print("intercept", linear.intercept_)

#Array of arrays, predicting labels based on test data that we did not train our model on
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x]) #3 parameters are predicted values based on linear model, attributes, and actual G3 values

#Plotting data
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()
