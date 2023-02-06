import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

file_path = '/Users/macbookpro/Python/GeeksForGeeks/Machine Learning/Supervised Learning/Linear Regression/bostonhousing.csv'
df = pd.read_csv(file_path)

X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

print("xtrain shape : ", xtrain.shape)
print("xtest shape  : ", xtest.shape)
print("ytrain shape : ", ytrain.shape)
print("ytest shape  : ", ytest.shape)

regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
y_pred = regressor.predict(xtest)

plt.scatter(ytest, y_pred, color='b')
plt.xlabel("Price: in $1000's")
plt.ylabel("Predicted value")
plt.title("True value vs predicted value : Linear Regression")
plt.show()

mse = mean_squared_error(ytest, y_pred)
mae = mean_absolute_error(ytest, y_pred)
print("Mean Square Error : ", mse)
print("Mean Absolute Error : ", mae)