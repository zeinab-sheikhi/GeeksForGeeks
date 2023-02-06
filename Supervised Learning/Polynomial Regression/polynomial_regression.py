import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data_path = '/Users/macbookpro/Python/GeeksForGeeks/Machine Learning/Supervised Learning/Polynomial Regression/temp.csv'
df = pd.read_csv(data_path)

X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

linear_model = LinearRegression()
linear_model.fit(X, y)

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
  
poly.fit(X_poly, y)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

plt.scatter(X, y, color='g')
plt.plot(X, linear_model.predict(X), color='r')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

plt.scatter(X, y, color='g')
plt.plot(X, poly_model.predict(X_poly), color='b')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

new_point = 110.0
new_point_array = np.array([[new_point]])
linear_pred = linear_model.predict(new_point_array)
poly_pred = poly_model.predict(poly.fit_transform(new_point_array))
