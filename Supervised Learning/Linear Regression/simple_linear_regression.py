from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

np.random.seed(0)
x = np.random.rand(100,1)
y = 2 + 3 * np.random.rand(100, 1)

model = LinearRegression()
model.fit(x, y)

x_new = np.array([[0], [1]])
y_new = model.predict(x_new)

print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_)

y_pred = model.predict(x)
mse = mean_squared_error(y, y_pred)
print('Mean Squared Error:', mse)