import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = 11 * np.random.random((10, 1))
y = 1.0 * x + 3.0


model = LinearRegression()
model.fit(x, y)

x_pred = np.linspace(0, 11, 100)
y_pred = model.predict(x_pred[:, np.newaxis])

plt.figure(figsize=(3, 5))
ax = plt.axes()
ax.scatter(x, y)

ax.plot(x_pred, y_pred)
ax.set_xlabel('predictors')
ax.set_ylabel('criterion')
ax.axis('tight')
 
plt.show()