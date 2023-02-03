import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

mu = 0.5
sigma = 0.1
np.random.seed(0)

X = np.random.normal(mu, sigma, (395, 1))
Y = np.random.normal(mu * 2, sigma * 3, (395, 1))
plt.scatter(X, Y, color='g')
# plt.show()

# We will generate a dataset with 4 columns. Each column in the dataset represents a feature.
# The 5th column of the dataset is the output label. It varies between 0-3. 
point1 = abs(np.random.normal(1, 12, 100))
point2 = abs(np.random.normal(2, 8, 100))
point3 = abs(np.random.normal(3, 2, 100))
point4 = abs(np.random.normal(10, 15, 100))

x = np.c_[point1, point2, point3, point4]
y = [int(np.random.randint(0, 4)) for i in range(100)]
data = pd.DataFrame()

data['col1'] = point1
data['col2'] = point2
data['col3'] = point3
data['col4'] = point4

plt.subplot(2, 2, 1)
plt.title('Col1')
plt.scatter(y, point1, color='r', label='col1')

plt.subplot(2, 2, 2)
plt.title('Col2')
plt.scatter(y, point2, color='g', label='col2')

plt.subplot(2, 2, 3)
plt.title('Col3')
plt.scatter(y, point3, color='b', label='col3')

plt.subplot(2, 2, 4)
plt.title('Col4')
plt.scatter(y, point4, color='y', label='col4')

plt.show()
