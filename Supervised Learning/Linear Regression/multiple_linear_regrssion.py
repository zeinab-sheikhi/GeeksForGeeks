# It is important to mention that Linear Regression model should be used under certain assumption, 
# such as Linearity, Homoscedasticity, Independence of errors, Normality of errors, and no multicollinearity, if these assumptions are not met,
# you should consider using other techniques or transforming your data.

# As we know in the Multiple Regression Model we use a lot of categorical data. 
# The Dummy Variable Trap is a condition in which two or more are Highly Correlated. 
# In the simple term, we can say that one variable can be predicted from the prediction of the other. 
# The solution of the Dummy Variable Trap is to drop one of the categorical variables. 
# So if there are m Dummy variables then m-1 variables are used in the model. 

# Multiple linear Regression is a standard statistical method used to assess the relationships between a dependent variable and a set of independent variables. 
# In many cases, there are too many independent variables to include all of them in the regression model. 
# In these situations, modelers can use a backward elimination process to iteratively remove the least important variables until only the most important ones remain.

# Backward Elimination 
# Backward elimination is a simple and effective way to select a subset of variables for a linear regression model. It is easy to implement and can be automated. 
# The backward elimination process begins by fitting a multiple linear regression model with all the independent variables. 
# The variable with the highest p-value is removed from the model, and a new model fits. This process is repeated until all variables in the model have a p-value below some threshold, typically 0.05.

# Forward Selection
# It is a greedy algorithm that starts with an empty set of features and adds features one by one until the model performance reaches a peak.

# Data Pre Processing Steps:

# Importing The Libraries.
# Importing the Data Set.
# Encoding the Categorical Data.
# Avoiding the Dummy Variable Trap.
# Splitting the Data set into Training Set and Test Set.


import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def generate_dataset(n):
    x = []
    y = []
    random_x1 = np.random.rand()
    random_x2 = np.random.rand()
    for i in range(n):
        x1 = i
        x2 = i / 2 + np.random.rand() * n
        x.append([1, x1, x2])
        y.append(random_x1 * x1 + random_x2 * x2 + 1)
    return np.array(x), np.array(y)

x, y = generate_dataset(200)

mpl.rcParams['legend.fontsize'] = 12
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(x[:, 1], x[:, 2], y, label = 'y', s = 5)
ax.legend()
ax.view_init(45, 0)

plt.show()

def mse(coef, x, y):
    return np.mean((np.dot(x, coef) - y) ** 2) / 2

def gradients(coef, x, y):
    return np.mean(x.transpose() * (np.dot(x, coef) - y), axis=1)

def multilinear_regression(coef, x, y, lr, b1=0.9, b2=0.999, epsilon=1e-8):
    prev_error = 0
    m_coef = np.zeros(coef.shape)
    v_coef = np.zeros(coef.shape)
    moment_m_coef = np.zeros(coef.shape)
    moment_v_coef = np.zeros(coef.shape)
    t = 0

    while True:
        error = mse(coef, x, y)
        if abs(error - prev_error) <= epsilon:
            break
        prev_error = error 
        grad = gradients(coef, x, y)
        t += 1
        m_coef = b1 * m_coef + (1 - b1) * grad
        v_coef = b2 * v_coef + (1 - b2) * grad ** 2
        moment_m_coef = m_coef / (1 - b1 ** t)
        moment_v_coef = v_coef / (1 - b2 ** t)

        delta = ((lr / moment_v_coef ** 0.5 + 1e-8) * (b1 * moment_m_coef + (1 - b1) * grad / (1 - b1 ** t)))
        coef = np.subtract(coef, delta)
        return coef

coef = np.array([0, 0, 0])
c = multilinear_regression(coef, x, y, 1e-1)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
 
ax.scatter(x[:, 1], x[:, 2], y, label='y',
           s=5, color="dodgerblue")
 
ax.scatter(x[:, 1], x[:, 2], c[0] + c[1]*x[:, 1] + c[2]*x[:, 2],
           label='regression', s=5, color="orange")
 
ax.view_init(45, 0)
ax.legend()
plt.show()

# Scikit-learn multiple linear regression model

from sklearn.linear_model import LinearRegression

X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([1, 2, 3, 4])

reg = LinearRegression()
reg.fit(X, y)
print(reg.coef_)
