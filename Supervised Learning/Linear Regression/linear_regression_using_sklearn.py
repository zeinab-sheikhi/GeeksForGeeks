import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def convert_dtype(x):
    if not x:
        return ''
    try:
        return str(x)   
    except:        
        return ''

csv_path = '/Users/macbookpro/Python/GeeksForGeeks/Machine Learning/Supervised Learning/Linear Regression/bottle.csv'
# Solve the following warning:
# DtypeWarning: Columns (47,73) have mixed types.
df = pd.read_csv(csv_path, converters={'IncTim': convert_dtype, 'DIC Quality Comment': convert_dtype})
df_binary = df[['Salnty', 'T_degC']]
df_binary.columns = ['Sal', 'Temp']

sns.lmplot(x='Sal', y='Temp', data=df_binary, order=2, ci=None)
# plt.show()

df_binary.fillna(method='ffill', inplace=True)

X = np.array(df_binary['Sal']).reshape(-1, 1)
y = np.array(df_binary["Temp"]).reshape(-1, 1)

df_binary.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
regr  = LinearRegression()

regr.fit(X_train, y_train)
score = regr.score(X_test, y_test)
# print(score)

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')
# plt.show()

df_binary500 = df_binary[:][:500]
sns.lmplot(x="Sal", y="Temp", data=df_binary500, order=2, ci=None)
# plt.show()

df_binary500.fillna(method='ffill', inplace=True)

X = np.array(df_binary500['Sal']).reshape(-1, 1)
y = np.array(df_binary500["Temp"]).reshape(-1, 1)

df_binary500.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
  
regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')
  
plt.show()

mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)