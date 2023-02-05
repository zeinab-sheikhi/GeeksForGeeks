import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('')
df_binary = df[['Salnty', 'T_degC']]
print(df_binary)
# df_binary_columns = ['Sal', 'Temp']
# df_binary.head()

# sns.lmplot(x = 'Sal', y = 'Temp', data = df_binary, order = 2, ci = None)
# df_binary.fillna(method = 'ffill', inplace = True)

# X = np.array(df_binary['Sal']).reshape(-1, 1)
# y = np.array(df_binary["Temp"]).reshape(-1, 1)

# df_binary.drop_na(replace = True)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# regr  = LinearRegression()

# regr.fit(X_train, y_train)
# score = regr.score(X_test, y_test)
# print(score)

# y_pred = regr.predict(X_test)
# plt.scatter(X_test, y_test, color = 'b')
# plt.plot(X_test, y_pred, color = 'k')
# plt.show()