import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

file_path = '/Users/macbookpro/Python/GeeksForGeeks/Machine Learning/Supervised Learning/Linear Regression/housing.csv'
df = pd.read_csv(file_path)
new_column_name = '0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98, 24.00'
df.rename(columns={' 0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00': new_column_name}, inplace=True)
# print(df.columns)
df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PITRATIO', 'B', 'LSTAT', 'Price']] = df[new_column_name].str.split(pat=", ", expand=True)
# df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PITRATIO', 'B', 'LSTAT', 'Price']] = df[new_column_name].str.split(", ", expand=True)
# df[new_column_name] = df[new_column_name].str.split(", ", expand=True)
# print(df.columns)

# # print(df.columns.str.split(" ", expand=True))
# df.drop(df.index[0])
print(df.shape)

# print(df.dtypes)

# df.columns = [''] * len(df.columns)

# columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PITRATIO', 'B', 'LSTAT', 'Price']
# df.set_axis(columns, axis=1, inplace=False)
# print((df.head()))
# x = df.drop('24.00', axis=1)
# print(x)
# y = df.iloc[:]
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2)
