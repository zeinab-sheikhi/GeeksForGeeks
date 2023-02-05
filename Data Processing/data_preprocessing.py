import numpy 
import pandas
import scipy

data_path = "/Users/macbookpro/Python/GeeksForGeeks/Data Processing/diabetes.csv"
# List of column names to use
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pandas.read_csv(data_path, on_bad_lines='skip')

# print(df.describe())
# print(df.head(10))

array = df.values
X = array[:, 0:8]
Y = array[:, 8]
numpy.set_printoptions(precision=3)

# Rescale data (between 0 and 1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
print(rescaledX[0:5, :])

# Binarize Data (Make Binary)
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
print(binaryX[0:5, :])

# Standardize Data
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler().fit(X)
standardX = standard_scaler.transform(X)
print(standardX[0:5, :])
