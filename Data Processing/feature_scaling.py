# Feature scaling is a technique to standardize the independant features present in the data in a fix range
# The distance between the new data point and the centroid can be calculated with the following methods:
# Euclidean Distance: It is the square root of the sum of squares of differences between the coordinates
# Manhattan Distance: It is calculated as the sum of absolute differences between the coordinates (feature values) 
# of data point and centroid of each class. 

# If an algorithm is not using the feature scaling method then it can consider the value 3000 meters to be greater 
# than 5 km but that’s actually not true and in this case, the algorithm will give wrong predictions. 
# So, we use Feature Scaling to bring all values to the same magnitudes and thus, tackle this issue.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

data_set = pd.read_csv('')
data_set.head()

x = data_set.iloc[:, 1:3].values

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled_x = min_max_scaler.fit_transform(x)
print("\nAfter min max Scaling : \n", scaled_x)

standardisation = preprocessing.StandardScaler()
standard_x = standardisation.fit_transform(x)
print("\nAfter Standardisation : \n", standard_x)
