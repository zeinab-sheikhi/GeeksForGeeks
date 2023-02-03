# Label Encoding refers to converting the labels into a numeric form
# so as to convert them into the machine-readable form.
# Limitation of label Encoding 
# Label encoding converts the data in machine-readable form, but it assigns a unique number(starting from 0)
# to each class of data. This may lead to the generation of priority issues in the training of data sets. 
# A label with a high value may be considered to have high priority than a label having a lower value.

import numpy as np
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("/Users/macbookpro/Python/GeeksForGeeks/Data Processing/iris.csv")

print("Species column values before label encoding")
print(df['Species'].unique())

label_encoder = preprocessing.LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

print("Species column values after label encoding")
print(df['Species'].unique())