# One hot encoding is a technique used to represent categorical variables as numerical values in a machine learning model.
# It can lead to increased dimensionality, sparsity and overfitting. 
# In this technique, the categorical parameters will prepare separate columns for each categorical value.

import numpy as np
import pandas as pd

df = pd.read_csv("/Users/macbookpro/Python/GeeksForGeeks/Data Processing/iris.csv")
print(df.head())
print(df['Species'].unique())
print(df['Species'].value_counts())

# One-Hot encoding the categorical parameters using get_dummies() 
one_hot_encoded_data = pd.get_dummies(df, columns=['Species'])
print(one_hot_encoded_data)

# One Hot Encoding using Sci-kit learn Library
# Before implementing this algorithm. Make sure the categorical values
# must be label encoded as one hot encoding takes only numerical categorical values. 
from sklearn.preprocessing import OneHotEncoder
df['Species'] = df['Species'].astype('category')
df['new_species'] = df['Species'].cat.codes

enc = OneHotEncoder()
enc_data = pd.DataFrame(enc.fit_transform(df[['new_species']]).toarray())
New_df = df.join(enc_data)
 
print(New_df)