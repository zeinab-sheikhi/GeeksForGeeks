# Data Cleaning steps:
# 1. Removal of unwanted such as redundant or irrelevant observations
# 2. Fixing structural errors including typos in the name of the features, the same attribute with a different name
# mislabeled classes, or inconsistent capitalization
# 3. Managing unwanted outliers 
# 4. Handling missing data by Dropping observations with missing values or Imputing mising values from past observations

import pandas as pd

df = pd.read_csv("/Users/macbookpro/Python/GeeksForGeeks/Data Processing/diabetes.csv")
has_null_values = df.isnull().values.any()

# Drop rows with missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Remove unnecessay columns
df = df.drop(columns=['Outcome'])

# Normalize numerical columns
df['BMI'] = (df['BMI'] - df['BMI'].mean()) / df['BMI'].std()

# Encode categorical columns
# df['columnname'] = pd.get_dummies(df['columnname'])
print(df)