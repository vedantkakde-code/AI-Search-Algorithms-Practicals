# Assignment 6: Data Preprocessing Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ---------- Step 1-3: Load & Explore ----------
# Use a small synthetic dataset (mirrors Titanic-like structure)
data = {
    'Age':    [22, 38, np.nan, 35, np.nan, 54, 2, 27, 14, 4],
    'Fare':   [7.25, 71.83, 8.05, 53.10, 8.05, 51.86, 21.07, 11.13, 30.07, 16.70],
    'Sex':    ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'male'],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'C', 'S', 'S', 'C', 'C'],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

print("=== ORIGINAL DATA ===")
print(df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ---------- Step 4: Handle Missing Values (Median imputation) ----------
df['Age'] = df['Age'].fillna(df['Age'].median())
print("\nAfter median imputation — missing values:\n", df.isnull().sum())

# ---------- Step 5: Encode Categorical Variables ----------
# Label Encoding for Sex
le = LabelEncoder()
df['Sex_LabelEncoded'] = le.fit_transform(df['Sex'])

# One-Hot Encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

print("\nAfter encoding:\n", df.head())

# ---------- Step 6: Normalize / Scale ----------
# Min-Max Scaling
minmax = MinMaxScaler()
df['Fare_MinMax'] = minmax.fit_transform(df[['Fare']])

# Standardization
std = StandardScaler()
df['Age_Standardized'] = std.fit_transform(df[['Age']])

print("\nAfter scaling:\n", df[['Fare', 'Fare_MinMax', 'Age', 'Age_Standardized']].head())

# Drop original 'Sex' now that it's encoded
df = df.drop(columns=['Sex'])
print("\nFinal preprocessed dataset:\n", df.head())
