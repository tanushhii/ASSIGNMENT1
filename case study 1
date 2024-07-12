import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('housing_data.csv')

# Create an imputer object
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the 'bedrooms' column and transform
df['bedrooms'] = imputer.fit_transform(df[['bedrooms']])

print(df['bedrooms'].head())
