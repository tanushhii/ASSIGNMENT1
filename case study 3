import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Create a KNN imputer
imputer = KNNImputer(n_neighbors=5)

# Fit the imputer and transform the data
imputed_data = imputer.fit_transform(df)

# Create a new DataFrame with the imputed data
imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

print(imputed_df.head())
