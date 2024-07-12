import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the dataset
df = pd.read_csv('environmental_data.csv')

# Create a random forest imputer
rf_imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100),
                              random_state=0,
                              max_iter=10)

# Fit the imputer and transform the data
imputed_data = rf_imputer.fit_transform(df)

# Create a new DataFrame with the imputed data
imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

print(imputed_df.head())
