import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('employee_data.csv')

# Separate the target variable (years_experience) and features
X = df[['age', 'education_level', 'salary']]
y = df['years_experience']

# Create a simple imputer for the features
feature_imputer = SimpleImputer(strategy='mean')
X_imputed = feature_imputer.fit_transform(X)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_imputed[~np.isnan(y)], y[~np.isnan(y)])

# Predict missing values
missing_indices = np.where(np.isnan(df['years_experience']))
df.loc[missing_indices[0], 'years_experience'] = model.predict(X_imputed[missing_indices])

print(df.head())
