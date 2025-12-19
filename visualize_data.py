import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Create a clear Pandas DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("--- ORIGINAL DATA SAMPLE (First 5 Rows) ---")
print(df.head())
print("\n")

# 2. Introduce missing values (Same logic as before)
np.random.seed(42)
missing_percentage = 0.1
n_samples, n_features = X.shape
n_missing = int(n_samples * n_features * missing_percentage)

missing_indices = (
    np.random.randint(0, n_samples, n_missing),
    np.random.randint(0, n_features, n_missing)
)

X_missing = X.copy()
X_missing[missing_indices] = np.nan

# Create DataFrame with missing values
df_missing = pd.DataFrame(X_missing, columns=feature_names)
df_missing['target'] = y

print("--- DATA WITH MISSING VALUES (First 5 Rows) ---")
print("Note: NaN represents missing data")
print(df_missing.head())
print("\n")

# Show statistics of missing values
print("--- MISSING VALUES COUNT PER FEATURE ---")
print(df_missing.isnull().sum())
