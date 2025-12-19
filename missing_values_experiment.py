#“What happens if real-world data is incomplete?”

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


# STEP 1: Load the dataset
  # Load a built-in medical dataset
data = load_breast_cancer()

  # X contains all the input features (numbers the model sees)
X = data.data

  # y contains the labels (what we want to predict: 0 or 1)
y = data.target

# STEP 2: Introduce missing values artificially

  # Fix random seed so results are reproducible 
np.random.seed(42) 

  # Decide how much data we want to remove
missing_percentage = 0.1  # 10% of values will be removed

  # Get dataset shape (rows, columns)
n_samples, n_features = X.shape

  # Total number of values to remove
n_missing = int(n_samples * n_features * missing_percentage)

  # Randomly choose positions (row index, column index)
missing_indices = (
    np.random.randint(0, n_samples, n_missing),
    np.random.randint(0, n_features, n_missing)
)

  # Copy original data so we don't destroy it
X_missing = X.copy()

  # Replace chosen positions with NaN (missing)
X_missing[missing_indices] = np.nan

# STEP 3: Handle missing values (very important)

  # Create an imputer that replaces missing values with column mean
imputer = SimpleImputer(strategy="mean")

  # Apply the imputer to the dataset
X_imputed = imputer.fit_transform(X_missing)

# STEP 4: Split data into training and testing

  # 80% data for learning, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# STEP 5: Train the machine learning model

  # Create the model
model = LogisticRegression(max_iter=5000)

  # Teach the model using training data
model.fit(X_train, y_train)

# STEP 6: Make predictions

  # Ask the model to predict labels for unseen data
y_pred = model.predict(X_test)

# STEP 7: Evaluate performance

  # Compare predictions with true answers
accuracy = accuracy_score(y_test, y_pred)

  # Print final result
print("Accuracy with 10% missing values:", accuracy)
