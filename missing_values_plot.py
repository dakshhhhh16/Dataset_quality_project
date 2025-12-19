import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# ---------------------------------------------------
# STEP 1: Load dataset
# ---------------------------------------------------

data = load_breast_cancer()
X = data.data      # input features
y = data.target    # labels (0 or 1)

# ---------------------------------------------------
# STEP 2: Define missing value levels to test
# ---------------------------------------------------

# Different percentages of missing data
missing_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

# To store accuracy results
accuracies = []

# Fix randomness so results are consistent
np.random.seed(42)

# ---------------------------------------------------
# STEP 3: Loop over each missing level
# ---------------------------------------------------

for missing_percentage in missing_levels:

    # Make a copy of original data
    X_corrupted = X.copy()

    # Only add missing values if percentage > 0
    if missing_percentage > 0:
        n_samples, n_features = X.shape
        n_missing = int(n_samples * n_features * missing_percentage)

        # Random positions to remove values
        missing_indices = (
            np.random.randint(0, n_samples, n_missing),
            np.random.randint(0, n_features, n_missing)
        )

        # Insert missing values
        X_corrupted[missing_indices] = np.nan

        # Replace missing values using column mean
        imputer = SimpleImputer(strategy="mean")
        X_corrupted = imputer.fit_transform(X_corrupted)

    # ---------------------------------------------------
    # STEP 4: Train-test split
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_corrupted, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------
    # STEP 5: Train model
    # ---------------------------------------------------

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # STEP 6: Predict and evaluate
    # ---------------------------------------------------

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store result
    accuracies.append(accuracy)

    print(f"Missing: {int(missing_percentage*100)}% | Accuracy: {accuracy:.3f}")

# ---------------------------------------------------
# STEP 7: Plot results
# ---------------------------------------------------

plt.figure()
plt.plot(
    [p * 100 for p in missing_levels],  # X-axis: missing %
    accuracies                           # Y-axis: accuracy
)

plt.xlabel("Percentage of Missing Values")
plt.ylabel("Model Accuracy")
plt.title("Impact of Missing Data on Model Accuracy")
plt.show()
