#“Can we correct the model’s behavior without changing the model?”

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

# ---------------------------------------------------
# STEP 1: Load datasett
# ---------------------------------------------------

data = load_breast_cancer()
X = data.data
y = data.target

# ---------------------------------------------------
# STEP 2: Create an imbalanced dataset
# (we intentionally remove cancer cases)
# ---------------------------------------------------

# Find indices for each class
class_0_idx = np.where(y == 0)[0]  # non-cancer
class_1_idx = np.where(y == 1)[0]  # cancer

# Keep only 40% of cancer cases
n_keep = int(len(class_1_idx) * 0.4)

selected_class_1 = np.random.choice(
    class_1_idx, n_keep, replace=False
)

selected_indices = np.concatenate([class_0_idx, selected_class_1])

X_imbalanced = X[selected_indices]
y_imbalanced = y[selected_indices]

# ---------------------------------------------------
# STEP 3: Train-test split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_imbalanced,
    y_imbalanced,
    test_size=0.2,
    random_state=42,
    stratify=y_imbalanced
)

# ---------------------------------------------------
# STEP 4: Train model WITHOUT fixing imbalance
# ---------------------------------------------------

model_plain = LogisticRegression(max_iter=5000)
model_plain.fit(X_train, y_train)

y_pred_plain = model_plain.predict(X_test)

acc_plain = accuracy_score(y_test, y_pred_plain)
rec_plain = recall_score(y_test, y_pred_plain)

# ---------------------------------------------------
# STEP 5: Train model WITH class weighting
# ---------------------------------------------------

model_weighted = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"  # ⭐ THIS IS THE FIX
)

model_weighted.fit(X_train, y_train)

y_pred_weighted = model_weighted.predict(X_test)

acc_weighted = accuracy_score(y_test, y_pred_weighted)
rec_weighted = recall_score(y_test, y_pred_weighted)

# ---------------------------------------------------
# STEP 6: Compare results
# ---------------------------------------------------

print("WITHOUT class weights:")
print("Accuracy:", acc_plain)
print("Recall:", rec_plain)

print("\nWITH class weights:")
print("Accuracy:", acc_weighted)
print("Recall:", rec_weighted)
