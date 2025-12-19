import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

# ---------------------------------------------------
# STEP 1: Load dataset
# ---------------------------------------------------

data = load_breast_cancer()
X = data.data
y = data.target

# ---------------------------------------------------
# STEP 2: Create class imbalance (same as before)
# ---------------------------------------------------

class_0_idx = np.where(y == 0)[0]
class_1_idx = np.where(y == 1)[0]

# Keep only 40% of cancer cases
n_keep = int(len(class_1_idx) * 0.4)

selected_class_1 = np.random.choice(
    class_1_idx, n_keep, replace=False
)

selected_indices = np.concatenate([class_0_idx, selected_class_1])

X_new = X[selected_indices]
y_new = y[selected_indices]

# ---------------------------------------------------
# STEP 3: Train-test split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_new,
    y_new,
    test_size=0.2,
    random_state=42,
    stratify=y_new
)

# ---------------------------------------------------
# STEP 4: Train model
# ---------------------------------------------------

model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# STEP 5: Get prediction probabilities
# ---------------------------------------------------

# Predict probabilities instead of final labels
# [:, 1] means probability of class "1" (cancer)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------
# STEP 6: Try different thresholds
# ---------------------------------------------------

thresholds = [0.5, 0.4, 0.3, 0.2]

for threshold in thresholds:

    # Convert probabilities into predictions manually
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(
        f"Threshold: {threshold} | "
        f"Accuracy: {acc:.3f} | "
        f"Recall: {rec:.3f}"
    )
