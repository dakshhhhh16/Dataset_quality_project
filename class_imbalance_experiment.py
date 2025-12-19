import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

# ---------------------------------------------------
# STEP 1: Load dataset
# ---------------------------------------------------

data = load_breast_cancer()
X = data.data      # features
y = data.target    # labels (0 or 1)

# ---------------------------------------------------
# STEP 2: Define imbalance levels
# ---------------------------------------------------

# Fraction of minority class we will KEEP
# 1.0 = fully balanced
# 0.5 = remove 50% of minority class
# 0.2 = remove 80% of minority class
imbalance_levels = [1.0, 0.7, 0.4, 0.2]

accuracies = []
recalls = []

# ---------------------------------------------------
# STEP 3: Loop through imbalance levels
# ---------------------------------------------------

for level in imbalance_levels:

    # Find indices of each class
    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    # Decide how many minority samples to keep
    n_keep = int(len(class_1_idx) * level)

    # Randomly select minority samples
    selected_class_1 = np.random.choice(
        class_1_idx, n_keep, replace=False
    )

    # Combine majority class with reduced minority class
    selected_indices = np.concatenate([class_0_idx, selected_class_1])

    # Create new imbalanced dataset
    X_imbalanced = X[selected_indices]
    y_imbalanced = y[selected_indices]

    # ---------------------------------------------------
    # STEP 4: Train-test split
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_imbalanced, y_imbalanced,
        test_size=0.2,
        random_state=42,
        stratify=y_imbalanced
    )

    # ---------------------------------------------------
    # STEP 5: Train model
    # ---------------------------------------------------

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # STEP 6: Predict & evaluate
    # ---------------------------------------------------

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    accuracies.append(accuracy)
    recalls.append(recall)

    print(
        f"Minority kept: {int(level*100)}% | "
        f"Accuracy: {accuracy:.3f} | "
        f"Recall: {recall:.3f}"
    )

# We take the dataset

# We remove cancer patients deliberately

# We train the same model

# We check:

  # How often the model is correct (accuracy)

  # How many cancer cases it catches (recall)

# As cancer cases decrease:
  # Accuracy may stay high

  # Recall usually collapses