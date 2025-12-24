import numpy as np
import matplotlib.pyplot as plt

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
# STEP 2: Define imbalance levels
# ---------------------------------------------------

# Percentage of minority class we keep
imbalance_levels = [1.0, 0.7, 0.4, 0.2]

accuracies = []
recalls = []

# ---------------------------------------------------
# STEP 3: Run experiment for each imbalance level
# ---------------------------------------------------

for level in imbalance_levels:

    # Get indices for each class
    class_0_idx = np.where(y == 0)[0]  # majority class
    class_1_idx = np.where(y == 1)[0]  # minority class

    # Decide how many minority samples to keep
    n_keep = int(len(class_1_idx) * level)

    # Randomly choose minority samples
    selected_class_1 = np.random.choice(
        class_1_idx, n_keep, replace=False  
    )

    # Combine majority and reduced minority
    selected_indices = np.concatenate([class_0_idx, selected_class_1])

    # Create new dataset
    X_new = X[selected_indices]
    y_new = y[selected_indices]

    # ---------------------------------------------------
    # Train-test split
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y_new,
        test_size=0.2,
        random_state=42,
        stratify=y_new
    )

    # ---------------------------------------------------
    # Train model
    # ---------------------------------------------------

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Predict and evaluate
    # ---------------------------------------------------

    y_pred = model.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))

# ---------------------------------------------------
# STEP 4: Plot results
# ---------------------------------------------------

# Convert levels to percentages for plotting
x_values = [int(level * 100) for level in imbalance_levels]

plt.figure()

plt.plot(x_values, accuracies, label="Accuracy")
plt.plot(x_values, recalls, label="Recall")

plt.xlabel("Minority Class Kept (%)")
plt.ylabel("Score")
plt.title("Effect of Class Imbalance on Model Performance")
plt.legend()

plt.show()
