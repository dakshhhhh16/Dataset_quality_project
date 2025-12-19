import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Introduce missing values
np.random.seed(42) 
missing_percentage = 0.1  # 10% missing values

n_samples, n_features = X.shape
n_missing = int(n_samples * n_features * missing_percentage)

missing_indices = (
    np.random.randint(0, n_samples, n_missing),
    np.random.randint(0, n_features, n_missing)
)

X_missing = X.copy()
X_missing[missing_indices] = np.nan

# 3. Handle missing values (simple strategy)
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_missing)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy with 10% missing values:", accuracy)
