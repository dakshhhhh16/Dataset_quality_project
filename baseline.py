#“How well does the model work when data is clean and fair?”

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train) #Look at the training data and learn from it.

# 4. Make predictions
y_pred = model.predict(X_test) #Use what you learned to make guesses on new data.

# 5. Check accuracy
accuracy = accuracy_score(y_test, y_pred) #Out of 100 predictions, how many were correct?
print("Baseline Accuracy:", accuracy)

