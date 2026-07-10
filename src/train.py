import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model_pipeline import create_model


# ===========================
# Load Dataset
# ===========================

data = pd.read_csv("data/0_final_dataset.csv")

# Remove unwanted index columns
data = data.drop(
    columns=["Unnamed: 0", "Unnamed: 0.1"],
    errors="ignore"
)

# ===========================
# Features & Target
# ===========================

TARGET = "final_result"

X = data.drop(columns=[TARGET, "id_student"])
y = data[TARGET]

# ===========================
# Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===========================
# Create Model Pipeline
# ===========================

model = create_model(
    model_name="gradient_boosting",
    numeric_strategy="median",
    categorical_strategy="most_frequent"
)

# ===========================
# Train
# ===========================

print("Training model...\n")

model.fit(X_train, y_train)

print("Training completed.\n")

# ===========================
# Prediction
# ===========================

y_pred = model.predict(X_test)

# ===========================
# Evaluation
# ===========================

print("=" * 50)
print("Accuracy")
print("=" * 50)

print(
    accuracy_score(y_test, y_pred)
)

print("\n")

print("=" * 50)
print("Classification Report")
print("=" * 50)

print(
    classification_report(y_test, y_pred)
)

print("\n")

print("=" * 50)
print("Confusion Matrix")
print("=" * 50)

print(
    confusion_matrix(y_test, y_pred)
)

# ===========================
# Save Model
# ===========================

joblib.dump(
    model,
    "models/gradient_boosting_model.pkl"
)

print("\nModel Saved Successfully.")