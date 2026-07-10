import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# ===========================
# Load Dataset
# ===========================

data = pd.read_csv("data/0_final_dataset.csv")

data = data.drop(
    columns=["Unnamed: 0", "Unnamed: 0.1"],
    errors="ignore"
)

TARGET = "final_result"

X = data.drop(
    columns=[
        TARGET,
        "id_student",
        "withdrew",
        "date_unregistration"
    ],
    errors="ignore"
)
y = data[TARGET]

# ===========================
# Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ===========================
# Load Trained Model
# ===========================

model = joblib.load("models/random_forest.pkl")

# ===========================
# Permutation Importance
# ===========================

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy",
    n_jobs=-1
)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("=" * 60)
print("Feature Importance")
print("=" * 60)

print(importance)

# ===========================
# Save Result
# ===========================

importance.to_csv(
    "models/random_forest_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully.")