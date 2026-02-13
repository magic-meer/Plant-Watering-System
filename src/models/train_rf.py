import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =====================================
# Load Data
# =====================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]
y_test  = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]


# =====================================
# FIX: Remove non-numeric columns (datetime etc.)
# =====================================

X_train = X_train.select_dtypes(include=["number"])
X_test  = X_test.select_dtypes(include=["number"])


# =====================================
# Train Random Forest Model
# =====================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)


# =====================================
# Evaluate Model
# =====================================

y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# =====================================
# Save Model
# =====================================

joblib.dump(rf_model, "models/random_forest_model.pkl")

print("✅ Random Forest model trained and saved successfully!")
