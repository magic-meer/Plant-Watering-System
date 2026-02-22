"""XGBoost model training script for Plant Watering System."""

import os
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Handle Timestamp column
if "Timestamp" in X_train.columns:
    for df in [X_train, X_test]:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Year"] = df["Timestamp"].dt.year
        df["Month"] = df["Timestamp"].dt.month
        df["Day"] = df["Timestamp"].dt.day
        df["Hour"] = df["Timestamp"].dt.hour
        df.drop("Timestamp", axis=1, inplace=True)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Train XGBoost Model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Compute Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save Metrics
metrics_text = f"""
Plant-Watering System - XGBoost Model Metrics

Overall Accuracy: {accuracy:.4f}

Classification Report:
- Class 0: Healthy plant
- Class 1: Needs Water
- Class 2: Overwatered

{report}

Confusion Matrix:
Rows represent actual labels, columns represent predicted labels.
"""

with open("reports/metrics.txt", "w", encoding="utf-8") as f:
    f.write(metrics_text)

print("Metrics saved.")


# Save Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Actual vs Predicted Labels")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

# Save Feature Importance Plot
plt.figure(figsize=(8, 6))
xgb.plot_importance(model, importance_type="weight", max_num_features=20)
plt.title("Feature Importance (Top 20)")
plt.xlabel("F Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

# Save Model
joblib.dump(model, "models/xgboost_v1.pkl")

print("Model, metrics, and plots saved successfully.")
