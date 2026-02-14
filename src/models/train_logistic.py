import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 0️⃣ Load Data
# --------------------------------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()


# --------------------------------------------------
# Handle NaN values (fill with column mean for numeric columns only)
# --------------------------------------------------
numeric_cols = X_train.select_dtypes(include=["number"]).columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())


# --------------------------------------------------
# Handle Timestamp column (if exists and is object type)
# --------------------------------------------------
if "Timestamp" in X_train.columns:
    # Check if Timestamp is object type
    if X_train["Timestamp"].dtype == "object":
        for df in [X_train, X_test]:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["Year"] = df["Timestamp"].dt.year
            df["Month"] = df["Timestamp"].dt.month
            df["Day"] = df["Timestamp"].dt.day
            df["Hour"] = df["Timestamp"].dt.hour
            df.drop("Timestamp", axis=1, inplace=True)


# --------------------------------------------------
# 1️⃣ Create folders if not exist
# --------------------------------------------------
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# --------------------------------------------------
# 2️⃣ Train Logistic Regression Model
# --------------------------------------------------
model = LogisticRegression(random_state=42, max_iter=1000)

model.fit(X_train, y_train)


# --------------------------------------------------
# 3️⃣ Predictions
# --------------------------------------------------
y_pred = model.predict(X_test)


# --------------------------------------------------
# 4️⃣ Compute Metrics
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# --------------------------------------------------
# 5️⃣ Save Metrics with Captions
# --------------------------------------------------
metrics_text = f"""
✅ Plant-Watering System – Logistic Regression Model Metrics

Overall Accuracy: {accuracy:.4f}
(This indicates that the model correctly predicted approximately {accuracy * 100:.2f}% of the samples.)

Classification Report:
This table shows precision, recall, and F1-score for each class.

{report}

Confusion Matrix:
Rows represent actual labels, columns represent predicted labels.
"""

with open("reports/logistic_metrics.txt", "w", encoding="utf-8") as f:
    f.write(metrics_text)

print("✅ Logistic Regression metrics saved.")


# --------------------------------------------------
# 6️⃣ Save Confusion Matrix Plot
# --------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Logistic Regression – Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_logistic.png")
plt.close()


# --------------------------------------------------
# 7️⃣ Save Model
# --------------------------------------------------
joblib.dump(model, "models/saved_models/logistic_model.pkl")

print("✅ Logistic Regression model, metrics, and plots saved successfully.")
