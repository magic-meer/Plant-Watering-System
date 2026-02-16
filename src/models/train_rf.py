import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 0️⃣ Load Data
# --------------------------------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()


# --------------------------------------------------
# Handle Timestamp column (if exists)
# --------------------------------------------------
if 'Timestamp' in X_train.columns:
    for df in [X_train, X_test]:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Year']  = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day']   = df['Timestamp'].dt.day
        df['Hour']  = df['Timestamp'].dt.hour
        df.drop('Timestamp', axis=1, inplace=True)


# --------------------------------------------------
# 1️⃣ Create folders if not exist
# --------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# --------------------------------------------------
# 2️⃣ Train Random Forest Model
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

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
✅ Plant-Watering System – Random Forest Model Metrics

Overall Accuracy: {accuracy:.4f}
(This indicates that the model correctly predicted approximately {accuracy*100:.2f}% of the samples.)

Classification Report:
This table shows precision, recall, and F1-score for each class.

{report}

Confusion Matrix:
Rows represent actual labels, columns represent predicted labels.
"""

with open("reports/rf_metrics.txt", "w", encoding="utf-8") as f:
    f.write(metrics_text)

print("✅ Random Forest metrics saved.")


# --------------------------------------------------
# 6️⃣ Save Confusion Matrix Plot
# --------------------------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest – Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_rf.png")
plt.close()


# --------------------------------------------------
# 7️⃣ Save Feature Importance Plot
# --------------------------------------------------
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(20)
)
plt.title("Random Forest – Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("plots/feature_importance_rf.png")
plt.close()


# --------------------------------------------------
# 8️⃣ Save Model
# --------------------------------------------------
joblib.dump(model, "models/random_forest_v1.pkl")

print("✅ Random Forest model, metrics, and plots saved successfully.")
