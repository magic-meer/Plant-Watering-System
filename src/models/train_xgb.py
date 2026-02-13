import os
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# 0️⃣ Load Data
# --------------------------------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# --------------------------------------------------
# Handle Timestamp column (if exists)
# --------------------------------------------------
if 'Timestamp' in X_train.columns:
    for df in [X_train, X_test]:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Extract useful datetime features
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour
        # Drop original Timestamp column
        df.drop('Timestamp', axis=1, inplace=True)

# --------------------------------------------------
# 1️⃣ Create folders if not exist
# --------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# --------------------------------------------------
# 2️⃣ Train XGBoost Model
# --------------------------------------------------
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
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
# 5️⃣ Save Metrics with Captions (UTF-8 encoding to handle special characters)
# --------------------------------------------------
metrics_text = f"""
✅ Plant-Watering System – XGBoost Model Metrics

Overall Accuracy: {accuracy:.4f}
(This indicates that the model correctly predicted approximately {accuracy*100:.2f}% of the samples.)

Classification Report:
This table shows precision, recall, and F1-score for each class.
- Class 0: Healthy plant
- Class 1: Needs Water
- Class 2: Overwatered

{report}

Confusion Matrix:
Rows represent actual labels, columns represent predicted labels.
- Top-left cell: True Positives for Class 0
- Bottom-right cell: True Positives for Class 2
"""

with open("reports/metrics.txt", "w", encoding="utf-8") as f:
    f.write(metrics_text)

print("✅ Metrics saved with captions.")


# --------------------------------------------------
# 6️⃣ Save Confusion Matrix Plot with Caption
# --------------------------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Actual vs Predicted Labels", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)
plt.text(0, -1.2, "Rows=Actual, Columns=Predicted", fontsize=10, color='gray')
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

# --------------------------------------------------
# 7️⃣ Save Feature Importance Plot with Caption
# --------------------------------------------------
plt.figure(figsize=(8,6))
xgb.plot_importance(model, importance_type='weight', max_num_features=20)
plt.title("Feature Importance (Top 20)", fontsize=14)
plt.xlabel("F Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.text(0, -1, "Features with higher bars are more important for prediction", fontsize=10, color='gray')
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

# --------------------------------------------------
# 8️⃣ Save Model
# --------------------------------------------------
joblib.dump(model, "models/xgboost_v1.pkl")

print("✅ Model, metrics, and plots saved successfully.")
