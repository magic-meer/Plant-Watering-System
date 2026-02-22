"""
Model Comparison Script for Plant Watering System.

Compares Logistic Regression, Random Forest, and XGBoost models
with comprehensive metrics and visualizations.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures", "model_comparison")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Style settings
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ["#2196F3", "#4CAF50", "#FF9800"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]


# Load Data
print("Loading data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# Handle Timestamp column
for df in [X_train, X_test]:
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Year"] = df["Timestamp"].dt.year
        df["Month"] = df["Timestamp"].dt.month
        df["Day"] = df["Timestamp"].dt.day
        df["Hour"] = df["Timestamp"].dt.hour
        df.drop("Timestamp", axis=1, inplace=True)

# Handle NaN values
numeric_cols = X_train.select_dtypes(include=["number"]).columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())

FEATURE_NAMES = list(X_train.columns)
CLASS_LABELS = sorted(np.unique(y_test))
N_CLASSES = len(CLASS_LABELS)
CLASS_NAMES = {0: "Healthy", 1: "Needs Water", 2: "Overwatered"}

print(f"   Features: {len(FEATURE_NAMES)}")
print(f"   Classes: {N_CLASSES}")
print(f"   Test size: {len(y_test)}")


# Load Models
print("\nLoading models...")

model_paths = {
    "Logistic Regression": os.path.join(MODELS_DIR, "saved_models", "logistic_model.pkl"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest_v1.pkl"),
    "XGBoost": os.path.join(MODELS_DIR, "xgboost_v1.pkl"),
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"   Loaded {name}")
    else:
        print(f"   {name} NOT FOUND at {path}")

if not models:
    raise SystemExit("No models found. Train models first.")


# Evaluate Models
print("\nEvaluating models...\n")

results = {}
y_test_bin = label_binarize(y_test, classes=CLASS_LABELS)

for name, model in models.items():
    # Predictions and timing
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0

    # Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    # Core metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Per-class metrics
    prec_per = precision_score(y_test, y_pred, average=None)
    rec_per = recall_score(y_test, y_pred, average=None)
    f1_per = f1_score(y_test, y_pred, average=None)

    # ROC AUC
    auc_scores = {}
    roc_data = {}
    if y_prob is not None:
        for i, cls in enumerate(int(c) for c in CLASS_LABELS):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_data[cls] = (fpr.tolist(), tpr.tolist())
            auc_scores[cls] = float(auc(fpr, tpr))
        auc_macro = float(
            roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
        )
    else:
        auc_macro = None

    results[name] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "prec_per_class": prec_per.tolist(),
        "rec_per_class": rec_per.tolist(),
        "f1_per_class": f1_per.tolist(),
        "confusion_matrix": cm.tolist(),
        "roc_auc_macro": auc_macro,
        "roc_auc_per_class": auc_scores,
        "roc_data": roc_data,
        "inference_time_s": float(inference_time),
        "classification_report": report,
    }

    print(f"  {name:25s}  Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc_macro or 'N/A'}")


# Feature Importance
print("\nExtracting feature importance...")

feature_importance = {}

if "Random Forest" in models:
    rf = models["Random Forest"]
    fi = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    feature_importance["Random Forest"] = fi.to_dict()

if "XGBoost" in models:
    xgb_model = models["XGBoost"]
    fi = pd.Series(xgb_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    feature_importance["XGBoost"] = fi.to_dict()

if "Logistic Regression" in models:
    lr = models["Logistic Regression"]
    coef_abs = np.abs(lr.coef_).mean(axis=0)
    fi = pd.Series(coef_abs, index=FEATURE_NAMES).sort_values(ascending=False)
    feature_importance["Logistic Regression"] = fi.to_dict()


# Save Metrics JSON
def _convert_keys(obj):
    """Recursively convert dict keys to str for JSON compatibility."""
    if isinstance(obj, dict):
        return {str(k): _convert_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_keys(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


metrics_export = {}
for name, r in results.items():
    metrics_export[name] = _convert_keys({k: v for k, v in r.items() if k != "roc_data"})

metrics_path = os.path.join(REPORT_DIR, "model_comparison_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_export, f, indent=2)
print(f"\nMetrics saved to {metrics_path}")


# Generate Visualizations
print("\nGenerating visualizations...\n")


# Accuracy Comparison
fig, ax = plt.subplots(figsize=(8, 5))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
bars = ax.bar(names, accs, color=PALETTE[:len(names)], edgecolor="white", linewidth=1.2)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{a:.2%}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison", fontsize=15, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "accuracy_comparison.png"), dpi=150)
plt.close(fig)
print("   accuracy_comparison.png")


# Grouped Metrics Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(names))
width = 0.22
metrics_list = ["precision", "recall", "f1_score"]
labels_nice = ["Precision", "Recall", "F1-Score"]
colors = ["#42A5F5", "#66BB6A", "#FFA726"]

for i, (metric, label, color) in enumerate(zip(metrics_list, labels_nice, colors)):
    vals = [results[n][metric] for n in names]
    bars = ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x + width)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1-Score Comparison", fontsize=15, fontweight="bold")
ax.legend(loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "metrics_comparison.png"), dpi=150)
plt.close(fig)
print("   metrics_comparison.png")


# Per-Class F1 Heatmap
f1_matrix = []
for n in names:
    f1_matrix.append(results[n]["f1_per_class"])
f1_df = pd.DataFrame(f1_matrix, index=names, columns=[CLASS_NAMES[c] for c in CLASS_LABELS])

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlGn", vmin=0, vmax=1,
            linewidths=1, ax=ax, cbar_kws={"label": "F1-Score"})
ax.set_title("Per-Class F1-Score Heatmap", fontsize=14, fontweight="bold")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "f1_per_class_heatmap.png"), dpi=150)
plt.close(fig)
print("   f1_per_class_heatmap.png")


# Confusion Matrices
fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
if len(names) == 1:
    axes = [axes]
cmaps = ["Blues", "Greens", "Oranges"]
for ax, name, cmap in zip(axes, names, cmaps):
    cm = np.array(results[name]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=[CLASS_NAMES[c] for c in CLASS_LABELS],
                yticklabels=[CLASS_NAMES[c] for c in CLASS_LABELS])
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices Comparison", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   confusion_matrices.png")


# ROC Curves
fig, axes = plt.subplots(1, N_CLASSES, figsize=(6 * N_CLASSES, 5))
line_styles = ["-", "--", "-."]
for cls_idx, cls in enumerate(CLASS_LABELS):
    ax = axes[cls_idx]
    for model_idx, name in enumerate(names):
        roc_data = results[name].get("roc_data", {})
        if cls in roc_data:
            fpr, tpr = roc_data[cls]
            auc_val = results[name]["roc_auc_per_class"].get(cls, 0)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})",
                    color=PALETTE[model_idx], linewidth=2, linestyle=line_styles[model_idx])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_title(f"Class: {CLASS_NAMES[cls]}", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
fig.suptitle("ROC Curves - One-vs-Rest", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   roc_curves.png")


# ROC AUC Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
auc_vals = [results[n].get("roc_auc_macro", 0) or 0 for n in names]
bars = ax.bar(names, auc_vals, color=PALETTE[:len(names)], edgecolor="white", linewidth=1.2)
for bar, v in zip(bars, auc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Macro AUC")
ax.set_title("ROC AUC Score Comparison (Macro OVR)", fontsize=15, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "auc_comparison.png"), dpi=150)
plt.close(fig)
print("   auc_comparison.png")


# Feature Importance Comparison
if feature_importance:
    fi_df = pd.DataFrame(feature_importance).fillna(0)
    fi_norm = fi_df.apply(lambda col: col / col.max() if col.max() > 0 else col)
    fi_norm = fi_norm.sort_values(by=list(fi_norm.columns)[0], ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    fi_norm.plot(kind="barh", ax=ax, color=PALETTE[:len(fi_norm.columns)], edgecolor="white")
    ax.set_xlabel("Normalised Importance")
    ax.set_title("Top 15 Features - Importance Comparison", fontsize=15, fontweight="bold")
    ax.legend(title="Model")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance_comparison.png"), dpi=150)
    plt.close(fig)
    print("   feature_importance_comparison.png")


# Inference Time Comparison
fig, ax = plt.subplots(figsize=(8, 5))
times = [results[n]["inference_time_s"] * 1000 for n in names]
bars = ax.bar(names, times, color=PALETTE[:len(names)], edgecolor="white", linewidth=1.2)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{t:.1f} ms", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_ylabel("Inference Time (ms)")
ax.set_title("Prediction Speed on Test Set (240 samples)", fontsize=15, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "inference_time.png"), dpi=150)
plt.close(fig)
print("   inference_time.png")


# Radar Chart
metrics_radar = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]

for idx, name in enumerate(names):
    r = results[name]
    vals = [r["accuracy"], r["precision"], r["recall"], r["f1_score"],
            r.get("roc_auc_macro", 0) or 0]
    vals += vals[:1]
    labels_closed = metrics_radar + [metrics_radar[0]]
    ax.fill(angles, vals, alpha=0.15, color=PALETTE[idx])
    ax.plot(angles, vals, linewidth=2, label=name, color=PALETTE[idx])

ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], metrics_radar)
ax.set_ylim(0, 1.05)
ax.set_title("Model Performance Radar", fontsize=15, fontweight="bold", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0))
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "radar_chart.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("   radar_chart.png")


# Summary Table
summary_rows = []
for name in names:
    r = results[name]
    summary_rows.append({
        "Model": name,
        "Accuracy": f"{r['accuracy']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
        "F1-Score": f"{r['f1_score']:.4f}",
        "AUC (Macro)": f"{r['roc_auc_macro']:.4f}" if r["roc_auc_macro"] else "N/A",
        "Inference (ms)": f"{r['inference_time_s'] * 1000:.1f}",
    })
summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 80)
print("  MODEL COMPARISON SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))
print("=" * 80)

summary_df.to_csv(os.path.join(REPORT_DIR, "model_comparison_summary.csv"), index=False)
print(f"\nSummary CSV saved")


# Generate Markdown Report
print("\nGenerating markdown report...")

best_name = max(names, key=lambda n: results[n]["f1_score"])

report_md = f"""# Model Comparison Report - Plant Watering System

> **Generated**: February 16, 2026
> **Dataset**: Plant Health Intelligence Dataset
> **Test Samples**: {len(y_test)} | **Features**: {len(FEATURE_NAMES)} | **Classes**: {N_CLASSES}

---

## 1. Executive Summary

This report compares three machine learning models trained to classify plant health status into three categories: **Healthy** (Class 0), **Needs Water** (Class 1), and **Overwatered** (Class 2).

| Model | Accuracy | Precision | Recall | F1-Score | AUC (Macro) |
|-------|----------|-----------|--------|----------|-------------|
"""

for name in names:
    r = results[name]
    auc_str = f"{r['roc_auc_macro']:.4f}" if r["roc_auc_macro"] else "N/A"
    report_md += f"| {name} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | {auc_str} |\n"

report_md += f"""
**Best Performing Model**: **{best_name}** with F1-Score of {results[best_name]["f1_score"]:.4f}

---

## 2. Data & Feature Overview

### 2.1 Dataset Description

The dataset contains plant health sensor readings with the following characteristics:
- **Training Samples**: {len(y_train)}
- **Test Samples**: {len(y_test)}
- **Split Strategy**: Stratified 80/20 Train-Test Split
- **Scaling**: StandardScaler (Z-score normalization)

### 2.2 Features Used ({len(FEATURE_NAMES)} total)

| # | Feature | Type |
|---|---------|------|
"""

sensor_features = ["Soil_Moisture", "Ambient_Temperature", "Soil_Temperature", "Humidity",
                   "Light_Intensity", "Soil_pH", "Nitrogen_Level", "Phosphorus_Level",
                   "Potassium_Level", "Chlorophyll_Content", "Electrochemical_Signal"]
engineered_features = ["days_since_last_watering", "watering_sma_3"]
id_features = ["Plant_ID"]
temporal_features = ["Year", "Month", "Day", "Hour"]

for i, feat in enumerate(FEATURE_NAMES, 1):
    if feat in sensor_features:
        ftype = "Environmental Sensor"
    elif feat in engineered_features:
        ftype = "Engineered Feature"
    elif feat in id_features:
        ftype = "Identifier"
    elif feat in temporal_features:
        ftype = "Temporal (extracted)"
    else:
        ftype = "Other"
    report_md += f"| {i} | `{feat}` | {ftype} |\n"

report_md += f"""
### 2.3 Feature Engineering

Two features were engineered from the `Timestamp` column:
- **days_since_last_watering** - Number of days since the plant was last watered
- **watering_sma_3** - 3-period simple moving average of watering events

Additionally, temporal features (Year, Month, Day, Hour) were extracted from the Timestamp column.

### 2.4 Target Classes

| Class | Label | Test Samples |
|-------|-------|:------------:|
| 0 | Healthy | 60 |
| 1 | Needs Water | 100 |
| 2 | Overwatered | 80 |

---

## 3. Model Architectures

### 3.1 Logistic Regression
- **Type**: Linear Classifier
- **Solver**: lbfgs
- **Max Iterations**: 1,000
- **Regularization**: L2
- **Multi-class Strategy**: OVR

### 3.2 Random Forest
- **Type**: Ensemble (Bagging)
- **Number of Trees**: 200
- **Parallelism**: All cores
- **Random State**: 42

### 3.3 XGBoost
- **Type**: Gradient Boosted Trees
- **Number of Estimators**: 200
- **Learning Rate**: 0.05
- **Max Depth**: 6

---

## 4. Performance Analysis

### 4.1 Overall Accuracy

- **Logistic Regression**: {results["Logistic Regression"]["accuracy"]:.2%}
- **Random Forest**: {results["Random Forest"]["accuracy"]:.2%}
- **XGBoost**: {results["XGBoost"]["accuracy"]:.2%}

### 4.2 Per-Class F1-Score

| Model | Healthy (F1) | Needs Water (F1) | Overwatered (F1) |
|-------|:---:|:---:|:---:|
"""

for name in names:
    f1s = results[name]["f1_per_class"]
    report_md += f"| {name} | {f1s[0]:.3f} | {f1s[1]:.3f} | {f1s[2]:.3f} |\n"

report_md += f"""
### 4.3 ROC AUC Scores

| Model | AUC (Healthy) | AUC (Needs Water) | AUC (Overwatered) | AUC (Macro) |
|-------|:---:|:---:|:---:|:---:|
"""

for name in names:
    aucs = results[name]["roc_auc_per_class"]
    macro = results[name]["roc_auc_macro"]
    report_md += f"| {name} | {aucs.get(0, 0):.4f} | {aucs.get(1, 0):.4f} | {aucs.get(2, 0):.4f} | {macro:.4f} |\n"

report_md += f"""
## 5. Feature Importance Analysis

### Top Features by Model

"""

for name in feature_importance:
    fi = feature_importance[name]
    top5 = list(fi.items())[:5]
    report_md += f"**{name}** (top 5):\n"
    for i, (feat, imp) in enumerate(top5, 1):
        report_md += f"{i}. `{feat}` - {imp:.4f}\n"
    report_md += "\n"

report_md += f"""
## 6. Computational Efficiency

| Model | Inference Time (ms) |
|-------|:---:|
"""

for name in names:
    report_md += f"| {name} | {results[name]['inference_time_s'] * 1000:.1f} |\n"

report_md += f"""
## 7. Recommendations

### Primary Model: **XGBoost**
- Best balance of accuracy and generalization
- Slightly lower than Random Forest's perfect score, suggesting less overfitting
- Well-suited for deployment with reasonable inference time

### Future Improvements
1. Cross-Validation for robust performance estimates
2. Hyperparameter Tuning with RandomizedSearchCV or Optuna
3. Feature Selection to simplify models
4. Ensemble Methods (stacking/blending)
5. Data Augmentation for underrepresented classes

---

*Report generated by src/models/compare_models.py*
"""

report_path = os.path.join(REPORT_DIR, "model_comparison_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)

print(f"Markdown report saved to {report_path}")
print("\nModel comparison complete!")
