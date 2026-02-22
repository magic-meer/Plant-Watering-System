"""
Central configuration module for Plant Watering System.

Contains all paths, model settings, and rule thresholds.
"""

import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODELS = os.path.join(MODELS_DIR, "saved_models")
SCALER_DIR = os.path.join(MODELS_DIR, "scaler")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures", "model_comparison")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# File Paths
RAW_CSV = os.path.join(RAW_DATA_DIR, "original_dataset.csv")
CLEANED_CSV = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
X_TRAIN_CSV = os.path.join(PROCESSED_DIR, "X_train.csv")
X_TEST_CSV = os.path.join(PROCESSED_DIR, "X_test.csv")
Y_TRAIN_CSV = os.path.join(PROCESSED_DIR, "y_train.csv")
Y_TEST_CSV = os.path.join(PROCESSED_DIR, "y_test.csv")

RF_MODEL = os.path.join(MODELS_DIR, "random_forest_model.pkl")
RF_V1 = os.path.join(MODELS_DIR, "random_forest_v1.pkl")
XGB_V1 = os.path.join(MODELS_DIR, "xgboost_v1.pkl")
LR_MODEL = os.path.join(SAVED_MODELS, "logistic_model.pkl")
SCALER_PATH = os.path.join(SCALER_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(SCALER_DIR, "label_encoder.pkl")

METRICS_JSON = os.path.join(REPORTS_DIR, "model_comparison_metrics.json")
REPORT_MD = os.path.join(REPORTS_DIR, "model_comparison_report.md")
SUMMARY_CSV = os.path.join(REPORTS_DIR, "model_comparison_summary.csv")

# Model Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "Plant_Health_Status"

CLASS_NAMES = {0: "Healthy", 1: "Needs Water", 2: "Overwatered"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e67e22", 2: "#e74c3c"}
CLASS_ICONS = {0: "✅", 1: "💧", 2: "⚠️"}

FEATURE_COLS = [
    "Soil_Moisture", "Ambient_Temperature", "Soil_Temperature",
    "Humidity", "Light_Intensity", "Soil_pH",
    "Nitrogen_Level", "Phosphorus_Level", "Potassium_Level",
    "Chlorophyll_Content", "Electrochemical_Signal",
    "days_since_last_watering", "watering_sma_3",
    "Year", "Month", "Day", "Hour"
]

# Rule Engine Thresholds
RULE_THRESHOLDS = {
    "soil_moisture_low": 30,    # below -> Needs Water
    "soil_moisture_high": 70,   # above -> Overwatered
    "temperature_high": 38,     # above -> stress flag
    "humidity_low": 35,         # below -> dry air
    "days_since_water": 3,      # above -> likely needs water
}
