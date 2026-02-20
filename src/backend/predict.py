"""
src/backend/predict.py  ─  ML Model Prediction Backend
==========================================================
Loads trained models + scaler, returns predictions with confidence.
This is what the Streamlit frontend calls.
"""

import os, sys, json, time, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    RF_MODEL, XGB_V1, LR_MODEL, SCALER_PATH,
    CLASS_NAMES, CLASS_ICONS, CLASS_COLORS,
    FEATURE_COLS, METRICS_JSON
)
from src.inference.decision_logic import rule_based_decision, get_watering_action


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER  (cached so files are read only once)
# ══════════════════════════════════════════════════════════════════════════════

_cache = {}

def _load_pickle(path: str):
    """Load a pickle file — tries joblib first, then pickle."""
    if path in _cache:
        return _cache[path]
    if not os.path.exists(path):
        return None
    try:
        import joblib
        obj = joblib.load(path)
        _cache[path] = obj
        return obj
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        _cache[path] = obj
        return obj
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None


def load_all_models() -> dict:
    """
    Returns dict of {model_name: model_object}.
    Only includes models whose .pkl files actually exist.
    """
    candidates = {
        "Random Forest":       RF_MODEL,
        "XGBoost":             XGB_V1,
        "Logistic Regression": LR_MODEL,
    }
    loaded = {}
    for name, path in candidates.items():
        m = _load_pickle(path)
        if m is not None:
            loaded[name] = m
    return loaded


def load_scaler():
    return _load_pickle(SCALER_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(sensor_input: dict) -> pd.DataFrame:
    """
    Convert a sensor reading dict into a DataFrame row
    that matches the training feature set (FEATURE_COLS).

    Missing features are filled with sensible defaults.
    """
    defaults = {
        "Plant_ID":                  0,
        "Soil_Moisture":             50.0,
        "Ambient_Temperature":       25.0,
        "Soil_Temperature":          24.0,
        "Humidity":                  60.0,
        "Light_Intensity":           500.0,
        "Soil_pH":                   6.5,
        "Nitrogen_Level":            50.0,
        "Phosphorus_Level":          40.0,
        "Potassium_Level":           45.0,
        "Chlorophyll_Content":       35.0,
        "Electrochemical_Signal":    0.5,
        "days_since_last_watering":  1.0,
        "watering_sma_3":            0.3,
        "Year":                      2026,
        "Month":                     1,
        "Day":                       1,
        "Hour":                      12,
    }
    merged = {**defaults, **sensor_input}
    row = {col: merged.get(col, 0) for col in FEATURE_COLS}
    return pd.DataFrame([row])


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE MODEL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(model, scaler, feature_df: pd.DataFrame) -> dict:
    """
    Run one model on a feature row and return structured result.
    """
    X = feature_df.values.astype(float)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass  # if scaler fails, use raw

    t0 = time.perf_counter()
    pred_class = int(model.predict(X)[0])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Probability / confidence
    confidence = None
    proba_dict = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred_class])
        proba_dict = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {
        "predicted_class": pred_class,
        "label":           CLASS_NAMES[pred_class],
        "icon":            CLASS_ICONS[pred_class],
        "color":           CLASS_COLORS[pred_class],
        "confidence":      confidence,
        "probabilities":   proba_dict,
        "inference_ms":    round(elapsed_ms, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ALL-MODELS COMPARISON PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_all_models(sensor_input: dict) -> dict:
    """
    Main function called by Streamlit frontend.

    Parameters
    ----------
    sensor_input : dict   e.g. {"Soil_Moisture": 25, "Humidity": 40, ...}

    Returns
    -------
    {
        "ml_predictions":   {model_name: prediction_dict, ...},
        "rule_decision":    dict from decision_logic,
        "watering_action":  dict from get_watering_action,
        "recommended":      str  (model name with highest confidence OR rule),
        "feature_vector":   list (the actual values used)
    }
    """
    models  = load_all_models()
    scaler  = load_scaler()
    feat_df = build_feature_vector(sensor_input)

    ml_results = {}
    for name, model in models.items():
        try:
            ml_results[name] = predict_single(model, scaler, feat_df)
        except Exception as e:
            ml_results[name] = {"error": str(e)}

    # Rule-based decision
    rule_result  = rule_based_decision(sensor_input)
    water_action = get_watering_action(rule_result)

    # Best ML model = highest confidence
    best_name = None
    best_conf = -1
    for name, res in ml_results.items():
        c = res.get("confidence") or 0
        if c > best_conf:
            best_conf = c
            best_name = name

    return {
        "ml_predictions":  ml_results,
        "rule_decision":   rule_result,
        "watering_action": water_action,
        "recommended":     best_name or "Rule-Based",
        "feature_vector":  feat_df.values[0].tolist(),
        "feature_names":   FEATURE_COLS,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOAD SAVED METRICS (for comparison page)
# ══════════════════════════════════════════════════════════════════════════════

def load_comparison_metrics() -> dict:
    """Load pre-computed model comparison metrics from JSON."""
    if not os.path.exists(METRICS_JSON):
        return {}
    with open(METRICS_JSON, "r") as f:
        return json.load(f)


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_input = {
        "Soil_Moisture":            22,
        "Ambient_Temperature":      36,
        "Humidity":                 30,
        "Nitrogen_Level":           40,
        "days_since_last_watering": 4,
    }
    result = predict_all_models(test_input)
    print("\n=== ML Predictions ===")
    for name, pred in result["ml_predictions"].items():
        if "error" in pred:
            print(f"  {name}: ERROR - {pred['error']}")
        else:
            print(f"  {name}: {pred['icon']} {pred['label']} ({pred['confidence']*100:.1f}%)")
    print(f"\n=== Rule Decision ===")
    print(f"  {result['rule_decision']['icon']} {result['rule_decision']['label']}")
    print(f"  Action: {result['watering_action']['action']}")
    print(f"\n=== Recommended: {result['recommended']} ===")