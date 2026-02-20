"""
app/views/settings.py
app/views/about.py
"""

import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.utils.config import RULE_THRESHOLDS, RF_MODEL, XGB_V1, LR_MODEL


def show_settings():
    st.header("⚙️ Settings")

    st.subheader("🔧 Rule Engine Thresholds")
    st.markdown("These thresholds are used by the **Rule-Based Decision Engine** in `src/inference/decision_logic.py`.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        | Threshold | Value |
        |-----------|-------|
        | Soil Moisture LOW  | `{RULE_THRESHOLDS['soil_moisture_low']}%` |
        | Soil Moisture HIGH | `{RULE_THRESHOLDS['soil_moisture_high']}%` |
        | Temperature HIGH   | `{RULE_THRESHOLDS['temperature_high']}°C` |
        | Humidity LOW       | `{RULE_THRESHOLDS['humidity_low']}%` |
        | Days Since Water   | `{RULE_THRESHOLDS['days_since_water']} days` |
        """)

    with col2:
        st.markdown("**To change thresholds**, edit `src/utils/config.py`:")
        st.code("""
RULE_THRESHOLDS = {
    "soil_moisture_low":  30,
    "soil_moisture_high": 70,
    "temperature_high":   38,
    "humidity_low":       35,
    "days_since_water":    3,
}
        """, language="python")

    st.markdown("---")
    st.subheader("📁 Model File Status")

    for label, path in [
        ("Random Forest",       RF_MODEL),
        ("XGBoost",             XGB_V1),
        ("Logistic Regression", LR_MODEL),
    ]:
        exists = os.path.exists(path)
        icon   = "✅" if exists else "❌"
        st.markdown(f"{icon} **{label}** — `{os.path.basename(path)}`")

    st.markdown("---")
    st.info("💡 To retrain models, run the scripts in `src/models/` folder.")
