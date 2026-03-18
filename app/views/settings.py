"""Settings page view for configuring rule engine thresholds and viewing model status."""

import streamlit as st
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.utils.config import (
    DEFAULT_RULE_THRESHOLDS,
    RULE_THRESHOLDS,
    RF_MODEL,
    XGB_V1,
    LR_MODEL,
    merge_rule_thresholds,
)


def show_settings():
    """Display the settings page with rule engine thresholds and model file status."""
    st.header("Settings")

    if "rule_thresholds" not in st.session_state:
        st.session_state.rule_thresholds = merge_rule_thresholds(RULE_THRESHOLDS)

    st.subheader("Rule Engine Thresholds")
    st.markdown(
        "Tune these thresholds to test how the **Rule-Based Decision Engine** responds "
        "without changing source code."
    )

    active = st.session_state.rule_thresholds.copy()

    col1, col2 = st.columns(2)
    with col1:
        soil_low = st.slider("Soil Moisture LOW (%)", 0, 60, int(active["soil_moisture_low"]))
        soil_high = st.slider("Soil Moisture HIGH (%)", max(soil_low + 1, 40), 100, int(max(active["soil_moisture_high"], soil_low + 1)))
        temp_high = st.slider("Temperature HIGH (°C)", 10, 55, int(active["temperature_high"]))

    with col2:
        humidity_low = st.slider("Humidity LOW (%)", 0, 80, int(active["humidity_low"]))
        days_since = st.slider("Days Since Water", 0, 14, int(active["days_since_water"]))

    updated_thresholds = {
        "soil_moisture_low": soil_low,
        "soil_moisture_high": soil_high,
        "temperature_high": temp_high,
        "humidity_low": humidity_low,
        "days_since_water": days_since,
    }

    st.session_state.rule_thresholds = merge_rule_thresholds(updated_thresholds)

    btn_col1, btn_col2 = st.columns([1, 2])
    with btn_col1:
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state.rule_thresholds = DEFAULT_RULE_THRESHOLDS.copy()
            st.rerun()
    with btn_col2:
        st.caption("Threshold updates apply immediately on the Predict page.")

    st.markdown("Current active thresholds:")
    st.json(st.session_state.rule_thresholds)

    st.markdown("---")
    st.subheader("Model File Status")

    for label, path in [
        ("Random Forest", RF_MODEL),
        ("XGBoost", XGB_V1),
        ("Logistic Regression", LR_MODEL),
    ]:
        exists = os.path.exists(path)
        icon = "✅" if exists else "❌"
        st.markdown(f"{icon} **{label}** — `{os.path.basename(path)}`")

    st.markdown("---")
    st.info("To retrain models, run the scripts in `src/models/` folder.")
