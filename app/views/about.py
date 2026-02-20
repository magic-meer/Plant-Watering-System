"""app/views/about.py"""
import streamlit as st


def show_about():
    st.header("ℹ️ About")

    st.markdown("""
    ## 🌱 Plant Watering Intelligence System

    An AI-powered plant health monitoring and watering decision system.

    ---

    ### 🏗️ Architecture
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Frontend (Streamlit)**
        - `app/app.py` — Main entry point
        - `app/views/dashboard.py` — Overview
        - `app/views/predict.py` — ML + Rule predictions
        - `app/views/model_comparison.py` — Compare models
        - `app/views/dataset_visualization.py` — Data EDA
        - `app/views/settings.py` — Configuration
        """)

    with col2:
        st.markdown("""
        **Backend (Python)**
        - `src/backend/predict.py` — Model loader + inference
        - `src/inference/decision_logic.py` — Rule engine
        - `src/utils/config.py` — Central config
        - `src/models/` — Training scripts
        - `models/` — Saved .pkl files
        """)

    st.markdown("---")

    st.markdown("""
    ### 🤖 Models
    | Model | Accuracy | Best For |
    |-------|----------|----------|
    | Logistic Regression | 71.25% | Baseline / Interpretability |
    | Random Forest | 100.00% | High accuracy |
    | XGBoost | 99.58% | **Recommended** for production |

    ### 🏷️ Classes
    - ✅ **Healthy (0)** — Soil moisture 30–70%, normal conditions
    - 💧 **Needs Water (1)** — Low moisture, high temp, low humidity
    - ⚠️ **Overwatered (2)** — Too much moisture, excess nitrogen

    ### 🛠️ Tech Stack
    - Python 3.10+, scikit-learn, XGBoost
    - Streamlit, Plotly, Pandas
    - Pickle model serialization

    ---
    *Plant Watering Intelligence System © 2026*
    """)
