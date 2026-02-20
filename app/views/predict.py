"""
app/views/predict.py  ─  Predict & Diagnose Page
Connected to: src/backend/predict.py  +  src/inference/decision_logic.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.backend.predict import predict_all_models


def show_predict():
    st.header("🔬 Predict & Diagnose Plant Health")
    st.markdown("Enter sensor readings below — backend will run **all 3 ML models** + **rule-based decision engine**.")

    # ── Input Form ──────────────────────────────────────────────────────────
    st.markdown("### 📥 Sensor Input")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🌊 Soil & Water**")
            soil_moisture  = st.slider("Soil Moisture (%)",       0,   100, 45)
            soil_temp      = st.slider("Soil Temperature (°C)",    0,    50, 24)
            soil_ph        = st.slider("Soil pH",                 3.0,  9.0, 6.5, step=0.1)
            days_watered   = st.slider("Days Since Last Watering", 0,   14,  1)

        with col2:
            st.markdown("**🌡️ Environment**")
            ambient_temp   = st.slider("Ambient Temperature (°C)", 0,   50, 26)
            humidity       = st.slider("Humidity (%)",             0,  100, 60)
            light_intensity= st.slider("Light Intensity (lux)",    0, 2000, 600)

        with col3:
            st.markdown("**🧪 Nutrients**")
            nitrogen       = st.slider("Nitrogen Level",    0, 100, 50)
            phosphorus     = st.slider("Phosphorus Level",  0, 100, 40)
            potassium      = st.slider("Potassium Level",   0, 100, 45)
            chlorophyll    = st.slider("Chlorophyll",       0,  80, 35)
            electro        = st.slider("Electrochemical Signal", 0.0, 2.0, 0.5, step=0.05)

        submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True, type="primary")

    # ── Prediction ──────────────────────────────────────────────────────────
    if submitted:
        sensor_input = {
            "Soil_Moisture":            soil_moisture,
            "Ambient_Temperature":      ambient_temp,
            "Soil_Temperature":         soil_temp,
            "Humidity":                 humidity,
            "Light_Intensity":          light_intensity,
            "Soil_pH":                  soil_ph,
            "Nitrogen_Level":           nitrogen,
            "Phosphorus_Level":         phosphorus,
            "Potassium_Level":          potassium,
            "Chlorophyll_Content":      chlorophyll,
            "Electrochemical_Signal":   electro,
            "days_since_last_watering": days_watered,
            "watering_sma_3":           round(1 / max(days_watered, 1), 2),
            "Year": 2026, "Month": 1, "Day": 1, "Hour": 12,
        }

        with st.spinner("🤖 Running ML models + Rule Engine..."):
            result = predict_all_models(sensor_input)

        st.markdown("---")

        # ── 1. Watering Action Banner ────────────────────────────────────
        action   = result["watering_action"]
        rule_dec = result["rule_decision"]

        urgency_color = {"HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#2ecc71"}
        color = urgency_color.get(action["urgency"], "#2ecc71")

        st.markdown(f"""
        <div style="background:{color}20; border-left:6px solid {color};
                    padding:1.2rem; border-radius:10px; margin-bottom:1rem;">
            <h2 style="color:{color}; margin:0;">
                {rule_dec['icon']} {action['action']}
            </h2>
            <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">
                <b>Pump:</b> {"🔵 ON" if action["pump_on"] else "⚫ OFF"} &nbsp;|&nbsp;
                <b>Duration:</b> {action["duration_minutes"]} min &nbsp;|&nbsp;
                <b>Urgency:</b> {action["urgency"]}
            </p>
            <p style="margin:0.3rem 0 0 0; color:#555;">{action["advice"]}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── 2. ML Model Results ─────────────────────────────────────────
        st.markdown("### 🤖 ML Model Predictions")
        ml_preds = result["ml_predictions"]

        cols = st.columns(len(ml_preds))
        for i, (model_name, pred) in enumerate(ml_preds.items()):
            with cols[i]:
                if "error" in pred:
                    st.error(f"**{model_name}**\n\nError: {pred['error']}")
                else:
                    conf_pct = f"{pred['confidence']*100:.1f}%" if pred["confidence"] else "N/A"
                    st.markdown(f"""
                    <div style="background:{pred['color']}15; border:2px solid {pred['color']};
                                border-radius:12px; padding:1rem; text-align:center;">
                        <h4 style="margin:0; color:#333;">{model_name}</h4>
                        <div style="font-size:2.5rem; margin:0.5rem 0;">{pred['icon']}</div>
                        <h3 style="color:{pred['color']}; margin:0;">{pred['label']}</h3>
                        <p style="color:#666; margin:0.3rem 0;">Confidence: <b>{conf_pct}</b></p>
                        <p style="color:#888; font-size:0.8rem;">⏱ {pred['inference_ms']:.1f} ms</p>
                    </div>
                    """, unsafe_allow_html=True)

        # ── 3. Rule-Based Decision ──────────────────────────────────────
        st.markdown("### 📋 Rule-Based Decision Engine")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            rd = result["rule_decision"]
            st.markdown(f"""
            <div style="background:{rd['color']}15; border:2px solid {rd['color']};
                        border-radius:12px; padding:1.2rem; text-align:center;">
                <div style="font-size:3rem;">{rd['icon']}</div>
                <h3 style="color:{rd['color']};">{rd['label']}</h3>
                <p>Confidence: <b>{rd['confidence']*100:.1f}%</b></p>
                <p style="font-size:0.85rem; color:#666;">
                    Scores → Healthy:{rd['scores'][0]:.0f}
                    Needs Water:{rd['scores'][1]:.0f}
                    Overwatered:{rd['scores'][2]:.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("**✅ Triggered Rules:**")
            for r in rd["triggered_rules"]:
                lbl = {0:"🟢", 1:"🟡", 2:"🔴"}.get(r["vote"], "⚪")
                st.markdown(f"- {lbl} **[{r['id']}]** {r['label']} *(weight={r['weight']})* — {r['reason']}")

            if rd["skipped_rules"]:
                with st.expander(f"❌ Skipped Rules ({len(rd['skipped_rules'])})"):
                    for r in rd["skipped_rules"]:
                        st.markdown(f"- ⬜ [{r['id']}] {r['label']}")

        # ── 4. Probability Chart ────────────────────────────────────────
        st.markdown("### 📊 Model Probability Comparison")

        chart_data = []
        for model_name, pred in ml_preds.items():
            if "error" not in pred and pred.get("probabilities"):
                for cls_label, prob in pred["probabilities"].items():
                    chart_data.append({
                        "Model": model_name,
                        "Class": cls_label,
                        "Probability": prob
                    })

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig = px.bar(
                df_chart, x="Class", y="Probability", color="Model",
                barmode="group",
                color_discrete_sequence=["#2ecc71", "#3498db", "#e67e22"],
                title="Prediction Probabilities by Model",
            )
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_range=[0, 1],
                font=dict(size=13),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 5. Input Summary ────────────────────────────────────────────
        with st.expander("🔍 View Full Feature Vector Sent to Models"):
            feat_df = pd.DataFrame({
                "Feature": result["feature_names"],
                "Value":   result["feature_vector"],
            })
            st.dataframe(feat_df, use_container_width=True)

        # ── 6. Recommended Model ────────────────────────────────────────
        st.success(f"🏆 **Recommended Model:** {result['recommended']} (highest confidence)")
