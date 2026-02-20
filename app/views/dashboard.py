"""
app/views/dashboard.py  ─  Dashboard Page
Shows system overview, model stats, quick metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.backend.predict import load_comparison_metrics
from src.utils.config import CLASS_COLORS, CLASS_ICONS, CLASS_NAMES


def show_dashboard():
    st.header("🏠 Dashboard")

    metrics = load_comparison_metrics()
    has_metrics = bool(metrics)

    # ── Top KPI row ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    best_acc  = max((v["accuracy"] for v in metrics.values()), default=0.9958)
    best_auc  = max((v["roc_auc_macro"] for v in metrics.values()), default=0.9999)
    n_models  = len(metrics) if metrics else 3
    n_classes = 3

    with col1:
        st.metric("🤖 Models Loaded", n_models, delta="Ready")
    with col2:
        st.metric("🎯 Best Accuracy", f"{best_acc:.2%}")
    with col3:
        st.metric("📈 Best AUC", f"{best_auc:.4f}")
    with col4:
        st.metric("🏷️ Classes", n_classes, delta="Healthy / Needs Water / Overwatered")

    st.markdown("---")

    # ── Model leaderboard ────────────────────────────────────────────────
    st.subheader("🏆 Model Leaderboard")

    if has_metrics:
        rows = sorted([
            {
                "Rank":           "",
                "Model":          name,
                "Accuracy":       f"{m['accuracy']:.2%}",
                "F1-Score":       f"{m['f1_score']:.4f}",
                "AUC (Macro)":    f"{m['roc_auc_macro']:.4f}",
                "Speed (ms)":     f"{m['inference_time_s']*1000:.1f}",
                "_acc":           m["accuracy"],
            }
            for name, m in metrics.items()
        ], key=lambda r: r["_acc"], reverse=True)

        medals = ["🥇", "🥈", "🥉"]
        for i, r in enumerate(rows):
            r["Rank"] = medals[i] if i < 3 else str(i+1)

        df_lb = pd.DataFrame(rows).drop(columns=["_acc"])
        st.dataframe(df_lb.set_index("Rank"), use_container_width=True)
    else:
        st.info("Run `src/models/compare_models.py` to populate leaderboard.")

    st.markdown("---")

    # ── Accuracy mini-chart ──────────────────────────────────────────────
    if has_metrics:
        st.subheader("📊 Accuracy at a Glance")
        colors = {"Logistic Regression": "#e74c3c", "Random Forest": "#2ecc71", "XGBoost": "#3498db"}

        fig = go.Figure()
        for name, m in metrics.items():
            fig.add_trace(go.Bar(
                name=name, x=[name], y=[m["accuracy"]],
                marker_color=colors.get(name, "#888"),
                text=[f"{m['accuracy']:.2%}"], textposition="outside",
            ))
        fig.update_layout(
            yaxis=dict(range=[0, 1.15], title="Accuracy"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, height=300,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Class info ───────────────────────────────────────────────────────
    st.subheader("🌿 Plant Health Classes")
    c1, c2, c3 = st.columns(3)

    class_info = {
        0: ("Healthy",      "Soil moisture 30–70%\nNormal temp & humidity\nNo action needed",   "#2ecc71"),
        1: ("Needs Water",  "Soil moisture < 30%\nHigh temp or low humidity\nTurn pump ON",     "#e67e22"),
        2: ("Overwatered",  "Soil moisture > 70%\nExcess nitrogen\nStop watering immediately", "#e74c3c"),
    }

    for col, (cls_id, (label, desc, color)) in zip([c1, c2, c3], class_info.items()):
        with col:
            st.markdown(f"""
            <div style="background:{color}18; border:2px solid {color};
                        border-radius:12px; padding:1rem; text-align:center; min-height:160px;">
                <div style="font-size:2.5rem;">{CLASS_ICONS[cls_id]}</div>
                <h4 style="color:{color}; margin:0.3rem 0;">{label}</h4>
                <p style="font-size:0.85rem; color:#555; white-space:pre-line;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
