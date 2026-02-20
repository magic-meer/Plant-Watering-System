"""
app/views/model_comparison.py  ─  Model Comparison Page
Reads: reports/model_comparison_metrics.json (real pre-computed data)
Shows: Accuracy, F1, AUC, Confusion Matrix, Inference Time, Radar Chart
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys, os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.backend.predict import load_comparison_metrics
from src.utils.config import FIGURES_DIR, REPORTS_DIR, CLASS_NAMES


# ── Helper to load pre-saved plot images ──────────────────────────────────
def _img(name):
    path = os.path.join(FIGURES_DIR, name)
    if os.path.exists(path):
        return Image.open(path)
    return None


def show_model_comparison():
    st.header("📊 Model Comparison")
    st.markdown("Comparing **Logistic Regression**, **Random Forest**, and **XGBoost** on plant health classification.")

    # ── Load metrics ──────────────────────────────────────────────────────
    metrics = load_comparison_metrics()

    if not metrics:
        st.warning("⚠️ `reports/model_comparison_metrics.json` not found. Run `src/models/compare_models.py` first.")
        st.info("Showing demo data instead.")
        metrics = _demo_metrics()

    model_names = list(metrics.keys())
    colors      = {"Logistic Regression": "#e74c3c", "Random Forest": "#2ecc71", "XGBoost": "#3498db"}

    # ════════════════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Metrics Overview",
        "🎯 Confusion Matrices",
        "📉 ROC & AUC",
        "🕸️ Radar Chart",
        "📄 Full Report",
    ])

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1: Metrics Overview
    # ─────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Summary Table")

        rows = []
        for name in model_names:
            m = metrics[name]
            rows.append({
                "Model":          name,
                "Accuracy":       f"{m['accuracy']:.4f}",
                "Precision":      f"{m['precision']:.4f}",
                "Recall":         f"{m['recall']:.4f}",
                "F1-Score":       f"{m['f1_score']:.4f}",
                "AUC (Macro)":    f"{m['roc_auc_macro']:.4f}",
                "Inference (ms)": f"{m['inference_time_s']*1000:.1f}",
            })
        df_summary = pd.DataFrame(rows)
        st.dataframe(df_summary.set_index("Model"), use_container_width=True)

        # ── Accuracy Bar Chart ───────────────────────────────────────────
        st.subheader("Accuracy Comparison")
        fig_acc = go.Figure()
        for name in model_names:
            fig_acc.add_trace(go.Bar(
                name=name,
                x=[name],
                y=[metrics[name]["accuracy"]],
                marker_color=colors.get(name, "#888"),
                text=[f"{metrics[name]['accuracy']:.2%}"],
                textposition="outside",
            ))
        fig_acc.update_layout(
            yaxis=dict(range=[0, 1.15], title="Accuracy"),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, font=dict(size=13),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # ── Multi-metric grouped bar ─────────────────────────────────────
        st.subheader("Precision / Recall / F1 Comparison")
        metric_keys = ["precision", "recall", "f1_score"]
        metric_lbls = ["Precision", "Recall", "F1-Score"]

        fig_multi = go.Figure()
        for name in model_names:
            fig_multi.add_trace(go.Bar(
                name=name,
                x=metric_lbls,
                y=[metrics[name][k] for k in metric_keys],
                marker_color=colors.get(name, "#888"),
            ))
        fig_multi.update_layout(
            barmode="group", yaxis=dict(range=[0, 1.1]),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(size=13),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_multi, use_container_width=True)

        # ── Per-class F1 heatmap ─────────────────────────────────────────
        st.subheader("Per-Class F1-Score Heatmap")
        cls_labels  = [CLASS_NAMES[i] for i in range(3)]
        z_values    = [[metrics[name]["f1_per_class"][i] for i in range(3)] for name in model_names]

        fig_heat = go.Figure(go.Heatmap(
            z=z_values,
            x=cls_labels,
            y=model_names,
            colorscale="Greens",
            zmin=0, zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in z_values],
            texttemplate="%{text}",
            textfont=dict(size=14),
        ))
        fig_heat.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(size=13),
            xaxis_title="Class", yaxis_title="Model",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Inference Time ────────────────────────────────────────────────
        st.subheader("Inference Time (ms)")
        fig_inf = go.Figure()
        for name in model_names:
            t_ms = metrics[name]["inference_time_s"] * 1000
            fig_inf.add_trace(go.Bar(
                name=name, x=[name], y=[t_ms],
                marker_color=colors.get(name, "#888"),
                text=[f"{t_ms:.1f} ms"], textposition="outside",
            ))
        fig_inf.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, font=dict(size=13),
            yaxis_title="Milliseconds",
        )
        st.plotly_chart(fig_inf, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2: Confusion Matrices
    # ─────────────────────────────────────────────────────────────────────
    with tab2:
        # Try saved image first
        img = _img("confusion_matrices.png")
        if img:
            st.image(img, caption="Confusion Matrices (pre-computed)", use_container_width=True)
        else:
            st.subheader("Confusion Matrices")
            cls_labels = [CLASS_NAMES[i] for i in range(3)]
            cols = st.columns(len(model_names))
            for i, name in enumerate(model_names):
                with cols[i]:
                    cm = metrics[name].get("confusion_matrix", [])
                    if cm:
                        fig_cm = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=cls_labels, y=cls_labels,
                            color_continuous_scale="Greens",
                            text_auto=True,
                            title=name,
                        )
                        fig_cm.update_layout(font=dict(size=12))
                        st.plotly_chart(fig_cm, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3: ROC & AUC
    # ─────────────────────────────────────────────────────────────────────
    with tab3:
        img_roc = _img("roc_curves.png")
        img_auc = _img("auc_comparison.png")

        if img_roc:
            st.image(img_roc, caption="ROC Curves", use_container_width=True)
        if img_auc:
            st.image(img_auc, caption="AUC Comparison", use_container_width=True)

        # AUC table
        st.subheader("AUC Score Table")
        auc_rows = []
        for name in model_names:
            per_cls = metrics[name].get("roc_auc_per_class", {})
            auc_rows.append({
                "Model":           name,
                "AUC (Healthy)":   f"{per_cls.get('0', per_cls.get(0, 0)):.4f}",
                "AUC (Needs Water)":f"{per_cls.get('1', per_cls.get(1, 0)):.4f}",
                "AUC (Overwatered)":f"{per_cls.get('2', per_cls.get(2, 0)):.4f}",
                "AUC (Macro)":     f"{metrics[name]['roc_auc_macro']:.4f}",
            })
        st.dataframe(pd.DataFrame(auc_rows).set_index("Model"), use_container_width=True)

        # Interactive AUC bar
        fig_auc = go.Figure()
        for name in model_names:
            fig_auc.add_trace(go.Bar(
                name=name, x=[name],
                y=[metrics[name]["roc_auc_macro"]],
                marker_color=colors.get(name, "#888"),
                text=[f"{metrics[name]['roc_auc_macro']:.4f}"],
                textposition="outside",
            ))
        fig_auc.update_layout(
            title="Macro AUC Comparison",
            yaxis=dict(range=[0, 1.1]),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4: Radar Chart
    # ─────────────────────────────────────────────────────────────────────
    with tab4:
        img_radar = _img("radar_chart.png")
        if img_radar:
            st.image(img_radar, caption="Radar Chart (pre-computed)", use_container_width=True)

        st.subheader("Interactive Radar Chart")
        radar_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc_macro"]
        radar_labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

        fig_radar = go.Figure()
        for name in model_names:
            vals = [metrics[name][m] for m in radar_metrics]
            vals_closed = vals + [vals[0]]
            labels_closed = radar_labels + [radar_labels[0]]

            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=labels_closed,
                fill="toself",
                name=name,
                opacity=0.5,
                line=dict(color=colors.get(name, "#888")),
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            font=dict(size=13),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 5: Full Report
    # ─────────────────────────────────────────────────────────────────────
    with tab5:
        report_path = os.path.join(REPORTS_DIR, "model_comparison_report.md")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Fix any template placeholders still in file
            if "{results" in content:
                st.warning("Report contains unfilled template variables — showing raw report.")
            st.markdown(content)

            st.download_button(
                "⬇️ Download Report (.md)",
                data=content,
                file_name="model_comparison_report.md",
                mime="text/markdown",
            )
        else:
            st.warning("Report file not found. Run `src/models/compare_models.py` to generate it.")


# ── Demo metrics fallback ──────────────────────────────────────────────────
def _demo_metrics():
    return {
        "Logistic Regression": {
            "accuracy": 0.7125, "precision": 0.7149, "recall": 0.7125,
            "f1_score": 0.7136, "roc_auc_macro": 0.8816,
            "inference_time_s": 0.0052,
            "f1_per_class": [0.739, 0.788, 0.601],
            "confusion_matrix": [[44, 0, 16], [4, 78, 18], [11, 20, 49]],
            "roc_auc_per_class": {"0": 0.9508, "1": 0.9002, "2": 0.7938},
        },
        "Random Forest": {
            "accuracy": 1.000, "precision": 1.000, "recall": 1.000,
            "f1_score": 1.000, "roc_auc_macro": 1.000,
            "inference_time_s": 0.2251,
            "f1_per_class": [1.000, 1.000, 1.000],
            "confusion_matrix": [[60, 0, 0], [0, 100, 0], [0, 0, 80]],
            "roc_auc_per_class": {"0": 1.0, "1": 1.0, "2": 1.0},
        },
        "XGBoost": {
            "accuracy": 0.9958, "precision": 0.9959, "recall": 0.9958,
            "f1_score": 0.9958, "roc_auc_macro": 0.9999,
            "inference_time_s": 0.0108,
            "f1_per_class": [0.9917, 1.000, 0.9937],
            "confusion_matrix": [[60, 0, 0], [0, 100, 0], [1, 0, 79]],
            "roc_auc_per_class": {"0": 0.9999, "1": 1.0, "2": 0.9999},
        },
    }
