"""Dataset Visualization page view for exploring the plant health dataset."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.utils.config import CLEANED_CSV, X_TRAIN_CSV, Y_TRAIN_CSV, TARGET_COL, CLASS_NAMES


@st.cache_data
def load_data():
    """Load the dataset from cleaned CSV or train/test splits."""
    if os.path.exists(CLEANED_CSV):
        return pd.read_csv(CLEANED_CSV)
    if os.path.exists(X_TRAIN_CSV) and os.path.exists(Y_TRAIN_CSV):
        X = pd.read_csv(X_TRAIN_CSV)
        y = pd.read_csv(Y_TRAIN_CSV)
        return pd.concat([X, y], axis=1)
    return None


def show_dataset_visualization():
    """Display the dataset visualization page with distributions and correlations."""
    st.header("Dataset Visualization")

    df = load_data()

    if df is None:
        st.warning("Dataset not found. Please ensure `data/processed/cleaned_data.csv` exists.")
        return

    st.success(f"Dataset loaded: **{len(df):,} rows x {len(df.columns)} columns**")

    # Basic info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Features", f"{len(df.columns)-1}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Raw Data"])

    with tab1:
        # Target distribution
        target_col = TARGET_COL if TARGET_COL in df.columns else df.columns[-1]
        if target_col in df.columns:
            st.subheader("Target Class Distribution")
            vc = df[target_col].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            vc["Label"] = vc["Class"].map(CLASS_NAMES)
            fig_pie = px.pie(
                vc, names="Label", values="Count",
                color_discrete_sequence=["#2ecc71", "#e67e22", "#e74c3c"],
                title="Class Distribution",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Feature distributions
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        selected_feat = st.selectbox("Select Feature", numeric_cols[:15])
        if target_col in df.columns:
            fig_hist = px.histogram(
                df, x=selected_feat, color=target_col,
                color_discrete_map={0: "#2ecc71", 1: "#e67e22", 2: "#e74c3c"},
                nbins=40, barmode="overlay",
                title=f"Distribution of {selected_feat} by Class",
            )
        else:
            fig_hist = px.histogram(df, x=selected_feat, nbins=40)
        fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Feature Correlation Heatmap")
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()

        fig_corr = px.imshow(
            corr.round(2),
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            text_auto=True,
            title="Correlation Matrix",
        )
        fig_corr.update_layout(height=600, font=dict(size=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="plant_data.csv",
            mime="text/csv",
        )
