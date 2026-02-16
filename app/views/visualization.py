import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def show_visualization():

    # ===== UI Styling =====
    st.markdown("""
    <style>

    body, .stApp, p, h1, h2, h3, h4, h5, h6, span, label {
        color: black !important;
    }

    .filter-box {
        background: #f4fff6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
    }

    .chart-card {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)

    # ===== Header =====
    st.header(" Visualization")
    st.write("Analyze plant health data with interactive charts")

    st.markdown("---")

    # ===== Sample Data =====
    np.random.seed(42)
    data = pd.DataFrame({
        'Status': np.random.choice(['Healthy', 'Diseased', 'At Risk'], 100),
        'Temperature': np.random.normal(25, 5, 100),
        'Humidity': np.random.normal(60, 10, 100),
        'Moisture': np.random.normal(40, 8, 100)
    })

    # ===== Filter Section =====
    st.markdown('<div class="filter-box">', unsafe_allow_html=True)

    status = st.multiselect(
        "Filter by Status",
        data['Status'].unique(),
        default=data['Status'].unique()
    )

    st.markdown('</div>', unsafe_allow_html=True)

    filtered = data[data['Status'].isin(status)]

    st.write(f"Showing {len(filtered)} records")

    st.markdown("---")

    # ===== Charts =====
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.subheader("Health Distribution")
        fig1 = px.pie(filtered, names='Status', color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12"])
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.subheader("Temperature Distribution")
        fig2 = px.histogram(filtered, x='Temperature', color='Status',
                            color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12"])
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ===== Scatter Plot =====
    st.subheader(" Feature Relationship")

    col1, col2 = st.columns(2)

    with col1:
        x = st.selectbox("Select X-axis", ['Temperature', 'Humidity', 'Moisture'])

    with col2:
        y = st.selectbox("Select Y-axis", ['Temperature', 'Humidity', 'Moisture'])

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    fig3 = px.scatter(filtered, x=x, y=y, color='Status',
                      color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12"])

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ===== Data Table =====
    with st.expander(" View Raw Data"):
        st.dataframe(filtered, use_container_width=True)
