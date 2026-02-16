import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def show_visualization():
    st.header("📈 Data Visualization")
    
    # Sample data (as per your CSV files)
    data = pd.DataFrame({
        'Plant_Health_Status': np.random.choice(['Healthy', 'Diseased', 'At Risk'], 100),
        'Temperature': np.random.normal(25, 5, 100),
        'Humidity': np.random.normal(60, 10, 100),
        'Soil_Moisture': np.random.normal(40, 8, 100)
    })
    
    # Filter
    status = st.multiselect(
        "Filter by Health Status",
        data['Plant_Health_Status'].unique(),
        default=data['Plant_Health_Status'].unique()
    )
    
    filtered = data[data['Plant_Health_Status'].isin(status)]
    st.write(f"Showing {len(filtered)} records")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Health Distribution")
        fig1 = px.pie(filtered, names='Plant_Health_Status')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Temperature Distribution")
        fig2 = px.histogram(filtered, x='Temperature', color='Plant_Health_Status')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plot
    st.subheader("Feature Relationship")
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("X-axis", ['Temperature', 'Humidity', 'Soil_Moisture'])
    
    with col2:
        y_axis = st.selectbox("Y-axis", ['Temperature', 'Humidity', 'Soil_Moisture'])
    
    fig3 = px.scatter(filtered, x=x_axis, y=y_axis, color='Plant_Health_Status')
    st.plotly_chart(fig3, use_container_width=True)
    
    # Data table
    with st.expander("View Raw Data"):
        st.dataframe(filtered)