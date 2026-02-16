import streamlit as st

def show_dashboard():

    # ===== Clean UI Styling =====
    st.markdown("""
    <style>

    /* Make ALL text black */
    body, .stApp, p, h1, h2, h3, h4, h5, h6, span, label {
        color: black !important;
    }

    /* Metric Card */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        text-align: center;
    }

    /* Action Buttons */
    .stButton button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
    }

    /* Status Boxes */
    .status-box {
        background: #f4fff6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        color: black;
    }

    </style>
    """, unsafe_allow_html=True)

    # ===== Header =====
    st.header(" Dashboard")
    st.write("Welcome to Plant Health System")
    st.write("Your plant monitoring system is active")

    st.markdown("---")

    # ===== Metrics =====
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Plants Monitored", "120")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Diseases Detected", "15")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Healthy Plants", "105")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ===== Quick Actions =====
    st.subheader("⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(" Scan"):
            st.success("Scanning started")

    with col2:
        if st.button(" Report"):
            st.info("Generating report")

    with col3:
        if st.button(" Update"):
            st.warning("Checking updates")

    with col4:
        if st.button(" Alert"):
            st.success("Alert sent")

    st.markdown("---")

    # ===== System Status =====
    st.subheader(" System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.write("• Sensors: 12/12 Active")
        st.write("• Last Update: 2 min ago")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.write("• Storage: 45% Used")
        st.write("• System Health: Good")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ===== Recent Activity =====
    st.subheader(" Recent Activity")

    st.markdown("""
    <div class="status-box">
    • New plant added - Tomato <br>
    • Disease detected - Potato <br>
    • System updated successfully
    </div>
    """, unsafe_allow_html=True)
