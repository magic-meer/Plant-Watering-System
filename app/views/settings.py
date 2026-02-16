import streamlit as st

def show_settings():

    # ===== Styling =====
    st.markdown("""
    <style>

    body, .stApp, p, h1, h2, h3, h4, span, label {
        color: black !important;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        margin-bottom: 20px;
    }

    .section {
        color: #2ecc71;
        font-weight: bold;
        margin-top: 10px;
    }

    .stButton button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        width: 200px;
        font-weight: bold;
    }

    </style>
    """, unsafe_allow_html=True)

    # ===== Header =====
    st.header("⚙️ Settings")

    tab1, tab2 = st.tabs(["General", "Notifications"])

    # ===== Tab 1 =====
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<p class="section">Appearance</p>', unsafe_allow_html=True)
        theme = st.selectbox("Theme", ["Light", "Dark"])
        language = st.selectbox("Language", ["English", "Spanish"])

        st.markdown('<p class="section">Data</p>', unsafe_allow_html=True)
        auto = st.checkbox("Auto-save data", value=True)
        backup = st.checkbox("Auto backup", value=True)
        days = st.number_input("Data retention (days)", 30, 365, 90)

        st.markdown('</div>', unsafe_allow_html=True)

    # ===== Tab 2 =====
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<p class="section">Alerts</p>', unsafe_allow_html=True)
        email = st.checkbox("Email alerts", value=True)
        push = st.checkbox("Push alerts", value=False)
        threshold = st.slider("Alert threshold (%)", 0, 100, 70)

        st.markdown('</div>', unsafe_allow_html=True)

    # ===== Save Button =====
    if st.button("Save Settings"):
        st.success("Settings saved!")
