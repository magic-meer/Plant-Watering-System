import streamlit as st
from streamlit_option_menu import option_menu

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Plant Health AI",
    page_icon="🌿",
    layout="wide"
)

# ================= GLOBAL LIGHT GREEN THEME =================
st.markdown("""
<style>

/* Hide Sidebar */
[data-testid="stSidebar"] {display:none;}

/* Remove side padding that causes black gap */
.block-container {
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Main Background */
.stApp {
    background: linear-gradient(to right, #e8fbe8, #f4fff4);
}

/* Make all text black */
body, .stApp, p, h1, h2, h3, h4, span, label {
    color: black !important;
}

/* ===== FIX NAVBAR GAP ===== */
div[data-testid="stHorizontalBlock"] {
    background: linear-gradient(to right, #2ecc71, #27ae60);
    padding: 12px;
    border-radius: 12px;
}

/* Hover effect */
.nav-link:hover {
    background-color: #1e8449 !important;
}

/* Buttons */
.stButton button {
    background-color: #2ecc71;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<h1 style='text-align:center; color:#2ecc71;'>
🌿 Plant Health Intelligence System
</h1>
<p style='text-align:center; color:black;'>
AI Powered Plant Monitoring & Disease Detection
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= TOP NAVBAR =================
selected = option_menu(
    None,
    ["Dashboard", "Visualization", "Predict", "Settings", "About"],
    icons=["speedometer", "bar-chart", "activity", "gear", "info-circle"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"background-color": "transparent"},
        "icon": {"color": "white"},
        "nav-link": {"color": "white", "font-size": "16px", "margin": "8px"},
        "nav-link-selected": {"background-color": "#1e8449"},
    },
)

st.markdown("<br>", unsafe_allow_html=True)

# ================= ROUTING =================
if selected == "Dashboard":
    from views.dashboard import show_dashboard
    show_dashboard()

elif selected == "Visualization":
    from views.visualization import show_visualization
    show_visualization()

elif selected == "Predict":
    from views.predict import show_predict
    show_predict()

elif selected == "Settings":
    from views.settings import show_settings
    show_settings()

elif selected == "About":
    from views.about import show_about
    show_about()

# ================= FOOTER =================
st.markdown("""
<hr>
<center style="color:black">
🌿 Plant Health AI © 2026 | Made with ❤️ using Streamlit
</center>
""", unsafe_allow_html=True)
