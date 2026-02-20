"""
app/app.py  ─  Main Streamlit Entry Point
Plant Watering System - Horizontal Top Navigation
Run with: streamlit run app/app.py
"""

import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Plant Health Intelligence System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Hide sidebar & default elements */
    [data-testid="stSidebar"]        { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
    .block-container { padding-top: 1rem !important; }

    /* Page background */
    .stApp { background-color: #f0f7f0; }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 0.5rem 1rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a1a;
        margin: 0.3rem 0;
    }
    .main-header p {
        color: #666;
        font-size: 1rem;
        margin: 0;
    }

    /* Nav bar */
    .stButton button {
        background-color: #111111 !important;
        color: white !important;
        border: none !important;
        border-radius: 0px !important;
        width: 100% !important;
        padding: 0.75rem !important;
        font-size: 0.95rem !important;
        transition: background 0.2s !important;
    }
    .stButton button:hover {
        background-color: #2e7d32 !important;
        color: white !important;
    }

    /* Footer */
    .custom-footer {
        background: #1b5e20;
        color: white;
        text-align: center;
        padding: 0.8rem;
        border-radius: 8px;
        margin-top: 3rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div style="font-size:2.8rem;">🌿</div>
    <h1>Plant Health Intelligence System</h1>
    <p>AI Powered Plant Monitoring & Watering Decision Engine</p>
</div>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ── Horizontal Nav ───────────────────────────────────────────────────────────
nav_items = [
    ("🏠", "Dashboard"),
    ("📊", "Visualization"),
    ("🔬", "Predict"),
    ("📈", "Model Comparison"),
    ("⚙️", "Settings"),
    ("ℹ️", "About"),
]

st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
cols = st.columns(len(nav_items))
for i, (icon, label) in enumerate(nav_items):
    with cols[i]:
        is_active = st.session_state.page == label
        # Highlight active button
        if is_active:
            st.markdown(f"""
            <style>
            div[data-testid="column"]:nth-child({i+1}) .stButton button {{
                background-color: #2e7d32 !important;
                font-weight: bold !important;
            }}
            </style>
            """, unsafe_allow_html=True)
        if st.button(f"{icon} {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin:0.5rem 0 1.5rem 0; border-color:#ccc;'>", unsafe_allow_html=True)

# ── Page Routing ─────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "Dashboard":
    from views.dashboard import show_dashboard
    show_dashboard()

elif page == "Predict":
    from views.predict import show_predict
    show_predict()

elif page == "Model Comparison":
    from views.model_comparison import show_model_comparison
    show_model_comparison()

elif page == "Visualization":
    from views.dataset_visualization import show_dataset_visualization
    show_dataset_visualization()

elif page == "Settings":
    from views.settings import show_settings
    show_settings()

elif page == "About":
    from views.about import show_about
    show_about()

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    🌿 Plant Health Intelligence System © 2026 | Streamlit + scikit-learn + XGBoost
</div>
""", unsafe_allow_html=True)