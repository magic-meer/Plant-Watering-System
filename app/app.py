import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Plant Health AI",
    page_icon="🌱",
    layout="wide"
)

# Very simple CSS - just green and white
st.markdown("""
<style>
    /* Simple green header */
    .green-header {
        background-color: #2e7d32;
        padding: 1rem;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    /* Simple navigation */
    .nav-menu {
        background-color: white;
        padding: 0.5rem;
        border: 2px solid #2e7d32;
        border-radius: 5px;
        margin: 1rem auto;
        width: fit-content;
    }
    
    /* Simple green text */
    .green-text {
        color: #2e7d32;
        font-weight: bold;
    }
    
    /* Simple divider */
    hr {
        border: 1px solid #2e7d32;
        margin: 2rem 0;
    }
    
    /* Simple footer */
    .footer {
        background-color: #2e7d32;
        color: white;
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    /* Fix for the warning */
    .stImage {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Simple header
st.markdown('<div class="green-header">🌱 PLANT HEALTH INTELLIGENCE SYSTEM</div>', unsafe_allow_html=True)

# Top navigation bar
with st.container():
    st.markdown('<div class="nav-menu">', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Visualization", "Predict", "Settings", "About"],
        icons=["house", "bar-chart", "camera", "gear", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0", "background-color": "white"},
            "icon": {"color": "#2e7d32", "font-size": "1rem"},
            "nav-link": {
                "font-size": "1rem",
                "text-align": "center",
                "margin": "0 2px",
                "padding": "0.5rem 1rem",
                "color": "black",
            },
            "nav-link-selected": {
                "background-color": "#2e7d32",
                "color": "white",
            }
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Page routing
if selected == "Dashboard":
    from views.dashboard import show_dashboard
    show_dashboard()
elif selected == "Visualization":
    from views.visualization import show_visualization
    show_visualization()
elif selected == "Predict":
    from views.predict import show_preview
    show_preview()
elif selected == "Settings":
    from views.settings import show_settings
    show_settings()
elif selected == "About":
    from views.about import show_about
    show_about()

# Simple footer
st.markdown('<div class="footer">Plant Health Intelligence System © 2026 | Built with Streamlit</div>', unsafe_allow_html=True)