import streamlit as st

def show_about():

    # ===== Styling =====
    st.markdown("""
    <style>

    body, .stApp, p, h1, h2, h3, h4, span {
        color: black !important;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .section-title {
        color: #2ecc71;
        font-weight: bold;
    }

    </style>
    """, unsafe_allow_html=True)

    # ===== Header =====
    st.header(" About Plant Health AI")
    st.write("Smart AI-based system for monitoring plant health and detecting diseases.")

    st.markdown("---")

    # ===== Layout =====
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)

        st.markdown('<p class="section-title">Plant Health AI</p>', unsafe_allow_html=True)
        st.write("Version: 2.0")
        st.write("Released: 2026")

        st.markdown('<p class="section-title">Features</p>', unsafe_allow_html=True)
        st.write("• Dashboard monitoring")
        st.write("• Disease prediction")
        st.write("• Data visualization")
        st.write("• Custom settings")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)

        st.markdown('<p class="section-title">Contact</p>', unsafe_allow_html=True)
        st.write(" Email: info@planthealth.com")
        st.write(" Website: www.planthealth.com")

        st.markdown('<p class="section-title">Tech Stack</p>', unsafe_allow_html=True)
        st.write("• Streamlit")
        st.write("• Python")
        st.write("• Plotly")
        st.write("• Pandas")

        st.markdown('</div>', unsafe_allow_html=True)
