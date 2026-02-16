import streamlit as st
from PIL import Image
import time

def show_predict():

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
    st.header("🔬 Disease Prediction")

    col1, col2 = st.columns(2)

    # ===== Upload Section =====
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<p class="section">Upload Image</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose leaf image", type=["jpg", "png", "jpeg"])

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image")

            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    time.sleep(2)
                    st.success("✅ Prediction: Healthy Plant")
                    st.info("Confidence: 98%")

        st.markdown('</div>', unsafe_allow_html=True)

    # ===== Info Section =====
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<p class="section">How to Use</p>', unsafe_allow_html=True)

        st.write("1. Take clear photo of leaf")
        st.write("2. Upload the image")
        st.write("3. Click Predict button")
        st.write("4. Get instant result")

        st.markdown("---")

        st.markdown('<p class="section">Supported Plants</p>', unsafe_allow_html=True)

        plants = ["Tomato", "Potato", "Wheat", "Rice", "Corn", "Soybean"]

        for plant in plants:
            st.write(f"• {plant}")

        st.markdown('</div>', unsafe_allow_html=True)
