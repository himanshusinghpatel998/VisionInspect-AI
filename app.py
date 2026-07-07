import os
import tempfile

import streamlit as st
import torch
from PIL import Image

from feature_extractor import FeatureExtractor
from patchcore import PatchCore
from heatmap import create_heatmap
from ai_report import generate_report
from pdf_report import create_pdf

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Industrial Surface Defect Detection",
    page_icon="🏭",
    layout="wide"
)

# -------------------------------------------------------
# Device
# -------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.title("🏭 Inspection Dashboard")

    st.markdown("---")

    st.subheader("Model Information")

    st.write("**Algorithm:** PatchCore")
    st.write("**Backbone:** ResNet50")
    st.write(f"**Device:** {device.upper()}")
    st.write("**Heatmap:** Enabled")
    st.write("**Generative AI:** Google Gemini")
    st.write("**PDF Report:** Enabled")

    st.markdown("---")

    st.success("System Ready ✅")

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🏭 AI-Powered Industrial Surface Defect Detection")

st.caption(
    "Real-Time Industrial Quality Inspection using PatchCore, ResNet50 and Google Gemini"
)

# -------------------------------------------------------
# Session State
# -------------------------------------------------------

if "report" not in st.session_state:
    st.session_state.report = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

extractor = FeatureExtractor(device)

patchcore = PatchCore()

# -------------------------------------------------------
# Upload Image
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Product Image",
    type=["png", "jpg", "jpeg"]
)

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if uploaded_file:

    if uploaded_file.name != st.session_state.last_file:

        st.session_state.report = None
        st.session_state.pdf_path = None
        st.session_state.last_file = uploaded_file.name

    image = Image.open(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:

        image.save(temp.name)

        feature_map = extractor.extract(temp.name)

        score, confidence, defect = patchcore.predict(feature_map)

        os.makedirs("outputs/heatmaps", exist_ok=True)

        heatmap_path = "outputs/heatmaps/result.png"

        create_heatmap(
            temp.name,
            feature_map,
            heatmap_path
        )

    # -------------------------------------------------------
    # Images
    # -------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔥 Heatmap")
        st.image(heatmap_path, use_container_width=True)

    st.divider()

    prediction = "Defective" if defect else "Normal"

    col1, col2, col3 = st.columns(3)

    with col1:

        if prediction == "Normal":
            st.success("🟢 Normal Product")
        else:
            st.error("🔴 Defective Product")

    with col2:
        st.metric("Anomaly Score", f"{score:.2f}")

    with col3:
        st.metric("Confidence", f"{confidence:.2f}%")

    st.divider()

    # -------------------------------------------------------
    # AI Report
    # -------------------------------------------------------

    st.subheader("🤖 AI Inspection Report")

    if st.button("Generate AI Report"):

        with st.spinner("Generating AI Report..."):

            try:

                st.session_state.report = generate_report(
                    prediction,
                    confidence,
                    score
                )

            except Exception:

                st.session_state.report = f"""
## AI Inspection Report

**Prediction:** {prediction}

**Confidence:** {confidence:.2f}%

**Anomaly Score:** {score:.2f}

---

⚠️ Gemini AI Report is temporarily unavailable.

The system has reached the free API quota.

### Recommendation

• Review the product manually.

• Verify the anomaly using the generated heatmap.

• Retry report generation after the quota resets.
"""

            pdf_path = create_pdf(
                prediction=prediction,
                confidence=confidence,
                score=score,
                ai_report=st.session_state.report,
                original_image=temp.name,
                heatmap_image=heatmap_path
            )

            st.session_state.pdf_path = pdf_path

    # -------------------------------------------------------
    # Display AI Report
    # -------------------------------------------------------

    if st.session_state.report:

        st.markdown(st.session_state.report)

    # -------------------------------------------------------
    # Download PDF
    # -------------------------------------------------------

    if st.session_state.pdf_path:

        with open(st.session_state.pdf_path, "rb") as pdf:

            st.download_button(
                label="📄 Download Inspection Report",
                data=pdf,
                file_name="Inspection_Report.pdf",
                mime="application/pdf"
            )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.markdown("---")

st.caption(
    "Developed by Himanshu Singh Patel | "
    "AI-Powered Industrial Surface Defect Detection | "
    "PatchCore • PyTorch • Streamlit • Google Gemini"
)