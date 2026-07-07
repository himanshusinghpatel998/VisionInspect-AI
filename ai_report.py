import os
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from google import genai

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)


def generate_report(prediction, confidence, score):

    prompt = f"""
You are an Industrial Quality Inspection AI.

Generate a professional report.

Product: Bottle

Prediction: {prediction}

Confidence: {confidence:.2f}%

Anomaly Score: {score:.4f}

Return:

1. Inspection Summary
2. Severity
3. Possible Cause
4. Recommendation
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text