import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


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