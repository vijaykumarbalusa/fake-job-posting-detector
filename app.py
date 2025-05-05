import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and vectorizer
model = joblib.load("job_fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Red flag feature extraction
def extract_red_flags(text):
    text = text.lower()
    return [
        "whatsapp" in text,
        any(x in text for x in ["gmail.com", "yahoo.com", "hotmail.com", "fastmail.com"]),
        any(sym in text for sym in ["$", "₹"]),
        any(x in text for x in ["registration fee", "paytm", "paypal", "deposit"]),
        any(x in text for x in ["urgent", "immediate", "start today", "limited", "immediate start"]),
        any(x in text for x in ["apple", "amazon", "google", "netflix", "airbnb", "usps", "microsoft"]),
        any(x in text for x in ["send your", "aadhar", "pan card", "photo id", "bank", "phone number"]),
    ]

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️‍♂️")
st.title("🕵️ Fake Job Posting Detector")
st.markdown("Paste any job description below and we'll detect if it's **real or fake** using AI and NLP.")

# Input
job_text = st.text_area("✍️ Paste job description here:", height=300)

if st.button("🔍 Detect"):
    if job_text.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Vectorize text
        tfidf_features = vectorizer.transform([job_text])
        red_flag_features = np.array(extract_red_flags(job_text)).reshape(1, -1)

        # Combine features
        from scipy.sparse import hstack
        final_input = hstack([tfidf_features, red_flag_features])

        # Predict
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][prediction]

        # Display result
        if prediction == 1:
            st.error(f"⚠️ This looks like a **FAKE** job posting.")
        else:
            st.success(f"✅ This looks like a **Genuine** job posting.")

        st.info(f"🧠 Model Confidence: **{round(probability * 100, 2)}%**")

        # Show red flags
        flags = extract_red_flags(job_text)
        flag_names = [
            "Mentions WhatsApp",
            "Free email domain (gmail/yahoo)",
            "Mentions money/salary",
            "Asks for registration fee",
            "Urgent hiring tone",
            "Mentions well-known brands",
            "Requests personal/banking info"
        ]
        suspicious = [f for f, v in zip(flag_names, flags) if v]
        if suspicious:
            st.warning("🚩 Potential Red Flags Detected:")
            for flag in suspicious:
                st.markdown(f"- {flag}")
