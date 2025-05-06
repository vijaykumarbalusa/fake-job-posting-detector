import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

# Load model and vectorizer
model = joblib.load("job_fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Red flag feature extraction function
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

# Page setup
st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️‍♂️")
st.title("🕵️ Fake Job Posting Detector")
st.markdown("Paste any job description below and we'll use AI to detect if it's **real or fake**.")

# Session state to handle reset
if "submitted" not in st.session_state:
    st.session_state.submitted = False

def submit():
    st.session_state.submitted = True

# User input
job_text = st.text_area("✍️ Paste job posting here:", height=300)

# Submit button
st.button("🔍 Detect", on_click=submit)

# Detection logic
if st.session_state.submitted:
    if job_text.strip() == "":
        st.warning("Please paste a job description to proceed.")
    else:
        # Feature extraction
        tfidf_features = vectorizer.transform([job_text])
        red_flag_features = np.array(extract_red_flags(job_text)).reshape(1, -1)
        final_input = hstack([tfidf_features, csr_matrix(red_flag_features)])

        # Prediction
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][prediction]

        # Result display
        if prediction == 1:
            st.error("⚠️ This looks like a **FAKE** job posting.")
        else:
            st.success("✅ This looks like a **GENUINE** job posting.")

        st.info(f"🧠 Model Confidence: **{round(probability * 100, 2)}%**")

        # Red flag breakdown
        flag_names = [
            "Mentions WhatsApp",
            "Uses free email domain (gmail/yahoo/etc)",
            "Mentions money or salary",
            "Requests registration fee",
            "Urgent hiring language",
            "Mentions well-known brands",
            "Asks for personal/banking info"
        ]
        flags = extract_red_flags(job_text)
        suspicious_flags = [name for name, val in zip(flag_names, flags) if val]

        if suspicious_flags:
            st.warning("🚩 Red Flags Detected:")
            for flag in suspicious_flags:
                st.markdown(f"- {flag}")
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

# Load model and vectorizer
model = joblib.load("job_fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Red flag feature extraction function
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

# Page setup
st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️‍♂️")
st.title("🕵️ Fake Job Posting Detector")
st.markdown("Paste any job description below and we'll use AI to detect if it's **real or fake**.")

# Session state to handle reset
if "submitted" not in st.session_state:
    st.session_state.submitted = False

def submit():
    st.session_state.submitted = True

# User input
job_text = st.text_area("✍️ Paste job posting here:", height=300)

# Submit button
st.button("🔍 Detect", on_click=submit)

# Detection logic
if st.session_state.submitted:
    if job_text.strip() == "":
        st.warning("Please paste a job description to proceed.")
    else:
        # Feature extraction
        tfidf_features = vectorizer.transform([job_text])
        red_flag_features = np.array(extract_red_flags(job_text)).reshape(1, -1)
        final_input = hstack([tfidf_features, csr_matrix(red_flag_features)])

        # Prediction
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][prediction]

        # Result display
        if prediction == 1:
            st.error("⚠️ This looks like a **FAKE** job posting.")
        else:
            st.success("✅ This looks like a **GENUINE** job posting.")

        st.info(f"🧠 Model Confidence: **{round(probability * 100, 2)}%**")

        # Red flag breakdown
        flag_names = [
            "Mentions WhatsApp",
            "Uses free email domain (gmail/yahoo/etc)",
            "Mentions money or salary",
            "Requests registration fee",
            "Urgent hiring language",
            "Mentions well-known brands",
            "Asks for personal/banking info"
        ]
        flags = extract_red_flags(job_text)
        suspicious_flags = [name for name, val in zip(flag_names, flags) if val]

        if suspicious_flags:
            st.warning("🚩 Red Flags Detected:")
            for flag in suspicious_flags:
                st.markdown(f"- {flag}")
