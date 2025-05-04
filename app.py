import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk

# Load the trained model and vectorizer
model = pickle.load(open("job_fraud_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("🕵️ Fake Job Posting Detector")

input_text = st.text_area("Paste a job description here:")

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some job posting text.")
    else:
        cleaned = clean_text(input_text)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        if prediction == 1:
            st.error("⚠️ This looks like a FAKE job posting!")
        else:
            st.success("✅ This looks like a REAL job posting.")