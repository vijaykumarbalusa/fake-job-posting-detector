# 🕵️ Fake Job Posting Detector

A Machine Learning-powered web app that detects **fake job postings** based on text patterns and scam signals. Built using **XGBoost, NLP, and Streamlit**, it helps users identify suspicious job offers before falling victim to scams.

---

## 🚀 Demo

![App Screenshot](screenshot.png)  
🌐 Try it live: [Streamlit App Link](https://your-streamlit-link-here)

---

## 🧠 Features

- ✅ Classifies job descriptions as **Fake** or **Genuine**
- 🔍 Detects scam signals like:
  - WhatsApp-only contact
  - Gmail/Yahoo email domains
  - Registration fee requests
  - Urgent hiring phrases
  - Brand impersonation (Amazon, Netflix, etc.)
- 📊 Displays **model confidence score**
- 🚩 Shows detected red flags to explain the decision

---

## 📂 Tech Stack

- **Machine Learning**: XGBoost, TF-IDF, Scikit-learn
- **NLP**: Text vectorization + handcrafted features
- **UI**: Python + Streamlit
- **Deployment**: Streamlit Cloud
- **Data**: Curated fake job postings + real-world samples

---

## 📦 How to Use

```bash
git clone https://github.com/your-username/fake-job-posting-detector.git
cd fake-job-posting-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Files

- `app.py` — Streamlit frontend app
- `job_fraud_model.pkl` — Trained XGBoost model
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer
- `requirements.txt` — Python dependencies

---

## 📌 Project Goals

- Help job seekers **avoid scams**
- Learn and apply ML + deployment workflow
- Build a real-world, resume-worthy app

---

## 🙋‍♂️ Author

[Vijay Kumar Balusa](https://www.linkedin.com/in/your-profile)

---

## ⭐ Show Your Support

If you found this useful, please consider giving it a ⭐ on GitHub!
