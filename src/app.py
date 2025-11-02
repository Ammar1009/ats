# src/app.py
import streamlit as st
from extract_text import extract_text_from_pdf
from preprocess import preprocess_text
from model_utils import load_model
import os

st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🤖 AI Resume Screener")

MODEL_PATH = "models/resume_clf.joblib"
VEC_PATH = "models/vectorizer.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
    st.error("Model or vectorizer not found. Train model first: python src/train_model.py")
else:
    model, vectorizer = load_model(MODEL_PATH, VEC_PATH)

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

keywords_input = st.text_input("Shortlisting keywords (comma separated)", "python, machine learning, sql")

if uploaded_file is not None:
    try:
        raw_text = extract_text_from_pdf(uploaded_file)
    except Exception as e:
        st.error("Failed to read PDF: " + str(e))
        raw_text = ""

    st.subheader("Extracted text (preview)")
    st.write(raw_text[:1000] + ("..." if len(raw_text) > 1000 else ""))

    cleaned = preprocess_text(raw_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Category: **{prediction}**")

    # Keyword scoring
    keywords = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]
    score = sum(1 for kw in keywords if kw in raw_text.lower())
    st.info(f"Keyword match: **{score}/{len(keywords)}**")

    threshold = st.slider("Shortlist threshold (match count ≥)", 0, len(keywords), max(1, len(keywords)//2))
    if score >= threshold:
        st.balloons()
        st.success("Candidate shortlisted ✅")
    else:
        st.warning("Candidate not shortlisted")