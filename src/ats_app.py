import streamlit as st
import fitz  # PyMuPDF
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")
def style_keyword_list(words, color_class):
    if not words:
        return "<p style='color:gray;'>None</p>"
    html = "".join(
        [f"<span class='chip {color_class}'>{w}</span>" for w in sorted(words)]
    )
    return html
# ---------- Helper Functions ----------

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF"""
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    return text

def calculate_ats_score(resume_text, job_description):
    """Hybrid similarity: TF-IDF + semantic"""
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform([resume_text, job_description])
    tfidf_score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    resume_doc = nlp(resume_text)
    jd_doc = nlp(job_description)
    semantic_score = resume_doc.similarity(jd_doc)

    final_score = (0.7 * semantic_score + 0.3 * tfidf_score) * 100
    return round(final_score, 2)

def keyword_analysis(resume_text, job_description):
    """Extract important nouns/skills"""
    resume_doc = nlp(resume_text.lower())
    jd_doc = nlp(job_description.lower())
    resume_nouns = {t.text for t in resume_doc if t.pos_ in ["NOUN", "PROPN"]}
    jd_nouns = {t.text for t in jd_doc if t.pos_ in ["NOUN", "PROPN"]}
    matched = resume_nouns & jd_nouns
    missing = jd_nouns - resume_nouns
    return list(matched), list(missing)

def style_keyword_list(words, color):
    """Display keywords as styled chips"""
    if not words:
        return "None"
    html = "".join(
        [f"<span style='background-color:{color}; color:white; padding:5px 10px; margin:4px; border-radius:20px; display:inline-block'>{w}</span>"
         for w in sorted(words)]
    )
    return html

# ---------- Page Configuration ----------
st.set_page_config(page_title="AI Resume ATS Analyzer", page_icon="🤖", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
/* ===== Elegant Futuristic Background ===== */
body {
    font-family: 'Poppins', sans-serif;
    color: #f0f0f0;
    background: radial-gradient(circle at 10% 20%, #1b2735 0%, #090a0f 100%);
}
.main {
    background: radial-gradient(circle at 25% 25%, #181a2e, #0d0f1b 60%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
}

/* ===== Header ===== */
h1 {
    font-size: 3em;
    text-align: center;
    font-weight: 700;
    background: linear-gradient(90deg, #9d4edd, #00b4d8, #4361ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3em;
}
h3 {
    text-align: center;
    color: #cfcfcf;
    margin-top: -0.8em;
    margin-bottom: 2em;
}

/* ===== Upload + TextArea ===== */
.stFileUploader, .stTextArea textarea {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 1em;
    color: #f1f1f1;
}

/* ===== Button ===== */
.stButton>button {
    background: linear-gradient(90deg, #4361ee, #7209b7);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    height: 3em;
    width: 16em;
    margin: 1em auto;
    display: block;
    box-shadow: 0 0 20px rgba(100, 100, 255, 0.4);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    box-shadow: 0 0 35px rgba(144, 55, 255, 0.6);
    transform: scale(1.05);
}

/* ===== Score Card ===== */
.score-box {
    background: linear-gradient(145deg, #111327, #1a1c33);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    text-align: center;
    padding: 2em;
    margin-top: 1.5em;
    box-shadow: 0 0 25px rgba(67, 97, 238, 0.25);
}

/* ===== Keyword Chips ===== */
.chip {
    display: inline-block;
    border-radius: 25px;
    padding: 6px 14px;
    margin: 6px;
    font-size: 0.9em;
    color: white;
}
.green-chip {
    background: linear-gradient(90deg, #06d6a0, #00b4d8);
}
.red-chip {
    background: linear-gradient(90deg, #ef233c, #d90429);
}

/* ===== Footer ===== */
hr {
    margin-top: 3em;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #7209b7, transparent);
}
footer {
    text-align: center;
    font-size: 0.9em;
    color: #aaa;
    margin-top: 1em;
}
</style>
""", unsafe_allow_html=True)
# ---------- App Title ----------
st.title("🤖 AI Resume ATS Analyzer")
st.markdown("### Evaluate how well your resume matches a job description — ATS optimized & skill-based insights")

# ---------- Upload and Input ----------
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
with col2:
    job_description = st.text_area("🧾 Paste Job Description Here")

# ---------- Analysis ----------
if st.button("Analyze ATS Score"):
    if uploaded_file and job_description.strip():
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            score = calculate_ats_score(resume_text, job_description)
            matched, missing = keyword_analysis(resume_text, job_description)

        # Score Section
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown(f"<h2>ATS Match Score: {score}%</h2>", unsafe_allow_html=True)
        if score >= 80:
            st.success("🔥 Excellent match! Your resume is well-aligned.")
        elif score >= 50:
            st.warning("🙂 Decent match. Add more relevant keywords.")
        else:
            st.error("⚠️ Low match. Update your resume with missing terms.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Keyword Section
        st.subheader("🟢 Matched Keywords")
        st.markdown(style_keyword_list(matched, "green-chip"), unsafe_allow_html=True)

        st.subheader("🔴 Missing Keywords")
        st.markdown(style_keyword_list(missing, "red-chip"), unsafe_allow_html=True)

        # Download Report
        st.download_button(
            label="📥 Download ATS Report",
            data=f"ATS Score: {score}%\n\nMatched Keywords: {', '.join(matched)}\n\nMissing Keywords: {', '.join(missing)}",
            file_name="ATS_Report.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please upload a resume and provide a job description.")