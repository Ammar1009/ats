# src/preprocess.py
import re
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
_stopwords = set(stopwords.words("english"))

def preprocess_text(text):
    # Basic cleaning
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s\.\,\-]', ' ', text)
    text = text.lower()
    doc = nlp(text)
    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.strip()
        if lemma and lemma not in _stopwords and len(lemma) > 1:
            tokens.append(lemma)
    return " ".join(tokens)