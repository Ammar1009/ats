# src/model_utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def build_vectorizer(max_features=3000):
    return TfidfVectorizer(max_features=max_features, ngram_range=(1,2))

def save_model(model, vectorizer, model_path="models/resume_clf.joblib", vec_path="models/vectorizer.joblib"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

def load_model(model_path="models/resume_clf.joblib", vec_path="models/vectorizer.joblib"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer