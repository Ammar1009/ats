# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from model_utils import build_vectorizer, save_model
from preprocess import preprocess_text
import os
import argparse

def train_model(csv_path):
    df = pd.read_csv(csv_path)  # expects columns: resume_text, label
    df['clean_text'] = df['resume_text'].astype(str).apply(preprocess_text)

    X = df['clean_text']
    y = df['label']

    vec = build_vectorizer()
    X_vec = vec.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    save_model(clf, vec)
    print("Saved model and vectorizer to /models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/resumes.csv", help="CSV with resume_text,label")
    args = parser.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"Dataset not found: {args.csv}")
    train_model(args.csv)