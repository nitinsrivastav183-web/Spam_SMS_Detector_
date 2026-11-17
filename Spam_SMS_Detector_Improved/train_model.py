"""Train and evaluate the SMS spam classifier."""
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "spam.csv"
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    df = pd.read_csv(DATA_PATH)
    required = {"label", "message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].map(clean_text)
    df = df[df["label"].isin(["ham", "spam"]) & df["message"].ne("")]
    X, y = df["message"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB(alpha=0.05)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    print(f"Rows used: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels ham, spam):\n", confusion_matrix(y_test, predictions, labels=["ham", "spam"]))
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved {MODEL_PATH} and {VECTORIZER_PATH}")

if __name__ == "__main__":
    main()
