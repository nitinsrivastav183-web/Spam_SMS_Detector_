import re
from pathlib import Path
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "spam_model.pkl"
VECTORIZER_FILE = BASE_DIR / "tfidf_vectorizer.pkl"

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_artifacts():
    if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
        raise FileNotFoundError("Model files are missing. Run: python train_model.py")
    return joblib.load(MODEL_FILE), joblib.load(VECTORIZER_FILE)

def set_example(value):
    st.session_state["message_input"] = value

st.title("📩 SMS Spam Detector")
st.write("Enter an English SMS message. The model classifies it as spam or ham (normal).")
try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.text_area("SMS message", key="message_input", height=150, placeholder="Example: Congratulations! You won a free prize. Call now!")
c1, c2 = st.columns(2)
with c1:
    st.button("Load spam example", on_click=set_example, args=("Congratulations! You have won a free prize. Call now!",))
with c2:
    st.button("Load normal example", on_click=set_example, args=("Hey, are we meeting today?",))

if st.button("Check message", type="primary", use_container_width=True):
    raw = st.session_state.get("message_input", "")
    cleaned = clean_text(raw)
    if not raw.strip():
        st.warning("Enter a message first.")
    elif not cleaned:
        st.warning("This version supports English letters only. The text became empty after preprocessing.")
    else:
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index])
        st.subheader("Result")
        if prediction == "spam":
            st.error("🚨 Likely SPAM")
        else:
            st.success("✅ Likely HAM (normal message)")
        st.metric("Model confidence", f"{confidence:.1%}")
        st.caption("Confidence is the model's estimated probability, not a guarantee that the message is safe.")

st.divider()
st.caption("TF-IDF + Multinomial Naive Bayes | Trained on the SMS Spam Collection dataset")
