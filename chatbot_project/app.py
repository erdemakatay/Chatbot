# app.py

import joblib
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords

# Bu en üstteki streamlit komutu olmalı
st.set_page_config(page_title="Öznel / Nesnel Chatbot", page_icon="💬")

# ----------- Load model & vectorizer -----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logreg_subjectivity.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    return model, tfidf

model, tfidf = load_artifacts()

# ----------- Basic preprocess ------------------------
nltk.download("stopwords", quiet=True)
stop_tr = set(stopwords.words("turkish"))

def preprocess(text: str) -> str:
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", " ", text.lower())
    tokens = [w for w in text.split() if w not in stop_tr]
    return " ".join(tokens)

def classify(text: str) -> str:
    clean = preprocess(text)
    vec   = tfidf.transform([clean])
    label = model.predict(vec)[0]
    return "Öznel" if label == 1 else "Nesnel"

# ----------- Streamlit UI ----------------------------
st.title("💬 Öznel–Nesnel Sınıflayıcı Chatbot")
st.markdown(
    """
    Yazdığınız cümleyi eğittiğimiz **Logistic Regression** modeline gönderiyorum  
    ve sonucunda cümlenin **öznel mi nesnel mi** olduğunu söylüyorum.
    """
)

# Chat-style container
if "history" not in st.session_state:
    st.session_state.history = []

# Input box at bottom
user_input = st.chat_input("Cümlenizi yazın ve Enter'a basın...")

# When user sends message
if user_input:
    result = classify(user_input)
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", result))

# Display history
for sender, msg in st.session_state.history:
    if sender == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(f"**{msg}**")
