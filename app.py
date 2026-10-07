import re
import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

stemmer = PorterStemmer()


def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [stemmer.stem(w) for w in text.split()
             if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(words)


@st.cache_resource(show_spinner="Training the model... (first run only)")
def train():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]].dropna().drop_duplicates()
    df.columns = ["label", "text"]
    df["y"] = (df["label"].str.lower().str.strip() == "spam").astype(int)
    df["clean"] = df["text"].apply(clean)

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5),
    }
    rows = []
    for name, m in models.items():
        m.fit(Xtr, y_tr)
        p = m.predict(Xte)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_te, p),
                     "Precision": precision_score(y_te, p),
                     "Recall": recall_score(y_te, p),
                     "F1": f1_score(y_te, p)})
    metrics = pd.DataFrame(rows).set_index("Model").round(4)

    lr = models["Logistic Regression"]
    feats = np.array(vec.get_feature_names_out())
    top = np.argsort(lr.coef_[0])[-15:]
    keywords = pd.Series(lr.coef_[0][top], index=feats[top])
    return vec, models, metrics, keywords, len(df)


vec, models, metrics, keywords, n_rows = train()

st.title("📧 Email / SMS Spam Classifier")
st.caption("Machine Learning + NLP (TF-IDF) | Naive Bayes, Logistic Regression, SVM")

model_name = st.sidebar.selectbox("Select model", list(models.keys()))
st.sidebar.write(f"Training data: **{n_rows}** messages")

examples = {
    "Spam example": "Congratulations! You have won a free prize. Call now to claim your reward!",
    "Ham example": "Hi, are we still meeting at 5pm today for the project discussion?",
}
c1, c2 = st.columns(2)
if c1.button("Try a spam example"):
    st.session_state["msg"] = examples["Spam example"]
if c2.button("Try a ham example"):
    st.session_state["msg"] = examples["Ham example"]

msg = st.text_area("Enter the email / SMS text:", key="msg", height=150)

if st.button("Check message", type="primary"):
    if not msg.strip():
        st.warning("Please enter a message first.")
    else:
        proba = models[model_name].predict_proba(vec.transform([clean(msg)]))[0]
        spam_p = float(proba[1])
        if spam_p >= 0.5:
            st.error(f"🚨 SPAM  ({spam_p:.1%} confidence)")
        else:
            st.success(f"✅ HAM / Not spam  ({1 - spam_p:.1%} confidence)")
        st.progress(spam_p, text=f"Spam probability: {spam_p:.1%}")

with st.expander("Model comparison"):
    st.dataframe(metrics)
with st.expander("Top spam-indicating keywords"):
    st.bar_chart(keywords)