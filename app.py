import streamlit as st
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📧 Spam Classifier")

text = st.text_area("Enter email text")

if st.button("Predict"):
    if text.strip():
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        st.success("SPAM" if pred == 1 else "NOT SPAM")
    else:
        st.warning("Please enter text")
