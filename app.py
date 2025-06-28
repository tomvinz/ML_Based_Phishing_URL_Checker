import streamlit as st
import joblib

st.title("🔒 Phishing URL Detector")

url_input = st.text_input("Enter a URL to check")

if st.button("Check"):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    
    input_vec = vectorizer.transform([url_input])
    prediction = model.predict(input_vec)

    if prediction[0] == 'bad':
        st.error("⚠️ This URL is likely malicious!")
    else:
        st.success("✅ This URL seems safe.")
