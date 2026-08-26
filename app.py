"""Streamlit demonstration interface; it loads saved artifacts and never retrains."""
import streamlit as st
from src.predict import load_artifacts, predict_email

EXAMPLES = {
    "Select an example": "",
    "Phishing: password scam": "Urgent: Your account will be suspended today. Verify your password immediately at http://secure-account-check.example.com to avoid losing access.",
    "Phishing: fake bank": "Security alert from your bank. We detected unusual activity. Click the link now and confirm your card number and PIN.",
    "Legitimate: professional": "Hello team, the project meeting is scheduled for Thursday at 10 AM in Conference Room B. Please review the attached agenda before the meeting.",
    "Legitimate: academic": "Dear students, the assignment deadline for Data Structures is Friday at 5 PM. Submit your work through the university learning portal.",
}

st.set_page_config(page_title="AI-Driven Phishing Email Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ AI-Driven Phishing Email Detector")
st.caption("Educational demonstration: a TF-IDF + Logistic Regression email-text classifier.")
try:
    model, vectorizer = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()

choice = st.selectbox("Try a demonstration example", list(EXAMPLES))
email_text = st.text_area("Paste the email text", value=EXAMPLES[choice], height=260, placeholder="Paste an email body here...")
if st.button("Analyze Email", type="primary"):
    try:
        result = predict_email(email_text, model, vectorizer)
        if result["label"] == "PHISHING": st.error("🚨 PHISHING EMAIL")
        else: st.success("✅ LEGITIMATE EMAIL")
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        st.caption(f"Estimated phishing probability: {result['phishing_probability']:.1%}")
        if result["indicators"]: st.write("Possible phishing-related words from this model:", ", ".join(result["indicators"]))
        else: st.write("No strong phishing-related words were found among the model's most influential text features.")
    except ValueError as error: st.warning(str(error))
st.divider()
st.info("Limitation: this educational model uses email text only. It cannot guarantee detection of every phishing email and should not replace human review or security controls.")
