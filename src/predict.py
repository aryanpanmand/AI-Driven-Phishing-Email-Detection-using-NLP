"""Load saved artifacts and predict a single email."""
from pathlib import Path
import joblib

from src.preprocessing import clean_email_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "phishing_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files are missing. Run: python -m src.train_model")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def predict_email(email_text: str, model=None, vectorizer=None) -> dict:
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("Please enter email text to analyze.")
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()
    features = vectorizer.transform([clean_email_text(email_text)])
    phishing_probability = float(model.predict_proba(features)[0, 1])
    is_phishing = phishing_probability >= 0.5
    feature_names, coefficients = vectorizer.get_feature_names_out(), model.coef_[0]
    present = features.nonzero()[1]
    contributions = sorted(((feature_names[i], float(features[0, i] * coefficients[i])) for i in present), key=lambda pair: abs(pair[1]), reverse=True)[:5]
    return {"label": "PHISHING" if is_phishing else "LEGITIMATE", "phishing_probability": phishing_probability, "confidence": phishing_probability if is_phishing else 1 - phishing_probability, "indicators": [word for word, score in contributions if score > 0]}
