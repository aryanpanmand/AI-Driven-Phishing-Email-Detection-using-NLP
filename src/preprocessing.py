"""Simple, shared text cleaning used by training and prediction."""
import re


def clean_email_text(text: object) -> str:
    """Return a normalized email string without removing useful phishing words."""
    if text is None:
        return ""
    text = str(text).lower()
    # Keep URL tokens, but replace the changing address with a stable word.
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " emailaddress ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
