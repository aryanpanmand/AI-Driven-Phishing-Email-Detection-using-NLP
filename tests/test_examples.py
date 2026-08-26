"""Smoke tests for saved artifacts using representative demonstration emails."""
import sys
from pathlib import Path

# Make this file runnable directly with: python tests/test_examples.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import load_artifacts, predict_email

CASES = [
    ("Urgent verify your account password now by clicking this link.", "PHISHING"),
    ("Your bank account is locked. Send your PIN to restore access today.", "PHISHING"),
    ("The department meeting will take place at 2 PM tomorrow in room 301.", "LEGITIMATE"),
    ("Dear students, lecture notes have been uploaded to the course portal.", "LEGITIMATE"),
]

def main() -> None:
    model, vectorizer = load_artifacts(); passed = 0
    for text, expected in CASES:
        actual = predict_email(text, model, vectorizer)["label"]
        passed += actual == expected
        print(f"Expected: {expected:<10} Predicted: {actual:<10} | {text}")
    print(f"\nSmoke tests passed: {passed}/{len(CASES)}")
    if passed != len(CASES): raise SystemExit("One or more smoke tests failed.")

if __name__ == "__main__": main()
