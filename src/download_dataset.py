"""Download the public Enron-labelled email CSV used by this project."""
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw_email_dataset.csv"
SOURCE_URL = "https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/Enron.csv"


def download_dataset() -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        print(f"Dataset already exists: {DATA_PATH}")
        return DATA_PATH
    print("Downloading public labelled email dataset...")
    urlretrieve(SOURCE_URL, DATA_PATH)
    print(f"Saved dataset to: {DATA_PATH}")
    return DATA_PATH


if __name__ == "__main__":
    download_dataset()
