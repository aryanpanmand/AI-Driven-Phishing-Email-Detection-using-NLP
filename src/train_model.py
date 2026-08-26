"""Train, evaluate, compare, and save phishing email text classifiers."""
import json
from pathlib import Path
import joblib
import matplotlib
# Use a non-GUI backend so charts can be generated on any machine or server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.download_dataset import download_dataset
from src.preprocessing import clean_email_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR, MODELS_DIR = PROJECT_ROOT / "results", PROJECT_ROOT / "models"


def load_and_prepare_data() -> pd.DataFrame:
    """Load expected columns, remove unusable rows, and clean email text."""
    data = pd.read_csv(download_dataset(), usecols=["body", "label"])
    data = data.dropna(subset=["body", "label"]).copy()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data[data["label"].isin([0, 1])].copy()
    data["label"] = data["label"].astype(int)
    data["clean_text"] = data["body"].map(clean_email_text)
    data = data[data["clean_text"].str.len() > 0].drop_duplicates(subset=["clean_text"])
    if data["label"].nunique() != 2:
        raise ValueError("The dataset must contain both label 0 and label 1.")
    return data[["clean_text", "label"]]


def save_charts(y_true, y_pred, scores: dict) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
    plt.title("Logistic Regression Confusion Matrix"); plt.xlabel("Predicted label"); plt.ylabel("Actual label")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160); plt.close()
    table = pd.DataFrame(scores).T.reset_index(names="Model")
    plot_data = table.melt(id_vars="Model", value_vars=["accuracy", "f1"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(7, 4.5)); sns.barplot(data=plot_data, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.ylim(0, 1); plt.title("Model Comparison on the Test Set")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=160); plt.close()


def train() -> dict:
    """Run the reproducible training pipeline and return real evaluation values."""
    RESULTS_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)
    data = load_and_prepare_data()
    x_train, x_test, y_train, y_test = train_test_split(data["clean_text"], data["label"], test_size=0.20, random_state=42, stratify=data["label"])
    # TF-IDF turns text into numeric word/phrase weights; informative words get higher values.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=30000)
    x_train_tfidf, x_test_tfidf = vectorizer.fit_transform(x_train), vectorizer.transform(x_test)
    candidates = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), "Multinomial Naive Bayes": MultinomialNB()}
    scores, fitted = {}, {}
    for name, model in candidates.items():
        model.fit(x_train_tfidf, y_train); predicted = model.predict(x_test_tfidf)
        scores[name] = {"accuracy": round(float(accuracy_score(y_test, predicted)), 4), "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4), "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4), "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4)}
        fitted[name] = model
    final_name = "Logistic Regression"  # Primary requested model; baseline metrics are retained for comparison.
    final_model, final_predictions = fitted[final_name], fitted[final_name].predict(x_test_tfidf)
    joblib.dump(final_model, MODELS_DIR / "phishing_model.pkl"); joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    save_charts(y_test, final_predictions, scores)
    report = classification_report(y_test, final_predictions, target_names=["legitimate", "phishing"], digits=4)
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    summary = {"dataset_samples_after_preprocessing": int(len(data)), "class_counts": {"legitimate (0)": int((data.label == 0).sum()), "phishing (1)": int((data.label == 1).sum())}, "train_samples": int(len(x_train)), "test_samples": int(len(x_test)), "selected_model": final_name, "model_metrics": scores}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); print("\nClassification report for Logistic Regression:\n" + report)
    return summary


if __name__ == "__main__":
    train()
