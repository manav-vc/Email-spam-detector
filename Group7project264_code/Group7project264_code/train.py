"""
Email Spam Detector - Training Script
COMP264 - Cloud Machine Learning - Group 7

This script trains a Multinomial Naive Bayes classifier on the Data5.csv
email dataset using a CountVectorizer + TF-IDF pipeline, performs
hyperparameter tuning on the smoothing parameter alpha, evaluates the
model, and saves the trained pipeline for deployment on AWS SageMaker.

Output artifacts:
    model/spam_pipeline.joblib   - full sklearn Pipeline (vectorizer + tfidf + NB)
    model/metrics.json           - accuracy, confusion matrix, best alpha
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def load_data(data_path: str) -> pd.DataFrame:
    """Load the email dataset and return a clean DataFrame with Body + Label."""
    df = pd.read_csv(data_path)
    df = df[["Body", "Label"]]
    df.dropna(inplace=True)
    df["Body"] = df["Body"].astype(str).str.lower().str.strip()
    df["Label"] = df["Label"].astype(int)
    return df


def build_pipeline() -> Pipeline:
    """Construct the training pipeline (vectorizer -> tfidf -> classifier)."""
    return Pipeline(
        steps=[
            ("count", CountVectorizer(stop_words="english", max_df=0.9, min_df=2)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )


def tune_alpha(pipeline: Pipeline, X_train, y_train) -> float:
    """Simple grid search over NB alpha using 3-fold cross validation."""
    best_alpha, best_score = 1.0, 0.0
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0]:
        pipeline.set_params(clf__alpha=alpha)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy")
        mean_cv = cv_scores.mean()
        print(f"  alpha={alpha:<5}  mean CV accuracy={mean_cv:.4f}")
        if mean_cv > best_score:
            best_score, best_alpha = mean_cv, alpha
    return best_alpha


def main(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("Email Spam Detector - Training")
    print("=" * 60)

    # 1. Load
    df = load_data(args.data_path)
    print(f"\nDataset shape: {df.shape}")
    print("Class distribution:")
    print(df["Label"].value_counts())

    # 2. Train/Test split (stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        df["Body"],
        df["Label"],
        test_size=0.20,
        random_state=42,
        stratify=df["Label"],
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. Build pipeline and tune alpha
    pipeline = build_pipeline()
    print("\nHyperparameter tuning on alpha...")
    best_alpha = tune_alpha(pipeline, X_train, y_train)
    print(f"Best alpha: {best_alpha}")

    # 4. Fit final pipeline with best alpha
    pipeline.set_params(clf__alpha=best_alpha)
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\nTest Accuracy:", round(acc, 4))
    print("Confusion Matrix:")
    print(np.array(cm))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the pipeline + metrics
    os.makedirs(args.model_dir, exist_ok=True)
    pipeline_path = os.path.join(args.model_dir, "spam_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"\nSaved trained pipeline -> {pipeline_path}")

    metrics = {
        "best_alpha": best_alpha,
        "test_accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics       -> {os.path.join(args.model_dir, 'metrics.json')}")

    # 7. Quick sanity check with a few custom samples
    samples = [
        "Hi team, please find attached the minutes from yesterday's meeting.",
        "Congratulations! You've won a FREE iPhone, click here to claim NOW!",
        "Your account has been compromised, verify your password immediately!",
        "Looking forward to catching up with you this weekend.",
    ]
    preds = pipeline.predict(samples)
    print("\nSample predictions:")
    for text, p in zip(samples, preds):
        label = "SPAM" if int(p) == 1 else "HAM"
        print(f"  [{label}] {text[:70]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/Data5.csv")
    parser.add_argument("--model-dir", type=str, default="model")
    args = parser.parse_args()
    main(args)
