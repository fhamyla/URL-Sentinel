"""CLI prediction utility for phishing URL classification."""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np

from feature_extractor import extract_features
from db import insert_prediction

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a URL is phishing or legitimate.")
    parser.add_argument("--model-path", default="phishing_model.pkl", help="Path to the trained model file.")
    parser.add_argument("--url", default=None, help="URL to classify. If omitted, interactive input is used.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        with open(args.model_path, "rb") as model_file:
            loaded_object = pickle.load(model_file)
    except FileNotFoundError:
        print(f"Model file not found: {args.model_path}")
        print("Run train_model.py first to generate phishing_model.pkl")
        sys.exit(1)

    threshold = 0.5
    best_model_name = "Unknown"
    if isinstance(loaded_object, dict) and "model" in loaded_object:
        model = loaded_object["model"]
        threshold = float(loaded_object.get("threshold", 0.5))
        best_model_name = loaded_object.get("best_model_name", "Unknown")
    else:
        model = loaded_object

    url = args.url if args.url is not None else input("Enter URL: ").strip()
    if not url:
        print("Please provide a non-empty URL.")
        sys.exit(1)

    features = np.asarray([extract_features(url)], dtype=np.float32)

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(features)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(features)[0])
    else:
        prediction = int(model.predict(features)[0])
        score = float(prediction)

    prediction = int(score >= threshold)

    if prediction == 1:
        print(f"⚠️ Phishing (score={score:.4f}, threshold={threshold:.4f})")
    else:
        print(f"✅ Legitimate (score={score:.4f}, threshold={threshold:.4f})")

    saved = insert_prediction(url, prediction, score, threshold, best_model_name)
    if saved:
        print(f"✓ Prediction saved to database")
    else:
        print(f"(Database not configured; prediction not saved)")


if __name__ == "__main__":
    main()