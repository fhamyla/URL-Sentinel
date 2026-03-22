"""CLI prediction utility for phishing URL classification."""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np

from feature_extractor import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a URL is phishing or legitimate.")
    parser.add_argument("--model-path", default="phishing_model.pkl", help="Path to the trained model file.")
    parser.add_argument("--url", default=None, help="URL to classify. If omitted, interactive input is used.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        with open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        print(f"Model file not found: {args.model_path}")
        print("Run train_model.py first to generate phishing_model.pkl")
        sys.exit(1)

    url = args.url if args.url is not None else input("Enter URL: ").strip()
    if not url:
        print("Please provide a non-empty URL.")
        sys.exit(1)

    features = np.asarray([extract_features(url)], dtype=np.float32)
    prediction = int(model.predict(features)[0])

    if prediction == 1:
        print("⚠️ Phishing")
    else:
        print("✅ Legitimate")


if __name__ == "__main__":
    main()