"""Train and compare ML models for phishing URL detection."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from feature_extractor import FEATURE_NAMES, extract_features_batch

LABEL_MAP = {
    "legitimate": 0,
    "good": 0,
    "benign": 0,
    "safe": 0,
    "0": 0,
    0: 0,
    "phishing": 1,
    "bad": 1,
    "malicious": 1,
    "unsafe": 1,
    "1": 1,
    1: 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phishing URL detection models.")
    parser.add_argument("--dataset-path", default="dataset/phishing_site_urls.csv", help="Path to input CSV dataset.")
    parser.add_argument("--features-cache", default="features.csv", help="Path to cached extracted features CSV.")
    parser.add_argument("--model-path", default="phishing_model.pkl", help="Path to save the best model.")
    parser.add_argument("--chunksize", type=int, default=10_000, help="Chunk size for CSV reading.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows for fast experimentation.")
    parser.add_argument("--rebuild-features", action="store_true", help="Force rebuilding features cache.")
    parser.add_argument("--svm-verbose", action="store_true", help="Show low-level SVM training logs (useful for long runs).")
    return parser.parse_args()


def _normalize_label(raw_label: object) -> Optional[int]:
    if raw_label is None:
        return None
    label = str(raw_label).strip().lower()
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    try:
        numeric = int(float(label))
        return LABEL_MAP.get(numeric)
    except ValueError:
        return None


def _get_row_count(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as infile:
        return max(0, sum(1 for _ in infile) - 1)


def _iter_dataset_chunks(
    dataset_path: str,
    chunksize: int,
    max_rows: Optional[int],
) -> Iterator[Tuple[List[str], List[int]]]:
    """Yield chunked (urls, labels) while safely handling commas inside URLs."""
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as infile:
        header = infile.readline().strip().lower().replace(" ", "")
        if "url" not in header or "label" not in header:
            raise ValueError("Dataset must contain URL and Label columns in the header.")

        urls: List[str] = []
        labels: List[int] = []
        selected_rows = 0
        class_counts = {0: 0, 1: 0}
        target_zero = (max_rows // 2) if max_rows is not None else None
        target_one = (max_rows - (max_rows // 2)) if max_rows is not None else None

        for line in infile:
            if max_rows is not None and selected_rows >= max_rows:
                break

            clean_line = line.strip()
            if not clean_line or "," not in clean_line:
                continue

            url_part, label_part = clean_line.rsplit(",", 1)
            label = _normalize_label(label_part)
            if label is None:
                continue

            if max_rows is not None:
                if label == 0:
                    take_row = class_counts[0] < target_zero or class_counts[1] >= target_one
                else:
                    take_row = class_counts[1] < target_one or class_counts[0] >= target_zero
                if not take_row:
                    continue

            urls.append(url_part.strip())
            labels.append(label)
            class_counts[label] += 1
            selected_rows += 1

            if len(urls) >= chunksize:
                yield urls, labels
                urls, labels = [], []

        if urls:
            yield urls, labels


def build_or_load_features(
    dataset_path: str,
    features_cache: str,
    chunksize: int,
    max_rows: Optional[int],
    rebuild_features: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(features_cache) and not rebuild_features:
        print(f"Loading cached features from: {features_cache}")
        return load_features_from_cache(features_cache, chunksize=max(50_000, chunksize))

    total_rows = _get_row_count(dataset_path)
    target_rows = max_rows if max_rows else total_rows
    print(f"Building features from dataset: {dataset_path}")
    print(f"Rows target: {target_rows if target_rows else 'unknown'}")

    first_write = True
    processed_rows = 0
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    with tqdm(total=target_rows or None, desc="Extracting features", unit="url") as progress:
        chunk_iter = _iter_dataset_chunks(dataset_path, chunksize=chunksize, max_rows=max_rows)
        for urls, labels in chunk_iter:
            if not urls:
                continue

            features = extract_features_batch(urls, show_progress=False)
            labels_array = np.asarray(labels, dtype=np.int8)

            x_parts.append(features)
            y_parts.append(labels_array)
            processed_rows += len(urls)
            progress.update(len(urls))

            feature_frame = pd.DataFrame(features, columns=FEATURE_NAMES)
            feature_frame["label"] = labels_array
            write_mode = "w" if first_write else "a"
            feature_frame.to_csv(features_cache, mode=write_mode, header=first_write, index=False)
            first_write = False

    if not x_parts:
        raise ValueError("No valid rows found after URL/label preprocessing.")

    print(f"Saved feature cache to: {features_cache}")
    X = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int8)
    return X, y


def load_features_from_cache(cache_path: str, chunksize: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for chunk in pd.read_csv(cache_path, chunksize=chunksize):
        x_parts.append(chunk[FEATURE_NAMES].to_numpy(dtype=np.float32))
        y_parts.append(chunk["label"].to_numpy(dtype=np.int8))

    if not x_parts:
        raise ValueError("Feature cache is empty or unreadable.")

    return np.vstack(x_parts), np.concatenate(y_parts)


def train_and_evaluate_models(X: np.ndarray, y: np.ndarray, svm_verbose: bool = False) -> Tuple[object, Dict[str, Dict[str, float]]]:
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            "Training requires at least two classes. Increase --max-rows or disable it to include both legitimate and phishing samples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=False, verbose=svm_verbose)),
            ]
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    metrics: Dict[str, Dict[str, float]] = {}
    best_name: Optional[str] = None
    best_key = (-1.0, -1.0)
    best_model = None

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        start = time.perf_counter()
        print(f"[{time.strftime('%H:%M:%S')}] Starting {name} training...", flush=True)
        if name == "SVM" and X_train.shape[0] > 100_000:
            print(
                "  SVM can take a very long time on large datasets. "
                "Use --svm-verbose to see iterative progress.",
                flush=True,
            )

        current_model = clone(model)
        current_model.fit(X_train, y_train)
        duration = time.perf_counter() - start

        y_pred = current_model.predict(X_test)
        model_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "train_seconds": duration,
        }
        metrics[name] = model_metrics

        print(
            f"{name:<20} "
            f"acc={model_metrics['accuracy']:.4f} "
            f"prec={model_metrics['precision']:.4f} "
            f"rec={model_metrics['recall']:.4f} "
            f"f1={model_metrics['f1']:.4f} "
            f"time={model_metrics['train_seconds']:.2f}s"
        )
        sys.stdout.flush()

        model_key = (model_metrics["f1"], model_metrics["accuracy"])
        if model_key > best_key:
            best_key = model_key
            best_name = name
            best_model = current_model

        if name == "Random Forest":
            # Tree-based models do not need scaling; report feature importance for interpretability.
            importances = current_model.feature_importances_
            sorted_pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda item: item[1], reverse=True)
            print("  Random Forest feature importance (top 5):")
            for feature, score in sorted_pairs[:5]:
                print(f"  - {feature}: {score:.4f}")

    if best_model is None or best_name is None:
        raise RuntimeError("Model training failed; no model selected.")

    print(f"\nBest model: {best_name} (F1={best_key[0]:.4f}, Accuracy={best_key[1]:.4f})")
    return best_model, metrics


def print_comparison_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    print("\nModel Comparison Summary")
    print("=" * 92)
    print(f"{'Model':<22}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Train(s)':>12}")
    print("-" * 92)
    ordered = sorted(metrics.items(), key=lambda item: (item[1]["f1"], item[1]["accuracy"]), reverse=True)
    for name, m in ordered:
        print(
            f"{name:<22}{m['accuracy']:>12.4f}{m['precision']:>12.4f}{m['recall']:>12.4f}"
            f"{m['f1']:>12.4f}{m['train_seconds']:>12.2f}"
        )
    print("=" * 92)


def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()

    X, y = build_or_load_features(
        dataset_path=args.dataset_path,
        features_cache=args.features_cache,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        rebuild_features=args.rebuild_features,
    )
    print(f"Feature matrix shape: {X.shape}, labels shape: {y.shape}")

    best_model, metrics = train_and_evaluate_models(X, y, svm_verbose=args.svm_verbose)
    print_comparison_summary(metrics)

    with open(args.model_path, "wb") as model_file:
        pickle.dump(best_model, model_file)
    print(f"Saved best model to: {args.model_path}")

    total_time = time.perf_counter() - overall_start
    print(f"Total pipeline time: {total_time:.2f}s")


if __name__ == "__main__":
    main()