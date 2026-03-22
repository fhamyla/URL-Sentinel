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
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from feature_extractor import FEATURE_NAMES, extract_features_batch, extract_registered_domain

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
    parser.add_argument("--skip-svm", action="store_true", help="Skip SVM for faster runs on very large datasets.")
    parser.add_argument("--tune-sample-size", type=int, default=120_000, help="Max rows from train split used for tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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


def _optimize_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Find threshold that maximizes F1 on validation data."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if thresholds.size == 0:
        return 0.5

    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx])


def _get_model_scores(model: object, X: np.ndarray) -> np.ndarray:
    """Return continuous scores suitable for threshold optimization."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(np.float32)


def _scores_to_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.int8)


def _sample_for_tuning(X: np.ndarray, y: np.ndarray, max_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_size:
        return X, y
    _, sampled_X, _, sampled_y = train_test_split(
        X,
        y,
        test_size=max_size,
        random_state=seed,
        stratify=y,
    )
    return sampled_X, sampled_y


def _candidate_params(model_name: str) -> List[Dict[str, object]]:
    candidates: Dict[str, List[Dict[str, object]]] = {
        "Logistic Regression": [
            {"model__C": 0.5},
            {"model__C": 1.0},
            {"model__C": 2.0},
        ],
        "Decision Tree": [
            {"max_depth": 10, "min_samples_leaf": 1},
            {"max_depth": 20, "min_samples_leaf": 2},
            {"max_depth": None, "min_samples_leaf": 5},
        ],
        "Random Forest": [
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 300, "max_depth": 30, "min_samples_leaf": 2},
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3},
        ],
        "SVM": [
            {"model__C": 1.0, "model__gamma": "scale"},
            {"model__C": 2.0, "model__gamma": "scale"},
            {"model__C": 1.0, "model__gamma": "auto"},
        ],
        "Gradient Boosting": [
            {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3},
            {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 3},
            {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
        ],
        "Extra Trees": [
            {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 350, "max_depth": 30, "min_samples_leaf": 2},
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 4},
        ],
        "Hist Gradient Boosting": [
            {"max_iter": 200, "learning_rate": 0.05, "max_depth": 8},
            {"max_iter": 300, "learning_rate": 0.05, "max_depth": 8},
            {"max_iter": 250, "learning_rate": 0.08, "max_depth": 10},
        ],
    }
    return candidates.get(model_name, [{}])


def build_or_load_features(
    dataset_path: str,
    features_cache: str,
    chunksize: int,
    max_rows: Optional[int],
    rebuild_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if os.path.exists(features_cache) and not rebuild_features:
        print(f"Loading cached features from: {features_cache}")
        return load_features_from_cache(features_cache, chunksize=max(50_000, chunksize))

    total_rows = _get_row_count(dataset_path)
    target_rows = max_rows if max_rows else total_rows
    print(f"Building features from dataset: {dataset_path}")
    print(f"Rows target: {target_rows if target_rows else 'unknown'}")

    first_write = True
    processed_rows = 0
    duplicate_count = 0
    seen_urls: set[str] = set()
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    group_parts: List[np.ndarray] = []

    with tqdm(total=target_rows or None, desc="Extracting features", unit="url") as progress:
        chunk_iter = _iter_dataset_chunks(dataset_path, chunksize=chunksize, max_rows=max_rows)
        for urls, labels in chunk_iter:
            if not urls:
                continue

            cleaned_urls: List[str] = []
            cleaned_labels: List[int] = []
            group_keys: List[str] = []
            for raw_url, label in zip(urls, labels):
                url = str(raw_url).strip()
                if len(url) < 4:
                    continue
                dedupe_key = url.lower()
                if dedupe_key in seen_urls:
                    duplicate_count += 1
                    continue
                seen_urls.add(dedupe_key)
                cleaned_urls.append(url)
                cleaned_labels.append(label)
                domain_key = extract_registered_domain(url) or "__unknown__"
                group_keys.append(domain_key)

            if not cleaned_urls:
                continue

            features = extract_features_batch(cleaned_urls, show_progress=False)
            labels_array = np.asarray(cleaned_labels, dtype=np.int8)
            groups_array = np.asarray(group_keys, dtype=object)

            x_parts.append(features)
            y_parts.append(labels_array)
            group_parts.append(groups_array)
            processed_rows += len(cleaned_urls)
            progress.update(len(cleaned_urls))

            feature_frame = pd.DataFrame(features, columns=FEATURE_NAMES)
            feature_frame["label"] = labels_array
            feature_frame["group_key"] = groups_array
            write_mode = "w" if first_write else "a"
            feature_frame.to_csv(features_cache, mode=write_mode, header=first_write, index=False)
            first_write = False

    if not x_parts:
        raise ValueError("No valid rows found after URL/label preprocessing.")

    print(f"Saved feature cache to: {features_cache}")
    print(f"Removed duplicate URLs during build: {duplicate_count}")
    X = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int8)
    groups = np.concatenate(group_parts)
    return X, y, groups


def load_features_from_cache(cache_path: str, chunksize: int = 100_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    group_parts: List[np.ndarray] = []
    for chunk in pd.read_csv(cache_path, chunksize=chunksize):
        x_parts.append(chunk[FEATURE_NAMES].to_numpy(dtype=np.float32))
        y_parts.append(chunk["label"].to_numpy(dtype=np.int8))
        if "group_key" in chunk.columns:
            group_parts.append(chunk["group_key"].fillna("__unknown__").to_numpy(dtype=object))
        else:
            group_parts.append(np.full(shape=len(chunk), fill_value="__unknown__", dtype=object))

    if not x_parts:
        raise ValueError("Feature cache is empty or unreadable.")

    return np.vstack(x_parts), np.concatenate(y_parts), np.concatenate(group_parts)


def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    svm_verbose: bool = False,
    skip_svm: bool = False,
    tune_sample_size: int = 120_000,
    seed: int = 42,
) -> Tuple[object, float, str, Dict[str, Dict[str, float]]]:
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            "Training requires at least two classes. Increase --max-rows or disable it to include both legitimate and phishing samples."
        )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=seed,
        stratify=y_train_val,
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1200, solver="lbfgs", class_weight="balanced")),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced_subsample"),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=False, verbose=svm_verbose, class_weight="balanced")),
            ]
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=seed),
    }

    if skip_svm:
        models.pop("SVM", None)

    metrics: Dict[str, Dict[str, float]] = {}
    best_name: Optional[str] = None
    best_threshold = 0.5
    best_key = (-1.0, -1.0)
    best_model = None

    tune_X, tune_y = _sample_for_tuning(X_train, y_train, max_size=tune_sample_size, seed=seed)

    print(
        f"Split sizes -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)} | "
        f"train positive ratio: {np.mean(y_train):.4f}"
    )

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

        best_candidate_model = None
        best_candidate_threshold = 0.5
        best_candidate_key = (-1.0, -1.0)
        best_params: Dict[str, object] = {}

        for params in _candidate_params(name):
            candidate_model = clone(model)
            if params:
                candidate_model.set_params(**params)
            candidate_model.fit(tune_X, tune_y)

            val_scores = _get_model_scores(candidate_model, X_val)
            threshold = _optimize_threshold(y_val, val_scores)
            val_pred = _scores_to_predictions(val_scores, threshold)
            candidate_key = (f1_score(y_val, val_pred, zero_division=0), accuracy_score(y_val, val_pred))

            if candidate_key > best_candidate_key:
                best_candidate_key = candidate_key
                best_candidate_model = candidate_model
                best_candidate_threshold = threshold
                best_params = params

        if best_candidate_model is None:
            raise RuntimeError(f"Failed to tune model: {name}")

        current_model = best_candidate_model
        threshold = best_candidate_threshold
        duration = time.perf_counter() - start

        y_scores = _get_model_scores(current_model, X_test)
        y_pred = _scores_to_predictions(y_scores, threshold)
        model_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "threshold": float(threshold),
            "train_seconds": duration,
        }
        metrics[name] = model_metrics

        print(
            f"{name:<20} "
            f"acc={model_metrics['accuracy']:.4f} "
            f"prec={model_metrics['precision']:.4f} "
            f"rec={model_metrics['recall']:.4f} "
            f"f1={model_metrics['f1']:.4f} "
            f"thr={model_metrics['threshold']:.4f} "
            f"time={model_metrics['train_seconds']:.2f}s"
        )
        if best_params:
            print(f"  tuned params: {best_params}")
        sys.stdout.flush()

        model_key = (model_metrics["f1"], model_metrics["accuracy"])
        if model_key > best_key:
            best_key = model_key
            best_name = name
            best_model = current_model
            best_threshold = threshold

        if name == "Random Forest":
            importances = current_model.feature_importances_
            sorted_pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda item: item[1], reverse=True)
            print("  Random Forest feature importance (top 5):")
            for feature, score in sorted_pairs[:5]:
                print(f"  - {feature}: {score:.4f}")

    if best_model is None or best_name is None:
        raise RuntimeError("Model training failed; no model selected.")

    print(f"\nBest model: {best_name} (F1={best_key[0]:.4f}, Accuracy={best_key[1]:.4f}, Threshold={best_threshold:.4f})")
    return best_model, best_threshold, best_name, metrics


def print_comparison_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    print("\nModel Comparison Summary")
    print("=" * 108)
    print(
        f"{'Model':<24}{'Accuracy':>12}{'Accuracy%':>11}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Thresh':>10}{'Train(s)':>12}"
    )
    print("-" * 108)
    ordered = sorted(metrics.items(), key=lambda item: (item[1]["f1"], item[1]["accuracy"]), reverse=True)
    for name, m in ordered:
        print(
            f"{name:<24}{m['accuracy']:>12.4f}{(m['accuracy'] * 100):>11.2f}%{m['precision']:>12.4f}{m['recall']:>12.4f}"
            f"{m['f1']:>12.4f}{m['threshold']:>10.4f}{m['train_seconds']:>12.2f}"
        )
    print("=" * 108)


def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()

    X, y, groups = build_or_load_features(
        dataset_path=args.dataset_path,
        features_cache=args.features_cache,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        rebuild_features=args.rebuild_features,
    )
    print(f"Feature matrix shape: {X.shape}, labels shape: {y.shape}, unique domains: {len(np.unique(groups))}")

    best_model, best_threshold, best_name, metrics = train_and_evaluate_models(
        X,
        y,
        svm_verbose=args.svm_verbose,
        skip_svm=args.skip_svm,
        tune_sample_size=args.tune_sample_size,
        seed=args.seed,
    )
    print_comparison_summary(metrics)

    model_bundle = {
        "model": best_model,
        "threshold": best_threshold,
        "feature_names": FEATURE_NAMES,
        "best_model_name": best_name,
        "metrics": metrics,
    }
    with open(args.model_path, "wb") as model_file:
        pickle.dump(model_bundle, model_file)
    print(f"Saved best model to: {args.model_path}")

    total_time = time.perf_counter() - overall_start
    print(f"Total pipeline time: {total_time:.2f}s")


if __name__ == "__main__":
    main()