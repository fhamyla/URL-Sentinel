"""Database connection and models for phishing URL predictions - In-Memory Mock (DB Removed)."""

from __future__ import annotations

import json
from datetime import datetime

# Global in-memory lists to mock database tables
_predictions_in_memory: list[dict] = []
_runs_in_memory: list[dict] = []
_pred_id_counter = 1
_run_id_counter = 1

def create_tables() -> None:
    """Mock create tables - do nothing."""
    pass

def insert_prediction(
    url: str,
    prediction: int,
    score: float,
    threshold: float,
    model_name: str,
) -> bool:
    """Insert a prediction into memory."""
    global _pred_id_counter
    try:
        _predictions_in_memory.append({
            "id": _pred_id_counter,
            "url": url,
            "prediction": prediction,
            "score": score,
            "threshold": threshold,
            "model_name": model_name,
            "created_at": datetime.now().isoformat()
        })
        _pred_id_counter += 1
        return True
    except Exception as e:
        print(f"Mock insert prediction failed: {e}")
        return False

def insert_training_run(
    model_name: str,
    best_threshold: float,
    best_f1: float,
    best_accuracy: float,
    rows_count: int,
    feature_count: int,
    metrics: dict,
) -> bool:
    """Insert a training run summary into memory."""
    global _run_id_counter
    try:
        _runs_in_memory.append({
            "id": _run_id_counter,
            "model_name": model_name,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "best_accuracy": best_accuracy,
            "rows_count": rows_count,
            "feature_count": feature_count,
            "metrics_json": metrics,
            "created_at": datetime.now().isoformat()
        })
        _run_id_counter += 1
        return True
    except Exception as e:
        print(f"Mock insert training run failed: {e}")
        return False

def get_recent_predictions(limit: int = 20) -> list[dict]:
    """Fetch the latest predictions from memory."""
    return sorted(_predictions_in_memory, key=lambda x: x["id"], reverse=True)[:limit]

def get_recent_training_runs(limit: int = 10) -> list[dict]:
    """Fetch the latest training runs from memory."""
    return sorted(_runs_in_memory, key=lambda x: x["id"], reverse=True)[:limit]