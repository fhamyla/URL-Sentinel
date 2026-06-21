"""Database connection and models for phishing URL predictions - Firebase Firestore & In-Memory Fallback."""

from __future__ import annotations

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_backend_dir, ".env")
load_dotenv(_env_path)

# Initialize Firebase Admin SDK
_firebase_initialized = False
db_client = None

project_id = os.getenv("FIREBASE_PROJECT_ID")
service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")

if project_id and service_account_path:
    try:
        if service_account_path.strip().startswith("{"):
            # Load credentials from inline JSON string
            cred_dict = json.loads(service_account_path)
            cred = credentials.Certificate(cred_dict)
        else:
            # Load credentials from file path
            full_path = service_account_path
            if not os.path.isabs(full_path):
                full_path = os.path.join(_backend_dir, full_path)
            cred = credentials.Certificate(full_path)
            
        firebase_admin.initialize_app(cred, {
            'projectId': project_id
        })
        db_client = firestore.client()
        _firebase_initialized = True
        print(f"Firebase Admin SDK initialized successfully (Project: {project_id}).")
    except Exception as e:
        print(f"Failed to initialize Firebase Admin SDK: {e}")
else:
    print("Firebase credentials not fully set. Falling back to in-memory database.")

# Global in-memory lists to mock database tables as fallbacks
_predictions_in_memory: list[dict] = []
_runs_in_memory: list[dict] = []
_pred_id_counter = 1
_run_id_counter = 1

def create_tables() -> None:
    """Create all tables if they don't exist - mock/no-op."""
    pass

def insert_prediction(
    url: str,
    prediction: int,
    score: float,
    threshold: float,
    model_name: str,
    session_id: str = "",
) -> bool:
    """Insert a prediction (to Firestore and in-memory)."""
    global _pred_id_counter
    created_at_str = datetime.now().isoformat()
    prediction_item = {
        "id": _pred_id_counter,
        "url": url,
        "prediction": prediction,
        "score": score,
        "threshold": threshold,
        "model_name": model_name,
        "session_id": session_id,
        "created_at": created_at_str
    }
    # Always store to in-memory lists as fallback
    _predictions_in_memory.append(prediction_item)
    _pred_id_counter += 1

    if _firebase_initialized and db_client:
        try:
            doc_ref = db_client.collection("predictions").document()
            doc_ref.set({
                "url": url,
                "prediction": prediction,
                "score": score,
                "threshold": threshold,
                "model_name": model_name,
                "session_id": session_id,
                "created_at": created_at_str
            })
            return True
        except Exception as e:
            print(f"Firestore insert_prediction failed: {e}. Falling back to in-memory.")
            return True
    return True

def insert_training_run(
    model_name: str,
    best_threshold: float,
    best_f1: float,
    best_accuracy: float,
    rows_count: int,
    feature_count: int,
    metrics: dict,
) -> bool:
    """Insert a training run summary (to Firestore and in-memory)."""
    global _run_id_counter
    created_at_str = datetime.now().isoformat()
    run_item = {
        "id": _run_id_counter,
        "model_name": model_name,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "best_accuracy": best_accuracy,
        "rows_count": rows_count,
        "feature_count": feature_count,
        "metrics_json": metrics,
        "created_at": created_at_str
    }
    _runs_in_memory.append(run_item)
    _run_id_counter += 1

    if _firebase_initialized and db_client:
        try:
            doc_ref = db_client.collection("training_runs").document()
            doc_ref.set({
                "model_name": model_name,
                "best_threshold": best_threshold,
                "best_f1": best_f1,
                "best_accuracy": best_accuracy,
                "rows_count": rows_count,
                "feature_count": feature_count,
                "metrics_json": metrics,
                "created_at": created_at_str
            })
            return True
        except Exception as e:
            print(f"Firestore insert_training_run failed: {e}. Falling back to in-memory.")
            return True
    return True

def get_recent_predictions(limit: int = 20, session_id: str = "") -> list[dict]:
    """Fetch the latest predictions from Firestore or memory.

    When *session_id* is provided, only predictions belonging to that
    session are returned.  When empty, an empty list is returned so that
    unauthenticated callers never see another user's scan history.
    """
    if not session_id:
        return []

    if _firebase_initialized and db_client:
        try:
            predictions_ref = db_client.collection("predictions")
            query = (
                predictions_ref
                .where("session_id", "==", session_id)
                .order_by("created_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            docs = query.stream()
            results = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                results.append(data)
            return results
        except Exception as e:
            print(f"Firestore get_recent_predictions failed: {e}. Falling back to in-memory.")

    # Fallback to local in-memory predictions filtered by session_id
    filtered = [p for p in _predictions_in_memory if p.get("session_id") == session_id]
    return sorted(filtered, key=lambda x: x["id"], reverse=True)[:limit]

def get_recent_training_runs(limit: int = 10) -> list[dict]:
    """Fetch the latest training runs from Firestore or memory."""
    if _firebase_initialized and db_client:
        try:
            runs_ref = db_client.collection("training_runs")
            query = runs_ref.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
            docs = query.stream()
            results = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                results.append(data)
            return results
        except Exception as e:
            print(f"Firestore get_recent_training_runs failed: {e}. Falling back to in-memory.")

    # Fallback to local in-memory training runs
    return sorted(_runs_in_memory, key=lambda x: x["id"], reverse=True)[:limit]