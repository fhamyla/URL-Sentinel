"""Database connection and models for phishing URL predictions with SQLite fallback."""

from __future__ import annotations

import json
import os
import re
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_SCHEMA = os.getenv("DB_SCHEMA", "sentinel")

_engine = None


def _safe_schema_name(schema_name: str) -> str:
    """Return a validated SQL identifier for schema usage."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema_name):
        raise ValueError(f"Invalid DB_SCHEMA: {schema_name!r}")
    return schema_name


def get_db_engine():
    """Retrieve SQLAlchemy engine. Fall back to SQLite if PostgreSQL is unavailable."""
    global _engine
    if _engine is not None:
        return _engine

    # Try PostgreSQL first if configured
    if DATABASE_URL:
        try:
            # Set a low timeout so we don't hang if the DB is unreachable
            engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 3})
            # Test connection
            with engine.connect() as conn:
                pass
            _engine = engine
            print("Successfully connected to PostgreSQL database.")
            return _engine
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")

    # Fall back to SQLite inside the workspace
    db_dir = os.path.dirname(os.path.abspath(__file__))
    sqlite_path = os.path.join(db_dir, "url_sentinel.db")
    engine = create_engine(f"sqlite:///{sqlite_path}")
    _engine = engine
    print(f"Using SQLite database at: {sqlite_path}")
    return _engine


def _ensure_predictions_table(engine) -> None:
    """Create predictions and training tables if they do not exist."""
    is_sqlite = engine.dialect.name == "sqlite"

    with engine.begin() as conn:
        if is_sqlite:
            # SQLite does not support custom schemas or BIGSERIAL/JSONB
            conn.execute(
                text("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL,
                        prediction INTEGER NOT NULL,
                        score REAL,
                        threshold REAL,
                        model_name TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            )
            conn.execute(
                text("""
                    CREATE TABLE IF NOT EXISTS training_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        best_threshold REAL,
                        best_f1 REAL,
                        best_accuracy REAL,
                        rows_count INTEGER,
                        feature_count INTEGER,
                        metrics_json TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            )
        else:
            schema = _safe_schema_name(DB_SCHEMA)
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            conn.execute(
                text(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.predictions (
                        id BIGSERIAL PRIMARY KEY,
                        url TEXT NOT NULL,
                        prediction INTEGER NOT NULL,
                        score DOUBLE PRECISION,
                        threshold DOUBLE PRECISION,
                        model_name TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
            )
            conn.execute(
                text(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.training_runs (
                        id BIGSERIAL PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        best_threshold DOUBLE PRECISION,
                        best_f1 DOUBLE PRECISION,
                        best_accuracy DOUBLE PRECISION,
                        rows_count INTEGER,
                        feature_count INTEGER,
                        metrics_json JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
            )


def create_tables() -> None:
    """Create all tables if they don't exist."""
    engine = get_db_engine()
    _ensure_predictions_table(engine)


def insert_prediction(
    url: str,
    prediction: int,
    score: float,
    threshold: float,
    model_name: str,
) -> bool:
    """Insert a prediction into the database. Returns True if successful."""
    try:
        engine = get_db_engine()
        _ensure_predictions_table(engine)
        is_sqlite = engine.dialect.name == "sqlite"
        table_name = "predictions" if is_sqlite else f"{_safe_schema_name(DB_SCHEMA)}.predictions"

        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {table_name} (url, prediction, score, threshold, model_name)
                    VALUES (:url, :prediction, :score, :threshold, :model_name)
                """),
                {
                    "url": url,
                    "prediction": prediction,
                    "score": score,
                    "threshold": threshold,
                    "model_name": model_name,
                },
            )
        return True
    except Exception as e:
        print(f"Database insert failed: {e}")
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
    """Insert a training run summary and per-model metrics snapshot."""
    try:
        engine = get_db_engine()
        _ensure_predictions_table(engine)
        is_sqlite = engine.dialect.name == "sqlite"
        table_name = "training_runs" if is_sqlite else f"{_safe_schema_name(DB_SCHEMA)}.training_runs"
        cast_expr = ":metrics_json" if is_sqlite else "CAST(:metrics_json AS JSONB)"

        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {table_name} (
                        model_name,
                        best_threshold,
                        best_f1,
                        best_accuracy,
                        rows_count,
                        feature_count,
                        metrics_json
                    )
                    VALUES (
                        :model_name,
                        :best_threshold,
                        :best_f1,
                        :best_accuracy,
                        :rows_count,
                        :feature_count,
                        {cast_expr}
                    )
                """),
                {
                    "model_name": model_name,
                    "best_threshold": best_threshold,
                    "best_f1": best_f1,
                    "best_accuracy": best_accuracy,
                    "rows_count": rows_count,
                    "feature_count": feature_count,
                    "metrics_json": json.dumps(metrics),
                },
            )
        return True
    except Exception as e:
        print(f"Database training-run insert failed: {e}")
        return False


def get_recent_predictions(limit: int = 20) -> list[dict]:
    """Fetch the latest predictions from the database."""
    try:
        engine = get_db_engine()
        _ensure_predictions_table(engine)
        is_sqlite = engine.dialect.name == "sqlite"
        table_name = "predictions" if is_sqlite else f"{_safe_schema_name(DB_SCHEMA)}.predictions"

        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT id, url, prediction, score, threshold, model_name, created_at
                    FROM {table_name}
                    ORDER BY id DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            predictions = []
            for row in result.mappings():
                pred_dict = dict(row)
                if pred_dict.get("created_at"):
                    # Handle string format if it is SQLite datetime or convert timestamp
                    if hasattr(pred_dict["created_at"], "isoformat"):
                        pred_dict["created_at"] = pred_dict["created_at"].isoformat()
                    else:
                        pred_dict["created_at"] = str(pred_dict["created_at"])
                predictions.append(pred_dict)
            return predictions
    except Exception as e:
        print(f"Database query failed for recent predictions: {e}")
        return []


def get_recent_training_runs(limit: int = 10) -> list[dict]:
    """Fetch the latest training runs from the database."""
    try:
        engine = get_db_engine()
        _ensure_predictions_table(engine)
        is_sqlite = engine.dialect.name == "sqlite"
        table_name = "training_runs" if is_sqlite else f"{_safe_schema_name(DB_SCHEMA)}.training_runs"

        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    SELECT id, model_name, best_threshold, best_f1, best_accuracy, rows_count, feature_count, metrics_json, created_at
                    FROM {table_name}
                    ORDER BY id DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            runs = []
            for row in result.mappings():
                run_dict = dict(row)
                if isinstance(run_dict.get("metrics_json"), str):
                    try:
                        run_dict["metrics_json"] = json.loads(run_dict["metrics_json"])
                    except Exception:
                        pass
                if run_dict.get("created_at"):
                    if hasattr(run_dict["created_at"], "isoformat"):
                        run_dict["created_at"] = run_dict["created_at"].isoformat()
                    else:
                        run_dict["created_at"] = str(run_dict["created_at"])
                runs.append(run_dict)
            return runs
    except Exception as e:
        print(f"Database query failed for training runs: {e}")
        return []