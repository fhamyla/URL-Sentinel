"""Database connection and models for phishing URL predictions."""

from __future__ import annotations

import json
import os
import re
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_SCHEMA = os.getenv("DB_SCHEMA", "sentinel")


def _safe_schema_name(schema_name: str) -> str:
    """Return a validated SQL identifier for schema usage."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema_name):
        raise ValueError(f"Invalid DB_SCHEMA: {schema_name!r}")
    return schema_name


def _ensure_predictions_table(engine) -> None:
    """Create predictions and training tables if they do not exist."""
    schema = _safe_schema_name(DB_SCHEMA)
    with engine.begin() as conn:
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
    if not DATABASE_URL:
        return
    engine = create_engine(DATABASE_URL)
    _ensure_predictions_table(engine)


def insert_prediction(
    url: str,
    prediction: int,
    score: float,
    threshold: float,
    model_name: str,
) -> bool:
    """Insert a prediction into the database. Returns True if successful."""
    if not DATABASE_URL:
        return False
    try:
        engine = create_engine(DATABASE_URL)
        _ensure_predictions_table(engine)
        schema = _safe_schema_name(DB_SCHEMA)
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.predictions (url, prediction, score, threshold, model_name)
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
    if not DATABASE_URL:
        return False
    try:
        engine = create_engine(DATABASE_URL)
        _ensure_predictions_table(engine)
        schema = _safe_schema_name(DB_SCHEMA)
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.training_runs (
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
                        CAST(:metrics_json AS JSONB)
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