"""Database connection and models for phishing URL predictions."""

from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")


def create_tables() -> None:
    """Create all tables if they don't exist."""
    if not DATABASE_URL:
        return
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS predictions (
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
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO predictions (url, prediction, score, threshold, model_name)
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