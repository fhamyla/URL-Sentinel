# Database Usage Guide

This project stores two types of records in PostgreSQL schema `sentinel` by default.

## Tables

- `sentinel.predictions`
  - One row per URL prediction made by `predict_url.py`
  - Stores: URL, predicted class, score/probability, threshold used, model name, timestamp

- `sentinel.training_runs`
  - One row per `train_model.py` execution
  - Stores: selected best model, best threshold, best F1/accuracy, dataset row/feature counts, full per-model metrics JSON, timestamp

## Why store this data

- Prediction history: monitor real usage, confidence drift, and suspicious spikes.
- Training history: compare experiments over time and keep a reproducible audit trail.
- Metrics JSON: retain full benchmark details without creating many extra tables.

## Suggested production additions

- Add `request_id`, `source`, or `user_id` (if available) to `predictions`.
- Add model artifact hash/version to `training_runs` for strict reproducibility.
- Add data snapshot ID (dataset version/date) to `training_runs`.

## Quick queries

Top recent predictions:

```sql
SELECT id, url, prediction, score, threshold, model_name, created_at
FROM sentinel.predictions
ORDER BY id DESC
LIMIT 20;
```

Latest training runs:

```sql
SELECT id, model_name, best_f1, best_accuracy, rows_count, feature_count, created_at
FROM sentinel.training_runs
ORDER BY id DESC
LIMIT 10;
```

Compare average best F1 per model:

```sql
SELECT model_name, ROUND(AVG(best_f1)::numeric, 4) AS avg_best_f1, COUNT(*) AS runs
FROM sentinel.training_runs
GROUP BY model_name
ORDER BY avg_best_f1 DESC;
```
