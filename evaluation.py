"""Shared evaluation helpers for the dissertation modelling pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAPE_EPSILON = 1e-3
SMAPE_EPSILON = 1e-9


def compute_smape(y_true, y_pred) -> float:
    """Compute symmetric mean absolute percentage error in percent."""

    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        return math.nan
    return float(np.mean(2.0 * np.abs(true - pred) / (np.abs(true) + np.abs(pred) + SMAPE_EPSILON)) * 100.0)


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=2000, ci=0.95, seed=42) -> dict:
    """Bootstrap a confidence interval for a regression metric."""

    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")

    point = float(metric_fn(true, pred))
    if true.size < 2:
        return {"point": point, "lo": math.nan, "hi": math.nan, "se": math.nan}

    rng = np.random.default_rng(seed)
    n = true.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(metric_fn(true[idx], pred[idx]))

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "point": point,
        "lo": float(np.quantile(boots, lo_q)),
        "hi": float(np.quantile(boots, hi_q)),
        "se": float(np.std(boots, ddof=1)),
    }


def bootstrap_paired_delta_ci(y_true, y_pred_a, y_pred_b, metric_fn, n_boot=2000, ci=0.95, seed=42) -> dict:
    """Bootstrap a paired confidence interval for metric(model A) - metric(model B)."""

    true = np.asarray(y_true, dtype=float).reshape(-1)
    a = np.asarray(y_pred_a, dtype=float).reshape(-1)
    b = np.asarray(y_pred_b, dtype=float).reshape(-1)
    if true.shape != a.shape or true.shape != b.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_pred_a {a.shape}, y_pred_b {b.shape}")

    point = float(metric_fn(true, a) - metric_fn(true, b))
    if true.size < 2:
        return {"point": point, "lo": math.nan, "hi": math.nan, "p_positive": math.nan}

    rng = np.random.default_rng(seed)
    n = true.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(metric_fn(true[idx], a[idx])) - float(metric_fn(true[idx], b[idx]))

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "point": point,
        "lo": float(np.quantile(boots, lo_q)),
        "hi": float(np.quantile(boots, hi_q)),
        "p_positive": float(np.mean(boots > 0)),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common regression metrics with guarded percentage errors.

    Parameters
    ----------
    y_true:
        Observed target values.
    y_pred:
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary containing r2, mae, rmse, smape, mape, and n_samples.
    """

    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        return {"r2": math.nan, "mae": math.nan, "rmse": math.nan, "smape": math.nan, "mape": math.nan, "n_samples": 0}

    denom = np.maximum(np.abs(true), MAPE_EPSILON)
    mape = float(np.mean(np.abs((true - pred) / denom)) * 100.0)
    return {
        "r2": float(r2_score(true, pred)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(true, pred))),
        "smape": compute_smape(true, pred),
        "mape": mape,
        "n_samples": int(true.size),
    }


def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a tidy comparison table from nested train/val/test metric dicts."""

    rows: list[dict[str, Any]] = []
    for model_name, splits in results.items():
        train = splits.get("train", {}) or {}
        val = splits.get("val", {}) or {}
        test = splits.get("test", {}) or {}
        rows.append(
            {
                "model": model_name,
                "r2_train": train.get("r2"),
                "r2_val": val.get("r2"),
                "r2_test": test.get("r2"),
                "mae_test": test.get("mae"),
                "rmse_test": test.get("rmse"),
                "smape_test": test.get("smape"),
                "mape_test": test.get("mape"),
                "n_test": test.get("n_samples"),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a dataframe as a simple markdown table without extra deps."""

    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(
                lambda value: ""
                if pd.isna(value)
                else (f"{float(value):.6f}" if isinstance(value, (float, np.floating)) else str(int(value)))
            )
        else:
            display[col] = display[col].fillna("").astype(str)

    headers = [str(col) for col in display.columns]
    align = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def save_metrics_table(df: pd.DataFrame, path_json: Path, path_md: Path) -> None:
    """Save a comparison table as JSON and markdown."""

    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_md.parent.mkdir(parents=True, exist_ok=True)

    records = df.to_dict(orient="records")
    with open(path_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    path_md.write_text(dataframe_to_markdown(df), encoding="utf-8")
