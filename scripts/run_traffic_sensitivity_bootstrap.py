"""Bootstrap confidence intervals for traffic sensitivity analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
EVENTS_PATH = REPORTS_DIR / "traffic_sensitivity_events.csv"
CONTROLS_PATH = REPORTS_DIR / "traffic_sensitivity_controls.csv"
SUMMARY_PATH = REPORTS_DIR / "traffic_sensitivity.json"
OUT_PATH = REPORTS_DIR / "traffic_sensitivity_bootstrap.json"
N_BOOT = 10_000
SEED = 42


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float, np.ndarray]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    point = float(values.mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), draws


def main() -> None:
    events = pd.read_csv(EVENTS_PATH)
    controls = pd.read_csv(CONTROLS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    event_values = pd.to_numeric(events["delta_pre_to_post"], errors="coerce").dropna().to_numpy(dtype=float)
    control_values = pd.to_numeric(controls["delta_pre_to_post"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(event_values) == 0 or len(control_values) == 0:
        raise ValueError("Traffic sensitivity CSVs do not contain usable delta_pre_to_post values.")

    rng = np.random.default_rng(SEED)
    event_mean, event_lo, event_hi, event_draws = bootstrap_mean_ci(event_values, rng, N_BOOT)
    control_mean, control_lo, control_hi, control_draws = bootstrap_mean_ci(control_values, rng, N_BOOT)
    diff_draws = event_draws - control_draws
    diff_mean = float(event_mean - control_mean)
    diff_lo, diff_hi = np.percentile(diff_draws, [2.5, 97.5])

    payload = {
        "traffic_metric_column": summary.get("traffic_metric_column"),
        "traffic_metric_kind": summary.get("traffic_metric_kind"),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "n_events": int(len(event_values)),
        "n_controls": int(len(control_values)),
        "event_mean": event_mean,
        "event_ci_low": event_lo,
        "event_ci_high": event_hi,
        "control_mean": control_mean,
        "control_ci_low": control_lo,
        "control_ci_high": control_hi,
        "diff_mean": diff_mean,
        "diff_ci_low": float(diff_lo),
        "diff_ci_high": float(diff_hi),
        "welch_p_value": summary.get("welch_p_value_events_vs_controls"),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
