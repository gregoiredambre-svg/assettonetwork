"""Apples-to-apples benchmark on MEPDG cracking: RF vs R-GCN vs ensemble."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presentation_model_utils import (
    fit_rf_predictions,
    fit_stacked_mlp_with_data,
    load_target_data,
    predict_single_target_rgcn_from_checkpoint,
)

GRAPH_DIR = ROOT / "graph_data"
REPORTS_DIR = ROOT / "reports"
TARGET = "MEPDG_CRACKING_PERCENT_AC"
BASELINE_CKPT = GRAPH_DIR / "temporal_rgcn_singletask_MEPDG_CRACKING_PERCENT_AC.pt"
MATERIALS_SWEEP = REPORTS_DIR / "materials_weight_sweep.json"
OUT_CSV = REPORTS_DIR / "mepdg_benchmark.csv"
OUT_JSON = REPORTS_DIR / "mepdg_benchmark.json"


def log(message: str) -> None:
    print(f"[mepdg_benchmark] {message}")


def main() -> None:
    data, _, split_masks = load_target_data(TARGET, graph_variant="full_refined", treatment_mode="experiment")

    log("Loading baseline R-GCN checkpoint and generating predictions ...")
    rgcn_rows, rgcn_metrics, _ = predict_single_target_rgcn_from_checkpoint(
        data=data,
        target=TARGET,
        checkpoint_path=BASELINE_CKPT,
        split_masks=split_masks,
        graph_variant="full_refined",
    )

    log("Training RF local on the same target split ...")
    rf_rows, rf_metrics, _ = fit_rf_predictions(rgcn_rows.copy(), data.local_feature_cols)

    merged = rgcn_rows[["node_id", "YEAR", "split", "year_order", "node_order", "target_value", "rgcn_pred"]].merge(
        rf_rows[["node_id", "YEAR", "split", "year_order", "node_order", "rf_pred"]],
        on=["node_id", "YEAR", "split", "year_order", "node_order"],
        how="inner",
    )
    merged = merged.merge(
        data.panel.drop(columns=[col for col in data.panel.columns if col.endswith("_t1") and col != f"{TARGET}_t1"]),
        on=["node_id", "YEAR"],
        how="left",
    )

    log("Fitting stacked MLP ensemble on validation predictions ...")
    ensemble_rows, ensemble_metrics = fit_stacked_mlp_with_data(data, merged)

    rows = [
        {
            "model": "RF local",
            "r2_train": rf_metrics["train"]["r2"],
            "r2_test": rf_metrics["test"]["r2"],
            "mae_test": rf_metrics["test"]["mae"],
            "rmse_test": rf_metrics["test"]["rmse"],
            "notes": "RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=3); standard temporal split.",
        },
        {
            "model": "R-GCN baseline",
            "r2_train": rgcn_metrics["train"]["r2"],
            "r2_test": rgcn_metrics["test"]["r2"],
            "mae_test": rgcn_metrics["test"]["mae"],
            "rmse_test": rgcn_metrics["test"]["rmse"],
            "notes": "Saved single-task checkpoint on full_refined without materials sweep.",
        },
        {
            "model": "Stacked MLP ensemble",
            "r2_train": ensemble_metrics["train"]["r2"],
            "r2_test": ensemble_metrics["test"]["r2"],
            "mae_test": ensemble_metrics["test"]["mae"],
            "rmse_test": ensemble_metrics["test"]["rmse"],
            "notes": "StackedEnsemble(meta_model=mlp) fit on validation predictions from RF + baseline R-GCN.",
        },
    ]

    if MATERIALS_SWEEP.exists():
        payload = json.loads(MATERIALS_SWEEP.read_text(encoding="utf-8"))
        best_label = payload.get("best_label")
        best_r2 = payload.get("best_test_r2")
        best_result = next((row for row in payload.get("results", []) if row.get("label") == best_label), None)
        if best_result is not None:
            rows.append(
                {
                    "model": f"R-GCN best materials sweep ({best_label})",
                    "r2_train": best_result.get("r2_train"),
                    "r2_test": best_result.get("r2_test"),
                    "mae_test": best_result.get("mae_test"),
                    "rmse_test": best_result.get("rmse_test"),
                    "notes": "Reference only; loaded from reports/materials_weight_sweep.json.",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({"target": TARGET, "results": rows}, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
