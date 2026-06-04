"""Representative leave-one-state-out extension for the RF + R-GCN ensemble."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import part1_extensions as ext_model
from presentation_model_utils import (
    fit_rf_predictions,
    fit_stacked_mlp_with_data,
    load_target_data,
    train_single_target_rgcn,
)

REPORTS_DIR = ROOT / "reports"
GRAPH_DIR = ROOT / "graph_data"
TARGET = "HPMS16_CRACKING_PERCENT_AC"
GRAPH_VARIANT = "full_refined"
MAX_STATES = 5
OUT_CSV = REPORTS_DIR / "part1_ood_ensemble.csv"
OUT_JSON = REPORTS_DIR / "part1_ood_ensemble_summary.json"


def log(message: str) -> None:
    print(f"[ood_ensemble] {message}")


def choose_states(rows_with_state: pd.DataFrame, max_states: int = MAX_STATES) -> list[str]:
    counts = rows_with_state["state_code"].astype(str).value_counts().sort_values(ascending=False)
    return counts.head(max_states).index.astype(str).tolist()


def build_masks_for_state(data, target_rows_with_state: pd.DataFrame, held_out_state: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    counts = target_rows_with_state[target_rows_with_state["state_code"].astype(str) != held_out_state]["state_code"].astype(str).value_counts()
    val_states, _ = ext_model.greedy_holdout_groups(counts, val_share=0.15, test_share=0.0)
    train_states = set(counts.index.astype(str)) - val_states

    split_masks = {
        "train": np.zeros((len(data.years), len(data.node_ids)), dtype=bool),
        "val": np.zeros((len(data.years), len(data.node_ids)), dtype=bool),
        "test": np.zeros((len(data.years), len(data.node_ids)), dtype=bool),
    }
    year_index = {year: idx for idx, year in enumerate(data.years)}
    node_index = {node_id: idx for idx, node_id in enumerate(data.node_ids)}

    for row in target_rows_with_state.itertuples(index=False):
        yi = year_index[int(row.YEAR)]
        ni = node_index[str(row.node_id)]
        state = str(row.state_code)
        if state == held_out_state:
            split_masks["test"][yi, ni] = True
        elif state in val_states:
            split_masks["val"][yi, ni] = True
        elif state in train_states:
            split_masks["train"][yi, ni] = True

    info = {
        "held_out_state": held_out_state,
        "train_states": sorted(train_states),
        "val_states": sorted(val_states),
        "test_states": [held_out_state],
        "transition_counts": {name: int(mask.sum()) for name, mask in split_masks.items()},
    }
    return split_masks, info


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    data, y_full, _ = load_target_data(TARGET, graph_variant=GRAPH_VARIANT, treatment_mode="experiment")
    node_meta = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)[["node_id", "state_code"]].copy()
    rows = data.panel.loc[data.panel["target_t1"].notna(), ["node_id", "YEAR"]].merge(node_meta, on="node_id", how="left")
    states = choose_states(rows, max_states=MAX_STATES)
    log(f"Evaluating representative held-out states: {states}")

    outputs: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for state in states:
        log(f"Running held-out state {state}")
        split_masks, split_info = build_masks_for_state(data, rows, state)
        if split_info["transition_counts"]["train"] < 500 or split_info["transition_counts"]["val"] < 50 or split_info["transition_counts"]["test"] < 25:
            skipped.append({"state": state, **split_info})
            log(f"Skipping state {state} due to low transition counts: {split_info['transition_counts']}")
            continue

        rgcn_rows, rgcn_metrics, _ = train_single_target_rgcn(
            data=data,
            target=TARGET,
            y_full=y_full,
            split_masks=split_masks,
            graph_variant=GRAPH_VARIANT,
            edges_df=None,
            max_epochs=120,
            patience=15,
        )
        rf_rows, rf_metrics, _ = fit_rf_predictions(rgcn_rows.copy(), data.local_feature_cols)
        merged = rgcn_rows.merge(
            rf_rows[["node_id", "YEAR", "split", "year_order", "node_order", "rf_pred"]],
            on=["node_id", "YEAR", "split", "year_order", "node_order"],
            how="left",
        )
        ensemble_rows, ensemble_metrics = fit_stacked_mlp_with_data(data, merged)
        test_rows = ensemble_rows[ensemble_rows["split"] == "test"].copy()
        outputs.append(
            {
                "state_held_out": state,
                "n_test_nodes": int(test_rows["node_id"].nunique()),
                "n_test_transitions": int(len(test_rows)),
                "rf_test_r2": float(rf_metrics["test"]["r2"]),
                "rgcn_test_r2": float(rgcn_metrics["test"]["r2"]),
                "ensemble_test_r2": float(ensemble_metrics["test"]["r2"]),
                "rf_test_mae": float(rf_metrics["test"]["mae"]),
                "ensemble_test_mae": float(ensemble_metrics["test"]["mae"]),
            }
        )

    df = pd.DataFrame(outputs)
    df.to_csv(OUT_CSV, index=False)
    summary = {
        "target": TARGET,
        "graph_variant": GRAPH_VARIANT,
        "subset_mode": True,
        "states_requested": states,
        "states_evaluated": df["state_held_out"].tolist() if not df.empty else [],
        "n_states_evaluated": int(len(df)),
        "skipped_states": skipped,
        "rf_r2": summarize(df["rf_test_r2"].tolist()) if not df.empty else {},
        "rgcn_r2": summarize(df["rgcn_test_r2"].tolist()) if not df.empty else {},
        "ensemble_r2": summarize(df["ensemble_test_r2"].tolist()) if not df.empty else {},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
