"""Targeted OOD ablations for edge weighting in the temporal R-GCN / ensemble stack."""

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
OUT_CSV = REPORTS_DIR / "edge_weight_ood_ablation.csv"
OUT_JSON = REPORTS_DIR / "edge_weight_ood_ablation_summary.json"
OUT_MD = REPORTS_DIR / "edge_weight_ood_ablation.md"


def log(message: str) -> None:
    print(f"[edge_weight_ood] {message}")


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


def build_edge_variants(edges: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    variants: dict[str, pd.DataFrame | None] = {"baseline_mixed_current": None}

    # Variant 1: same deterioration formula applied consistently to all relations.
    full_formula = edges.copy()
    spatial = pd.to_numeric(full_formula.get("spatial_score"), errors="coerce").fillna(0.0)
    route = pd.to_numeric(full_formula.get("route_score"), errors="coerce").fillna(0.0)
    traffic = pd.to_numeric(full_formula.get("traffic_similarity"), errors="coerce").fillna(0.5)
    climate = pd.to_numeric(full_formula.get("climate_similarity"), errors="coerce").fillna(0.5)
    pavement = pd.to_numeric(full_formula.get("pavement_similarity"), errors="coerce").fillna(0.5)
    full_formula["weight_deterioration"] = (
        0.35 * spatial
        + 0.15 * route
        + 0.20 * traffic
        + 0.20 * climate
        + 0.10 * pavement
    ).astype(float)
    variants["unified_full_formula"] = full_formula

    # Variant 2: simplified distance-only weighting, removing the 5.0 closeness multiplier entirely.
    distance_only = edges.copy()
    distance_km = pd.to_numeric(distance_only.get("distance_km"), errors="coerce").fillna(80.0)
    distance_only["weight_deterioration"] = (1.0 / (1.0 + distance_km / 80.0)).astype(float)
    variants["distance_only"] = distance_only

    return variants


def write_markdown(df: pd.DataFrame, summary: dict[str, object]) -> None:
    columns = list(df.columns)
    lines = [
        "# Edge Weight OOD Ablation",
        "",
        f"- Target: `{summary['target']}`",
        f"- Graph variant: `{summary['graph_variant']}`",
        f"- States evaluated: `{', '.join(summary['states_evaluated'])}`",
        "",
        "## Mean OOD R² by variant",
        "",
        "| variant | rgcn_mean_r2 | ensemble_mean_r2 | rf_mean_r2 |",
        "| --- | --- | --- | --- |",
    ]
    for variant, payload in summary["variants"].items():
        rf_mean = payload["rf_r2"]["mean"] if payload["rf_r2"] else float("nan")
        rgcn_mean = payload["rgcn_r2"]["mean"] if payload["rgcn_r2"] else float("nan")
        ens_mean = payload["ensemble_r2"]["mean"] if payload["ensemble_r2"] else float("nan")
        lines.append(f"| {variant} | {rgcn_mean:.4f} | {ens_mean:.4f} | {rf_mean:.4f} |")
    lines.extend(
        [
            "",
            "## Per-state results",
            "",
        ]
    )
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in df.itertuples(index=False):
        values: list[str] = []
        for col in columns:
            val = getattr(row, col)
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data, y_full, _ = load_target_data(TARGET, graph_variant=GRAPH_VARIANT, treatment_mode="experiment")
    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    variants = build_edge_variants(edges)

    node_meta = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)[["node_id", "state_code"]].copy()
    rows = data.panel.loc[data.panel["target_t1"].notna(), ["node_id", "YEAR"]].merge(node_meta, on="node_id", how="left")
    states = choose_states(rows, max_states=MAX_STATES)
    log(f"Evaluating representative held-out states: {states}")

    outputs: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    summary_variants: dict[str, dict[str, object]] = {}

    for state in states:
        split_masks, split_info = build_masks_for_state(data, rows, state)
        if split_info["transition_counts"]["train"] < 500 or split_info["transition_counts"]["val"] < 50 or split_info["transition_counts"]["test"] < 25:
            skipped.append({"state": state, **split_info})
            log(f"Skipping state {state} due to low transition counts: {split_info['transition_counts']}")
            continue

        log(f"Training RF baseline for held-out state {state}")
        baseline_rows, _, _ = train_single_target_rgcn(
            data=data,
            target=TARGET,
            y_full=y_full,
            split_masks=split_masks,
            graph_variant=GRAPH_VARIANT,
            edges_df=None,
            max_epochs=1,
            patience=1,
        )
        # Reuse row structure only; actual graph metrics are recomputed per variant below.
        rf_rows, rf_metrics, _ = fit_rf_predictions(baseline_rows.copy(), data.local_feature_cols)

        for variant_name, edges_df in variants.items():
            log(f"Running {variant_name} on held-out state {state}")
            rgcn_rows, rgcn_metrics, _ = train_single_target_rgcn(
                data=data,
                target=TARGET,
                y_full=y_full,
                split_masks=split_masks,
                graph_variant=GRAPH_VARIANT,
                edges_df=edges_df,
                max_epochs=120,
                patience=15,
            )
            merged = rgcn_rows.merge(
                rf_rows[["node_id", "YEAR", "split", "year_order", "node_order", "rf_pred"]],
                on=["node_id", "YEAR", "split", "year_order", "node_order"],
                how="left",
            )
            ensemble_rows, ensemble_metrics = fit_stacked_mlp_with_data(data, merged)
            test_rows = ensemble_rows[ensemble_rows["split"] == "test"].copy()
            outputs.append(
                {
                    "variant": variant_name,
                    "state_held_out": state,
                    "n_test_nodes": int(test_rows["node_id"].nunique()),
                    "n_test_transitions": int(len(test_rows)),
                    "rf_test_r2": float(rf_metrics["test"]["r2"]),
                    "rgcn_test_r2": float(rgcn_metrics["test"]["r2"]),
                    "ensemble_test_r2": float(ensemble_metrics["test"]["r2"]),
                    "rf_test_mae": float(rf_metrics["test"]["mae"]),
                    "rgcn_test_mae": float(rgcn_metrics["test"]["mae"]),
                    "ensemble_test_mae": float(ensemble_metrics["test"]["mae"]),
                }
            )

    df = pd.DataFrame(outputs)
    df.to_csv(OUT_CSV, index=False)

    for variant_name in sorted(df["variant"].unique()) if not df.empty else []:
        subset = df[df["variant"] == variant_name].copy()
        summary_variants[variant_name] = {
            "n_states_evaluated": int(len(subset)),
            "rf_r2": summarize(subset["rf_test_r2"].tolist()),
            "rgcn_r2": summarize(subset["rgcn_test_r2"].tolist()),
            "ensemble_r2": summarize(subset["ensemble_test_r2"].tolist()),
            "rf_mae": summarize(subset["rf_test_mae"].tolist()),
            "rgcn_mae": summarize(subset["rgcn_test_mae"].tolist()),
            "ensemble_mae": summarize(subset["ensemble_test_mae"].tolist()),
        }

    summary = {
        "target": TARGET,
        "graph_variant": GRAPH_VARIANT,
        "subset_mode": True,
        "states_requested": states,
        "states_evaluated": sorted(df["state_held_out"].astype(str).unique().tolist()) if not df.empty else [],
        "n_states_evaluated": int(df["state_held_out"].nunique()) if not df.empty else 0,
        "variants_tested": list(variants.keys()),
        "skipped_states": skipped,
        "variants": summary_variants,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(df, summary)

    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
