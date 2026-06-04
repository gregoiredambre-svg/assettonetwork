"""Similarity-factor ablation and cluster-size sweep using existing graph artifacts."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graph_construction as gc
import graph_model_temporal as temporal_model
import part1_extensions as ext_model
from evaluation import dataframe_to_markdown, save_metrics_table

GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"
SEED = 42

FACTOR_COMBOS: list[list[str]] = []
ALL_FACTORS = ["traffic", "climate", "pavement"]
for size in range(0, len(ALL_FACTORS) + 1):
    for combo in combinations(ALL_FACTORS, size):
        FACTOR_COMBOS.append(list(combo))

CLUSTER_SIZES = [1, 5, 10, 20, 50]


def log(msg: str) -> None:
    """Emit a standard ablation log line."""

    print(f"[run_ablation] {msg}")


def build_trimmed_split_masks(split_masks: dict[str, np.ndarray], keep_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Trim year-by-node split masks to the surviving node subset."""

    return {name: mask[:, keep_mask] for name, mask in split_masks.items()}


def train_and_eval(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, label: str) -> dict[str, dict[str, float]]:
    """Build relation adjacencies from an in-memory edge table and run temporal R-GCN."""

    temporal_model.set_seed(SEED)
    ext_model.set_seed(SEED)
    data, _ = temporal_model.prepare_temporal_data("full_refined", treatment_mode="experiment")
    surviving = set(nodes_df["node_id"].astype(str))
    keep_mask = np.array([node_id in surviving for node_id in data.node_ids], dtype=bool)
    if keep_mask.sum() == 0:
        raise ValueError(f"{label}: no temporal nodes survive after filtering.")

    kept_node_ids = [node_id for node_id, keep in zip(data.node_ids, keep_mask) if keep]
    relation_names, relation_adjs = ext_model.build_relation_adjacencies(
        kept_node_ids,
        "full_refined",
        edges_df=edges_df,
    )
    if not relation_names:
        raise ValueError(f"{label}: no relation adjacencies could be built.")

    split_masks = ext_model.build_year_split_masks(data)
    if keep_mask.sum() < len(data.node_ids):
        x = data.x_with_maint[:, keep_mask, :]
        y = data.y[:, keep_mask]
        split_masks = build_trimmed_split_masks(split_masks, keep_mask)
    else:
        x = data.x_with_maint
        y = data.y

    _, metrics = ext_model.train_temporal_rgcn(x, y, split_masks, relation_adjs)
    log(
        f"{label}: train R²={metrics['train']['r2']:.3f}  "
        f"val R²={metrics['val']['r2']:.3f}  test R²={metrics['test']['r2']:.3f}"
    )
    return metrics


def run_similarity_ablation(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> list[dict]:
    """Evaluate which similarity factors matter most for R-GCN performance."""

    rows = []
    for combo in FACTOR_COMBOS:
        label = "+".join(["spatial"] + list(combo)) if combo else "spatial_only"
        log(f"--- Similarity combo: {label} ---")
        edges_re = gc.recompute_edge_weights(edges_df, combo)
        comp_summary = gc.summarise_components(nodes_df, edges_re)
        metrics = train_and_eval(nodes_df, edges_re, label)
        rows.append(
            {
                "combo": label,
                "factors": combo,
                "n_nodes": int(len(nodes_df)),
                "n_edges": int(len(edges_re)),
                "n_components": comp_summary["n_components"],
                "max_component": comp_summary["max_size"],
                "mean_component": comp_summary["mean_size"],
                "r2_train": metrics["train"]["r2"],
                "r2_val": metrics["val"]["r2"],
                "r2_test": metrics["test"]["r2"],
                "mae_test": metrics["test"]["mae"],
                "rmse_test": metrics["test"]["rmse"],
                "smape_test": metrics["test"]["smape"],
                "n_test": metrics["test"]["n_samples"],
            }
        )
    return rows


def run_cluster_size_sweep(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> list[dict]:
    """Evaluate the effect of removing small connected components."""

    rows = []
    edges_re = gc.recompute_edge_weights(edges_df, ["traffic", "climate", "pavement"])
    for min_size in CLUSTER_SIZES:
        label = f"min_size={min_size}"
        log(f"--- Cluster sweep: {label} ---")
        kept_nodes, kept_edges, info = gc.filter_by_cluster_size(nodes_df, edges_re, min_size)
        comp_summary = gc.summarise_components(kept_nodes, kept_edges)
        if len(kept_nodes) < 50 or len(kept_edges) < 100:
            log(f"  skipped (too few nodes/edges left): n_nodes={len(kept_nodes)} n_edges={len(kept_edges)}")
            rows.append(
                {
                    "min_cluster_size": min_size,
                    "n_nodes_kept": int(len(kept_nodes)),
                    "n_edges_kept": int(len(kept_edges)),
                    "n_components": comp_summary["n_components"],
                    "r2_test": None,
                    "mae_test": None,
                    "rmse_test": None,
                    "smape_test": None,
                    "n_test": 0,
                    "skipped": True,
                    "drop_info": info,
                }
            )
            continue
        metrics = train_and_eval(kept_nodes, kept_edges, label)
        rows.append(
            {
                "min_cluster_size": min_size,
                "n_nodes_kept": int(len(kept_nodes)),
                "n_edges_kept": int(len(kept_edges)),
                "n_components": comp_summary["n_components"],
                "r2_train": metrics["train"]["r2"],
                "r2_val": metrics["val"]["r2"],
                "r2_test": metrics["test"]["r2"],
                "mae_test": metrics["test"]["mae"],
                "rmse_test": metrics["test"]["rmse"],
                "smape_test": metrics["test"]["smape"],
                "n_test": metrics["test"]["n_samples"],
                "skipped": False,
                "drop_info": info,
            }
        )
    return rows


def main() -> None:
    """Run both ablation studies and save JSON/markdown outputs."""

    log("Loading base nodes and edges from graph_data/...")
    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    edges_df = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    log(f"Loaded {len(nodes_df)} nodes, {len(edges_df)} edges.")

    log("Running similarity-factor ablation...")
    sim_rows = run_similarity_ablation(nodes_df, edges_df)

    log("Running cluster-size sweep...")
    clu_rows = run_cluster_size_sweep(nodes_df, edges_df)

    GRAPH_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    (GRAPH_DIR / "ablation_similarity.json").write_text(json.dumps(sim_rows, indent=2), encoding="utf-8")
    (GRAPH_DIR / "ablation_cluster_size.json").write_text(json.dumps(clu_rows, indent=2), encoding="utf-8")

    sim_df = pd.DataFrame(sim_rows)
    clu_df = pd.DataFrame(clu_rows)
    save_metrics_table(sim_df, REPORT_DIR / "ablation_similarity_table.json", REPORT_DIR / "ablation_similarity.md")
    save_metrics_table(clu_df, REPORT_DIR / "ablation_cluster_size_table.json", REPORT_DIR / "ablation_cluster_size.md")

    print("\n## Similarity ablation (R-GCN, full_refined edge types, varying weight composition)")
    print(dataframe_to_markdown(sim_df))
    print("\n## Cluster-size sweep (comprehensive similarity weights)")
    print(dataframe_to_markdown(clu_df))

    best_combo = sim_df.sort_values("r2_test", ascending=False).iloc[0]
    print(f"\nBest similarity combo: {best_combo['combo']} (test R²={best_combo['r2_test']:.4f})")
    valid_clu = clu_df[clu_df["r2_test"].notna()].sort_values("r2_test", ascending=False)
    if not valid_clu.empty:
        best_clu = valid_clu.iloc[0]
        print(f"Best cluster threshold: min_size={int(best_clu['min_cluster_size'])} (test R²={best_clu['r2_test']:.4f})")


if __name__ == "__main__":
    main()
