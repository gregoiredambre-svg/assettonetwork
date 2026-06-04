"""Compare full_refined with and without the same-state barrier on functional edges."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graph_construction as gc
from evaluation import dataframe_to_markdown
from presentation_model_utils import load_target_data, train_single_target_rgcn

REPORTS_DIR = ROOT / "reports"
TARGET = "HPMS16_CRACKING_PERCENT_AC"
GRAPH_VARIANT = "full_refined"


def log(message: str) -> None:
    print(f"[state_barrier_ablation] {message}")


def prepare_nodes_once() -> pd.DataFrame:
    nodes = gc.prepare_node_table()
    climate = gc.load_climate_features()
    distress = gc.load_distress_features()
    traffic = gc.load_traffic_features()
    materials = gc.load_section_materials()
    nodes = nodes.merge(climate, on="node_id_join", how="left")
    if "merra_grid_elevation" in nodes.columns and "elevation" in nodes.columns:
        nodes["elevation"] = pd.to_numeric(nodes["elevation"], errors="coerce").fillna(nodes["merra_grid_elevation"])
    nodes = nodes.merge(distress, on="node_id_join", how="left")
    nodes = nodes.merge(traffic, on="node_id_join", how="left")
    if not materials.empty:
        nodes = nodes.merge(materials, on="node_id_join", how="left")
    return nodes


def build_edges(nodes: pd.DataFrame, same_state_only: bool) -> pd.DataFrame:
    spatial_edges = gc.build_spatial_edges(nodes)
    route_edges = gc.build_route_edges(nodes)
    functional_edges = gc.build_functional_edges(nodes, same_state_only=same_state_only)
    edges = pd.concat([spatial_edges, route_edges, functional_edges], ignore_index=True, sort=False)
    edges = edges.drop_duplicates(subset=["source", "target", "edge_type"])
    edges = gc.augment_edges(edges, nodes)
    edges = gc.augment_diversion(edges, nodes)
    edges = gc.add_edge_weight_views(edges)
    return edges


def component_stats(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, float | int]:
    summary = gc.summarise_components(nodes, edges)
    return {
        "connected_components": int(summary["connected_components"]),
        "largest_component_size": int(summary["largest_component_size"]),
        "largest_component_share_pct": float(summary["largest_component_share_pct"]),
        "isolated_nodes": int(summary["isolated_nodes"]),
    }


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    nodes = prepare_nodes_once()
    data, y_full, split_masks = load_target_data(TARGET, graph_variant=GRAPH_VARIANT, treatment_mode="experiment")

    rows: list[dict[str, object]] = []
    for label, same_state_only in [
        ("full_refined_same_state", True),
        ("full_refined_cross_state_functional", False),
    ]:
        log(f"Running {label}")
        edges = build_edges(nodes, same_state_only=same_state_only)
        functional_edges = edges[edges["edge_type"] == "same_functional_class"].copy()
        cross_state_count = 0
        if {"source_state", "target_state"}.issubset(functional_edges.columns):
            cross_state_count = int((functional_edges["source_state"].astype(str) != functional_edges["target_state"].astype(str)).sum())
        _, metrics, _ = train_single_target_rgcn(
            data=data,
            target=TARGET,
            y_full=y_full,
            split_masks=split_masks,
            graph_variant=GRAPH_VARIANT,
            edges_df=edges,
            max_epochs=180,
            patience=20,
        )
        rows.append(
            {
                "label": label,
                "same_state_only": same_state_only,
                "n_edges_total": int(len(edges)),
                "n_edges_functional": int(len(functional_edges)),
                "n_cross_state_functional_edges": cross_state_count,
                **component_stats(nodes, edges),
                "r2_train": float(metrics["train"]["r2"]),
                "r2_val": float(metrics["val"]["r2"]),
                "r2_test": float(metrics["test"]["r2"]),
                "mae_test": float(metrics["test"]["mae"]),
                "rmse_test": float(metrics["test"]["rmse"]),
                "n_test": int(metrics["test"]["n_samples"]),
            }
        )

    df = pd.DataFrame(rows).sort_values("r2_test", ascending=False).reset_index(drop=True)
    summary = {
        "target": TARGET,
        "graph_variant": GRAPH_VARIANT,
        "results": rows,
        "best_label": str(df.iloc[0]["label"]) if not df.empty else None,
        "delta_r2_cross_state_minus_same_state": (
            float(df.set_index("label").loc["full_refined_cross_state_functional", "r2_test"])
            - float(df.set_index("label").loc["full_refined_same_state", "r2_test"])
        )
        if len(df) == 2
        else None,
    }
    (REPORTS_DIR / "state_barrier_ablation.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (REPORTS_DIR / "state_barrier_ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS_DIR / "state_barrier_ablation.md").write_text(dataframe_to_markdown(df), encoding="utf-8")
    print(dataframe_to_markdown(df))


if __name__ == "__main__":
    main()
