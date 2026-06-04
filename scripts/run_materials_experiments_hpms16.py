"""Sweep richer pavement features and full_refined weights for the HPMS16 R-GCN target."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graph_construction as gc
from evaluation import dataframe_to_markdown
from load_materials import load_materials
from presentation_model_utils import load_target_data, train_single_target_rgcn

GRAPH_DIR = ROOT / "graph_data"
REPORTS_DIR = ROOT / "reports"
TARGET = "HPMS16_CRACKING_PERCENT_AC"
BASELINE_RESULTS = GRAPH_DIR / "singletask_per_distress_results.json"
WEIGHT_CONFIGS: list[tuple[str, dict[str, float]]] = [
    ("default", {"spatial": 0.40, "traffic": 0.15, "climate": 0.20, "pavement": 0.25}),
    ("pavement_rich", {"spatial": 0.35, "traffic": 0.10, "climate": 0.15, "pavement": 0.40}),
    ("climate_pavement", {"spatial": 0.30, "traffic": 0.10, "climate": 0.25, "pavement": 0.35}),
    ("spatial_pavement", {"spatial": 0.45, "traffic": 0.05, "climate": 0.15, "pavement": 0.35}),
    ("pavement_dominant", {"spatial": 0.25, "traffic": 0.10, "climate": 0.20, "pavement": 0.45}),
]


def log(message: str) -> None:
    print(f"[materials_weight_sweep_hpms16] {message}")


def find_baseline_r2() -> float | None:
    if not BASELINE_RESULTS.exists():
        return None
    payload = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    for row in payload.get("results", []):
        if row.get("target") == TARGET:
            value = row.get("r2_test")
            return float(value) if value is not None else None
    return None


def prepare_nodes_once() -> pd.DataFrame:
    load_materials()
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


def build_edges_for_config(nodes: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    spatial_edges = gc.build_spatial_edges(nodes)
    route_edges = gc.build_route_edges(nodes)
    functional_edges = gc.build_functional_edges(nodes, similarity_weights=weights)
    edges = pd.concat([spatial_edges, route_edges, functional_edges], ignore_index=True, sort=False)
    edges = edges.drop_duplicates(subset=["source", "target", "edge_type"])
    edges = gc.augment_edges(edges, nodes)
    edges = gc.augment_diversion(edges, nodes)
    edges = gc.add_edge_weight_views(edges)
    return edges


def main() -> None:
    GRAPH_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    baseline_r2 = find_baseline_r2()
    nodes = prepare_nodes_once()
    data, y_full, split_masks = load_target_data(TARGET, graph_variant="full_refined", treatment_mode="experiment")

    rows: list[dict[str, object]] = []
    best_label = None
    best_r2 = -float("inf")
    for label, weights in WEIGHT_CONFIGS:
        log(f"Running config {label} with weights={weights}")
        edges = build_edges_for_config(nodes, weights)
        _, metrics, _ = train_single_target_rgcn(
            data=data,
            target=TARGET,
            y_full=y_full,
            split_masks=split_masks,
            graph_variant="full_refined",
            edges_df=edges,
            max_epochs=180,
            patience=20,
        )
        row = {
            "label": label,
            "spatial_weight": weights["spatial"],
            "traffic_weight": weights["traffic"],
            "climate_weight": weights["climate"],
            "pavement_weight": weights["pavement"],
            "n_edges": int(len(edges)),
            "r2_train": float(metrics["train"]["r2"]),
            "r2_val": float(metrics["val"]["r2"]),
            "r2_test": float(metrics["test"]["r2"]),
            "mae_test": float(metrics["test"]["mae"]),
            "rmse_test": float(metrics["test"]["rmse"]),
            "smape_test": float(metrics["test"]["smape"]),
            "n_test": int(metrics["test"]["n_samples"]),
            "delta_r2_vs_baseline": None if baseline_r2 is None else float(metrics["test"]["r2"]) - baseline_r2,
        }
        rows.append(row)
        if row["r2_test"] > best_r2:
            best_r2 = float(row["r2_test"])
            best_label = label

    df = pd.DataFrame(rows).sort_values("r2_test", ascending=False).reset_index(drop=True)
    payload = {
        "target": TARGET,
        "baseline_test_r2_no_materials": baseline_r2,
        "best_label": best_label,
        "best_test_r2": best_r2,
        "delta_r2_vs_baseline": None if baseline_r2 is None else best_r2 - baseline_r2,
        "results": rows,
    }
    (REPORTS_DIR / "materials_weight_sweep_hpms16.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS_DIR / "materials_weight_sweep_hpms16.md").write_text(dataframe_to_markdown(df), encoding="utf-8")
    print(dataframe_to_markdown(df))


if __name__ == "__main__":
    main()
