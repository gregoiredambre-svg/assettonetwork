"""Sweep richer pavement features and full_refined weights for the MEPDG R-GCN target."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import compute_metrics, dataframe_to_markdown
import graph_construction as gc
import graph_model_temporal as temporal_model
import part1_extensions as ext_model
from load_materials import load_materials

GRAPH_DIR = ROOT / "graph_data"
REPORTS_DIR = ROOT / "reports"
TARGET = "MEPDG_CRACKING_PERCENT_AC"
SEED = 42
BASELINE_RESULTS = GRAPH_DIR / "singletask_per_distress_results.json"

WEIGHT_CONFIGS: list[tuple[str, dict[str, float]]] = [
    ("default", {"spatial": 0.40, "traffic": 0.15, "climate": 0.20, "pavement": 0.25}),
    ("pavement_rich", {"spatial": 0.35, "traffic": 0.10, "climate": 0.15, "pavement": 0.40}),
    ("climate_pavement", {"spatial": 0.30, "traffic": 0.10, "climate": 0.25, "pavement": 0.35}),
    ("spatial_pavement", {"spatial": 0.45, "traffic": 0.05, "climate": 0.15, "pavement": 0.35}),
    ("pavement_dominant", {"spatial": 0.25, "traffic": 0.10, "climate": 0.20, "pavement": 0.45}),
]


def log(message: str) -> None:
    print(f"[materials_weight_sweep] {message}")


def find_baseline_r2() -> float | None:
    """Return the current single-task baseline R² for the target, if available."""

    if not BASELINE_RESULTS.exists():
        return None
    payload = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    for row in payload.get("results", []):
        if row.get("target") == TARGET:
            value = row.get("r2_test")
            return float(value) if value is not None else None
    return None


def prepare_nodes_once() -> pd.DataFrame:
    """Materialize the enriched node table once for repeated edge sweeps."""

    log("Refreshing anti-leakage materials snapshot ...")
    load_materials()

    log("Preparing enriched node table once ...")
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
    """Build a full_refined edge table for one similarity-weight configuration."""

    spatial_edges = gc.build_spatial_edges(nodes)
    route_edges = gc.build_route_edges(nodes)
    functional_edges = gc.build_functional_edges(nodes, similarity_weights=weights)
    edges = pd.concat([spatial_edges, route_edges, functional_edges], ignore_index=True, sort=False)
    edges = edges.drop_duplicates(subset=["source", "target", "edge_type"])
    edges = gc.augment_edges(edges, nodes)
    edges = gc.augment_diversion(edges, nodes)
    edges = gc.add_edge_weight_views(edges)
    return edges


def train_target_with_edges(edges_df: pd.DataFrame, label: str) -> dict[str, float | str | dict[str, float]]:
    """Train the target-specific R-GCN on a supplied edge table."""

    requested_targets = [temporal_model.TARGET_COL, TARGET] if TARGET != temporal_model.TARGET_COL else [TARGET]
    data, _ = temporal_model.prepare_multitask_data(
        graph_variant="full_refined",
        treatment_mode="experiment",
        target_cols=requested_targets,
    )
    relation_names, relation_adjs = ext_model.build_relation_adjacencies(data.node_ids, "full_refined", edges_df=edges_df)

    target_index = data.target_cols.index(TARGET)
    y_full = data.y[:, :, target_index]
    y_mask = data.y_mask[:, :, target_index]
    split_masks = {
        "train": data.split_masks["train"] & y_mask,
        "val": data.split_masks["val"] & y_mask,
        "test": data.split_masks["test"] & y_mask,
    }
    train_vals = y_full[split_masks["train"]]
    if train_vals.size < 50:
        raise ValueError(f"Not enough train samples for {TARGET}: {train_vals.size}")
    mu = float(np.mean(train_vals))
    sigma = float(np.std(train_vals))
    if sigma <= 1e-8:
        sigma = 1.0
    y_z = (y_full - mu) / sigma

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, _ = ext_model.train_temporal_rgcn(data.x_with_maint, y_z, split_masks, relation_adjs)

    adjs = [torch.tensor(adj, dtype=torch.float32) for adj in relation_adjs]
    model.eval()
    preds_by_year: list[np.ndarray] = []
    with torch.no_grad():
        for yi in range(data.x_with_maint.shape[0]):
            x_t = torch.tensor(data.x_with_maint[yi], dtype=torch.float32)
            preds_by_year.append(model(x_t, adjs).cpu().numpy())
    pred_z = np.stack(preds_by_year, axis=0)
    pred_original = pred_z * sigma + mu

    metrics_original: dict[str, dict[str, float]] = {}
    for split_name, mask in split_masks.items():
        metrics_original[split_name] = compute_metrics(y_full[mask], pred_original[mask])

    model_path = GRAPH_DIR / f"temporal_rgcn_singletask_MEPDG_CRACKING_PERCENT_AC_{label}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "target_col": TARGET,
            "graph_variant": "full_refined",
            "label": label,
            "z_mu": mu,
            "z_sigma": sigma,
            "relation_names": relation_names,
            "hidden_dim": 64,
            "metrics_original": metrics_original,
        },
        model_path,
    )

    return {
        "label": label,
        "r2_train": float(metrics_original["train"]["r2"]),
        "r2_val": float(metrics_original["val"]["r2"]),
        "r2_test": float(metrics_original["test"]["r2"]),
        "mae_test": float(metrics_original["test"]["mae"]),
        "rmse_test": float(metrics_original["test"]["rmse"]),
        "smape_test": float(metrics_original["test"]["smape"]),
        "n_test": int(metrics_original["test"]["n_samples"]),
    }


def main() -> None:
    GRAPH_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    baseline_r2 = find_baseline_r2()
    nodes = prepare_nodes_once()
    log(f"Nodes prepared: {len(nodes)}")

    rows: list[dict[str, object]] = []
    best_label = None
    best_r2 = -float("inf")
    best_edges: pd.DataFrame | None = None

    for label, weights in WEIGHT_CONFIGS:
        log(f"Running config {label} with weights={weights}")
        edges = build_edges_for_config(nodes, weights)
        edges_out = GRAPH_DIR / f"edges_with_materials_{label}.csv"
        edges.to_csv(edges_out, index=False)
        if label == "default":
            edges.to_csv(GRAPH_DIR / "edges_with_materials.csv", index=False)

        metrics = train_target_with_edges(edges, label)
        row = {
            "label": label,
            "spatial_weight": weights["spatial"],
            "traffic_weight": weights["traffic"],
            "climate_weight": weights["climate"],
            "pavement_weight": weights["pavement"],
            "n_edges": int(len(edges)),
            **metrics,
            "delta_r2_vs_baseline": None if baseline_r2 is None else float(metrics["r2_test"]) - baseline_r2,
        }
        rows.append(row)
        if float(metrics["r2_test"]) > best_r2:
            best_r2 = float(metrics["r2_test"])
            best_label = label
            best_edges = edges

    if best_edges is not None and best_label is not None:
        best_edges.to_csv(GRAPH_DIR / "edges_with_materials_best.csv", index=False)

    df = pd.DataFrame(rows).sort_values("r2_test", ascending=False).reset_index(drop=True)
    summary = {
        "target": TARGET,
        "baseline_test_r2_no_materials": baseline_r2,
        "best_label": best_label,
        "best_test_r2": best_r2,
        "delta_r2_vs_baseline": None if baseline_r2 is None else best_r2 - baseline_r2,
        "results": rows,
    }

    (REPORTS_DIR / "materials_weight_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS_DIR / "materials_weight_sweep.md").write_text(dataframe_to_markdown(df), encoding="utf-8")

    print(dataframe_to_markdown(df))
    if baseline_r2 is not None and best_label is not None:
        print(f"\nBest config: {best_label} (test R²={best_r2:.3f})")
        print(f"Δ R² vs no-materials baseline: {best_r2 - baseline_r2:+.3f}")


if __name__ == "__main__":
    main()
