"""Part 1 reinforcement sprint: R-GCN, OOD validation, and OSM comparison.

Purpose:
- add a relation-aware graph model variant (R-GCN style) for the temporal task,
- test out-of-distribution generalisation by state holdout,
- compare the current interdependency graph against local OSM drive topology.

Outputs:
- reports/part1_rgcn_temporal.csv
- reports/part1_rgcn_temporal.json
- reports/part1_ood_temporal.csv
- reports/part1_ood_temporal.json
- reports/part1_ood_static.csv
- reports/part1_ood_static.json
- reports/osm_comparison_summary.csv
- reports/osm_comparison_summary.json
- graph_data/osm_edge_comparison.csv
- graph_data/edges_osm_supported.csv
- reports/same_route_real_axis_audit.csv
- reports/same_route_real_axis_summary.csv
- reports/same_route_real_axis_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import torch
from pyproj import Transformer
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from evaluation import compare_models, compute_metrics, dataframe_to_markdown, save_metrics_table
import graph_model as static_model
import graph_model_temporal as temporal_model

ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"

SEED = 42
TEMPORAL_VARIANTS = ["spatial", "spatial_route", "full_refined"]
STATIC_VARIANTS = ["spatial", "spatial_route", "full_refined"]
OSM_VARIANT = "spatial_route"
OSM_MAX_COMPONENTS = 40
OSM_BUFFER_DEG = 0.18
OSM_MAX_FAILED_ATTEMPTS = 12
OSM_COMPONENT_TIMEOUT_SEC = 90
OSM_MAX_SNAP_M = 1000.0
OSM_MAX_COMPONENT_DIAGONAL_KM = 250.0
OSM_MIN_VALID_SNAP_SHARE = 0.5
OSM_MAX_ATTEMPTED_UNITS = 80
OSM_DIRECT_RATIO = 1.5
OSM_SUPPORTED_RATIO = 3.0
SAME_ROUTE_AUDIT_MIN_DISTANCE_KM = 30.0
SAME_ROUTE_AUDIT_MAX_EDGES = 100
SAME_ROUTE_OSM_BUFFER_DEG = 0.08


def log(message: str) -> None:
    print(f"[part1_extensions] {message}", flush=True)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(2.0 * r * math.asin(math.sqrt(a)))


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return compute_metrics(y_true, y_pred)


def normalize_adjacency(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.copy().astype(np.float32)
    degree = matrix.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    return (matrix * inv_sqrt[:, None] * inv_sqrt[None, :]).astype(np.float32)


def build_relation_adjacencies(
    node_ids: list[str],
    graph_variant: str,
    edges_df: pd.DataFrame | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """Build per-relation adjacency matrices, optionally from a supplied edge table."""

    edges = edges_df.copy() if edges_df is not None else pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    edges = temporal_model.filter_edges_for_variant(edges, graph_variant)
    relation_order = [edge_type for edge_type in ["spatial", "same_route", "same_functional_class"] if edge_type in set(edges["edge_type"])]
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    out_names: list[str] = []
    out_adjs: list[np.ndarray] = []
    for edge_type in relation_order:
        mat = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
        subset = edges[edges["edge_type"] == edge_type].copy()
        subset["distance_km"] = pd.to_numeric(subset["distance_km"], errors="coerce").fillna(50.0)
        subset["diversion_potential"] = pd.to_numeric(subset.get("diversion_potential"), errors="coerce").fillna(0.0)
        for row in subset.itertuples(index=False):
            if str(row.source) not in index or str(row.target) not in index:
                continue
            i = index[str(row.source)]
            j = index[str(row.target)]
            learned_weight = getattr(row, "weight_deterioration", np.nan)
            if pd.notna(learned_weight):
                weight = float(learned_weight)
            else:
                closeness = 1.0 / (1.0 + float(row.distance_km))
                base = {"same_route": 1.0, "spatial": 0.7, "same_functional_class": 0.4}.get(edge_type, 0.3)
                weight = base + 0.5 * float(row.diversion_potential) + 5.0 * closeness
            mat[i, j] = max(mat[i, j], weight)
            mat[j, i] = max(mat[j, i], weight)
        out_names.append(edge_type)
        out_adjs.append(normalize_adjacency(mat))
    return out_names, out_adjs


class RelationSnapshotGCN(nn.Module):
    def __init__(self, input_dim: int, num_relations: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.rel1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(num_relations)])
        self.rel2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)])
        self.self1 = nn.Linear(input_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        h = self.self1(x)
        for adj, layer in zip(adjs, self.rel1):
            h = h + layer(torch.matmul(adj, x))
        h = F.relu(h)
        h = self.dropout(h)

        z = self.self2(h)
        for adj, layer in zip(adjs, self.rel2):
            z = z + layer(torch.matmul(adj, h))
        z = F.relu(z)
        z = self.dropout(z)
        return self.out(z).squeeze(-1)


class MultiTaskRelationGCN(nn.Module):
    """Relation-aware GCN with a shared encoder and one head per distress target."""

    def __init__(self, input_dim: int, num_relations: int, n_targets: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.rel1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for _ in range(num_relations)])
        self.rel2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)])
        self.self1 = nn.Linear(input_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        head_hidden = max(hidden_dim // 2, 1)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(n_targets)
            ]
        )

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        h = self.self1(x)
        for adj, layer in zip(adjs, self.rel1):
            h = h + layer(torch.matmul(adj, x))
        h = F.relu(h)
        h = self.dropout(h)

        z = self.self2(h)
        for adj, layer in zip(adjs, self.rel2):
            z = z + layer(torch.matmul(adj, h))
        z = F.relu(z)
        z = self.dropout(z)
        outputs = [head(z).squeeze(-1) for head in self.heads]
        return torch.stack(outputs, dim=-1)


def evaluate_temporal_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    split_masks: dict[str, np.ndarray],
    adjacency_payload,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        if isinstance(adjacency_payload, list):
            adjs = [torch.tensor(adj, dtype=torch.float32) for adj in adjacency_payload]
        else:
            adjs = torch.tensor(adjacency_payload, dtype=torch.float32)
        preds_by_year = []
        for yi in range(x.shape[0]):
            x_t = torch.tensor(x[yi], dtype=torch.float32)
            pred = model(x_t, adjs).cpu().numpy()
            preds_by_year.append(pred)
        preds = np.stack(preds_by_year, axis=0)
    for split_name, split_mask in split_masks.items():
        true = y[split_mask]
        pred = preds[split_mask]
        metrics[split_name] = metric_block(true, pred)
    return metrics


def train_temporal_rgcn(
    x: np.ndarray,
    y: np.ndarray,
    split_masks: dict[str, np.ndarray],
    relation_adjs: list[np.ndarray],
    hidden_dim: int = 64,
    max_epochs: int = 180,
    patience: int = 20,
) -> tuple[RelationSnapshotGCN, dict[str, dict[str, float]]]:
    model = RelationSnapshotGCN(input_dim=x.shape[-1], num_relations=len(relation_adjs), hidden_dim=hidden_dim, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    adjs = [torch.tensor(adj, dtype=torch.float32) for adj in relation_adjs]
    train_mask = split_masks["train"]
    val_mask = split_masks["val"]
    best_state = None
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        losses = []
        for yi in range(x.shape[0]):
            if not train_mask[yi].any():
                continue
            pred = model(torch.tensor(x[yi], dtype=torch.float32), adjs)
            target = torch.tensor(y[yi], dtype=torch.float32)
            mask_t = torch.tensor(train_mask[yi], dtype=torch.bool)
            losses.append(F.mse_loss(pred[mask_t], target[mask_t]))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        val_metrics = evaluate_temporal_model(model, x, y, {"val": val_mask}, relation_adjs)["val"]
        val_loss = val_metrics["rmse"] ** 2
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 20 == 0:
            log(f"R-GCN epoch={epoch:03d} train_loss={loss.item():.4f} val_mse={val_loss:.4f}")
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_temporal_model(model, x, y, split_masks, relation_adjs)
    return model, metrics


def train_multitask_rgcn(
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    split_masks: dict[str, np.ndarray],
    relation_adjs: list[np.ndarray],
    target_means: dict[str, float],
    target_stds: dict[str, float],
    hidden_dim: int = 64,
    max_epochs: int = 180,
    patience: int = 20,
) -> tuple[MultiTaskRelationGCN, float, dict[str, object]]:
    """Train a multi-task relation-aware GCN with train-only target scaling."""

    target_cols = list(target_means.keys())
    model = MultiTaskRelationGCN(
        input_dim=x.shape[-1],
        num_relations=len(relation_adjs),
        n_targets=len(target_cols),
        hidden_dim=hidden_dim,
        dropout=0.2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    adjs = [torch.tensor(adj, dtype=torch.float32) for adj in relation_adjs]
    train_mask = split_masks["train"]
    val_mask = split_masks["val"]
    means_t = torch.tensor([target_means[target] for target in target_cols], dtype=torch.float32)
    stds_t = torch.tensor([target_stds[target] for target in target_cols], dtype=torch.float32)
    best_state = None
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        losses = []
        for yi in range(x.shape[0]):
            if not train_mask[yi].any():
                continue
            pred = model(torch.tensor(x[yi], dtype=torch.float32), adjs)
            target = torch.tensor(y[yi], dtype=torch.float32)
            target_mask = torch.tensor(y_mask[yi], dtype=torch.bool)
            split_mask_t = torch.tensor(train_mask[yi], dtype=torch.bool)
            target_losses = []
            for ti in range(len(target_cols)):
                joint_mask = split_mask_t & target_mask[:, ti]
                if not joint_mask.any():
                    continue
                pred_t = pred[:, ti][joint_mask]
                true_t = target[:, ti][joint_mask]
                z_pred = (pred_t - means_t[ti]) / stds_t[ti]
                z_true = (true_t - means_t[ti]) / stds_t[ti]
                target_losses.append(F.mse_loss(z_pred, z_true))
            if target_losses:
                losses.append(torch.stack(target_losses).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        _, val_metrics = evaluate_multitask_model(
            model,
            x,
            y,
            y_mask,
            {"val": val_mask},
            adjs,
            target_cols,
            target_means,
            target_stds,
            return_val_z_mse=True,
        )
        val_loss = float(val_metrics["val_z_mse"])
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 20 == 0:
            log(f"Multi-task R-GCN epoch={epoch:03d} train_loss={loss.item():.4f} val_z_mse={val_loss:.4f}")
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, aux = evaluate_multitask_model(
        model,
        x,
        y,
        y_mask,
        split_masks,
        adjs,
        target_cols,
        target_means,
        target_stds,
        return_val_z_mse=True,
    )
    return model, best_val, {"metrics": metrics, "val_z_mse": aux["val_z_mse"]}


def evaluate_multitask_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    split_masks: dict[str, np.ndarray],
    adjs,
    target_cols: list[str],
    target_means: dict[str, float],
    target_stds: dict[str, float],
    return_val_z_mse: bool = False,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, float]] | dict[str, dict[str, dict[str, float]]]:
    """Evaluate multi-task predictions on the original scale, per distress target."""

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    model.eval()
    if not isinstance(adjs, list):
        adjs = [torch.tensor(adj, dtype=torch.float32) for adj in adjs]
    aux: dict[str, float] = {}
    means = np.asarray([target_means[target] for target in target_cols], dtype=float)
    stds = np.asarray([target_stds[target] for target in target_cols], dtype=float)
    with torch.no_grad():
        preds_by_year = []
        for yi in range(x.shape[0]):
            x_t = torch.tensor(x[yi], dtype=torch.float32)
            pred = model(x_t, adjs).cpu().numpy()
            preds_by_year.append(pred)
        preds = np.stack(preds_by_year, axis=0)

    for split_name, split_mask in split_masks.items():
        metrics[split_name] = {}
        z_losses = []
        valid_targets = []
        for ti, target in enumerate(target_cols):
            joint_mask = split_mask & y_mask[:, :, ti]
            true = y[:, :, ti][joint_mask]
            pred = preds[:, :, ti][joint_mask]
            if true.size == 0:
                metrics[split_name][target] = {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "smape": np.nan, "mape": np.nan, "n_samples": 0}
                continue
            metrics[split_name][target] = compute_metrics(true, pred)
            if true.size >= 30:
                valid_targets.append(target)
            z_pred = (pred - means[ti]) / stds[ti]
            z_true = (true - means[ti]) / stds[ti]
            z_losses.append(float(np.mean((z_pred - z_true) ** 2)))

        if valid_targets:
            metrics[split_name]["MACRO_MEAN"] = {
                "r2": float(np.mean([metrics[split_name][target]["r2"] for target in valid_targets])),
                "mae": float(np.mean([metrics[split_name][target]["mae"] for target in valid_targets])),
                "rmse": float(np.mean([metrics[split_name][target]["rmse"] for target in valid_targets])),
                "smape": float(np.mean([metrics[split_name][target]["smape"] for target in valid_targets])),
                "mape": float(np.mean([metrics[split_name][target]["mape"] for target in valid_targets])),
                "n_samples": int(sum(metrics[split_name][target]["n_samples"] for target in valid_targets)),
            }
        else:
            metrics[split_name]["MACRO_MEAN"] = {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "smape": np.nan, "mape": np.nan, "n_samples": 0}

        if z_losses:
            aux[f"{split_name}_z_mse"] = float(np.mean(z_losses))

    if return_val_z_mse:
        return metrics, {"val_z_mse": aux.get("val_z_mse", np.inf)}
    return metrics


def train_temporal_gcn_with_masks(
    x: np.ndarray,
    y: np.ndarray,
    split_masks: dict[str, np.ndarray],
    adjacency: np.ndarray,
    hidden_dim: int = 64,
    max_epochs: int = 180,
    patience: int = 20,
) -> tuple[temporal_model.SnapshotGCN, dict[str, dict[str, float]]]:
    model = temporal_model.SnapshotGCN(input_dim=x.shape[-1], hidden_dim=hidden_dim, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    a_t = torch.tensor(adjacency, dtype=torch.float32)
    train_mask = split_masks["train"]
    val_mask = split_masks["val"]
    best_state = None
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        losses = []
        for yi in range(x.shape[0]):
            if not train_mask[yi].any():
                continue
            pred = model(torch.tensor(x[yi], dtype=torch.float32), a_t)
            target = torch.tensor(y[yi], dtype=torch.float32)
            mask_t = torch.tensor(train_mask[yi], dtype=torch.bool)
            losses.append(F.mse_loss(pred[mask_t], target[mask_t]))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        val_metrics = evaluate_temporal_model(model, x, y, {"val": val_mask}, adjacency)["val"]
        val_loss = val_metrics["rmse"] ** 2
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 20 == 0:
            log(f"GCN OOD epoch={epoch:03d} train_loss={loss.item():.4f} val_mse={val_loss:.4f}")
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_temporal_model(model, x, y, split_masks, adjacency)
    return model, metrics


def build_year_split_masks(data: temporal_model.TemporalData) -> dict[str, np.ndarray]:
    year_index = {year: idx for idx, year in enumerate(data.years)}
    masks = {
        "train": np.zeros_like(data.mask, dtype=bool),
        "val": np.zeros_like(data.mask, dtype=bool),
        "test": np.zeros_like(data.mask, dtype=bool),
    }
    for year in data.train_years:
        masks["train"][year_index[year]] = data.mask[year_index[year]]
    for year in data.val_years:
        masks["val"][year_index[year]] = data.mask[year_index[year]]
    for year in data.test_years:
        masks["test"][year_index[year]] = data.mask[year_index[year]]
    return masks


def save_v2_comparison(stem: str, results: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Save a comparison table for a collection of model results."""

    frame = compare_models(results)
    save_metrics_table(
        frame,
        REPORT_DIR / f"{stem}.json",
        REPORT_DIR / f"{stem}.md",
    )
    print(f"\n### {stem}")
    print(dataframe_to_markdown(frame))
    return frame


def greedy_holdout_groups(counts: pd.Series, val_share: float = 0.15, test_share: float = 0.20) -> tuple[set[str], set[str]]:
    total = float(counts.sum())
    ordered = counts.sort_values(ascending=False)
    val_groups: set[str] = set()
    test_groups: set[str] = set()
    val_target = total * val_share
    test_target = total * test_share
    val_acc = 0.0
    test_acc = 0.0
    toggle = "test"
    for group, count in ordered.items():
        if toggle == "test" and test_acc < test_target:
            test_groups.add(str(group))
            test_acc += float(count)
        elif val_acc < val_target:
            val_groups.add(str(group))
            val_acc += float(count)
        elif test_acc < test_target:
            test_groups.add(str(group))
            test_acc += float(count)
        toggle = "val" if toggle == "test" else "test"
    return val_groups, test_groups


def build_state_ood_masks(data: temporal_model.TemporalData) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    node_meta = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)[["node_id", "state_code", "route_key"]].copy()
    node_meta["node_id"] = node_meta["node_id"].astype(str)
    rows = data.panel.loc[data.panel["target_t1"].notna(), ["node_id", "YEAR"]].copy()
    rows = rows.merge(node_meta, on="node_id", how="left")
    counts = rows["state_code"].astype(str).value_counts()
    val_states, test_states = greedy_holdout_groups(counts)
    train_states = set(counts.index.astype(str)) - val_states - test_states
    split_masks = {
        "train": np.zeros_like(data.mask, dtype=bool),
        "val": np.zeros_like(data.mask, dtype=bool),
        "test": np.zeros_like(data.mask, dtype=bool),
    }
    year_index = {year: idx for idx, year in enumerate(data.years)}
    node_index = {node_id: idx for idx, node_id in enumerate(data.node_ids)}
    for row in rows.itertuples(index=False):
        yi = year_index[int(row.YEAR)]
        ni = node_index[str(row.node_id)]
        state = str(row.state_code)
        if state in test_states:
            split_masks["test"][yi, ni] = True
        elif state in val_states:
            split_masks["val"][yi, ni] = True
        elif state in train_states:
            split_masks["train"][yi, ni] = True
    info = {
        "train_states": sorted(train_states),
        "val_states": sorted(val_states),
        "test_states": sorted(test_states),
        "transition_counts": {name: int(mask.sum()) for name, mask in split_masks.items()},
    }
    return split_masks, info


def train_tabular_on_masks(data: temporal_model.TemporalData, split_masks: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, float]]]:
    feature_cols = data.local_feature_cols
    panel = data.panel.copy()
    year_index = {year: idx for idx, year in enumerate(data.years)}
    node_index = {node_id: idx for idx, node_id in enumerate(data.node_ids)}
    rows = panel.loc[panel["target_t1"].notna()].copy()
    split_labels = []
    for row in rows.itertuples(index=False):
        yi = year_index[int(row.YEAR)]
        ni = node_index[str(row.node_id)]
        if split_masks["train"][yi, ni]:
            split_labels.append("train")
        elif split_masks["val"][yi, ni]:
            split_labels.append("val")
        elif split_masks["test"][yi, ni]:
            split_labels.append("test")
        else:
            split_labels.append("drop")
    rows["split"] = split_labels
    rows = rows[rows["split"] != "drop"].copy()
    train = rows[rows["split"] == "train"].copy()
    val = rows[rows["split"] == "val"].copy()
    test = rows[rows["split"] == "test"].copy()

    x_train = train[feature_cols].to_numpy()
    x_val = val[feature_cols].to_numpy()
    x_test = test[feature_cols].to_numpy()
    y_train = train["target_t1"].to_numpy(dtype=float)
    y_val = val["target_t1"].to_numpy(dtype=float)
    y_test = test["target_t1"].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train))
    x_val = scaler.transform(imputer.transform(x_val))
    x_test = scaler.transform(imputer.transform(x_test))

    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    ridge.fit(x_train, y_train)
    rf.fit(x_train, y_train)

    out = {}
    for name, model in [("ridge_local", ridge), ("rf_local", rf)]:
        out[name] = {
            "train": metric_block(y_train, model.predict(x_train)),
            "val": metric_block(y_val, model.predict(x_val)),
            "test": metric_block(y_test, model.predict(x_test)),
        }
    return out


def temporal_rgcn_experiment() -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    summary: dict[str, object] = {"variants": {}}
    for variant in TEMPORAL_VARIANTS:
        log(f"Running temporal R-GCN extension for {variant}")
        data, _ = temporal_model.prepare_temporal_data(variant, treatment_mode="experiment")
        relation_names, relation_adjs = build_relation_adjacencies(data.node_ids, variant)
        split_masks = build_year_split_masks(data)
        model, metrics = train_temporal_rgcn(data.x_with_maint, data.y, split_masks, relation_adjs)
        rows.append(
            {
                "graph_variant": variant,
                "relations": ",".join(relation_names),
                "train_r2": metrics["train"]["r2"],
                "val_r2": metrics["val"]["r2"],
                "test_r2": metrics["test"]["r2"],
                "test_rmse": metrics["test"]["rmse"],
                "test_mae": metrics["test"]["mae"],
            }
        )
        summary["variants"][variant] = {"relations": relation_names, "metrics": metrics}
        torch.save(
            {
                "model_type": "relation_snapshot_gcn",
                "graph_variant": variant,
                "treatment_mode": "experiment",
                "relation_names": relation_names,
                "feature_cols": data.local_feature_cols,
                "node_ids": data.node_ids,
                "years": data.years,
                "hidden_dim": 64,
                "state_dict": model.state_dict(),
            },
            GRAPH_DIR / f"temporal_rgcn_{variant}.pt",
        )
    frame = pd.DataFrame(rows)
    save_v2_comparison(
        "rgcn_temporal_metrics_v2",
        {
            f"R-GCN ({row['graph_variant']})": summary["variants"][row["graph_variant"]]["metrics"]
            for row in rows
        },
    )
    return frame, summary


def temporal_ood_experiment() -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    summary: dict[str, object] = {"variants": {}}
    for variant in TEMPORAL_VARIANTS:
        log(f"Running temporal OOD validation for {variant}")
        data, _ = temporal_model.prepare_temporal_data(variant, treatment_mode="experiment")
        split_masks, split_info = build_state_ood_masks(data)
        tabular = train_tabular_on_masks(data, split_masks)
        relation_names, relation_adjs = build_relation_adjacencies(data.node_ids, variant)
        _, rgcn_metrics = train_temporal_rgcn(
            data.x_with_maint,
            data.y,
            split_masks,
            relation_adjs,
            max_epochs=120,
            patience=15,
        )
        _, gcn_eval = train_temporal_gcn_with_masks(
            data.x_with_maint,
            data.y,
            split_masks,
            temporal_model.load_graph_adjacency(data.node_ids, variant),
            max_epochs=120,
            patience=15,
        )
        rows.append(
            {
                "graph_variant": variant,
                "train_states": ";".join(split_info["train_states"]),
                "val_states": ";".join(split_info["val_states"]),
                "test_states": ";".join(split_info["test_states"]),
                "rf_test_r2": tabular["rf_local"]["test"]["r2"],
                "ridge_test_r2": tabular["ridge_local"]["test"]["r2"],
                "gcn_test_r2": gcn_eval["test"]["r2"],
                "rgcn_test_r2": rgcn_metrics["test"]["r2"],
                "gcn_test_rmse": gcn_eval["test"]["rmse"],
                "rgcn_test_rmse": rgcn_metrics["test"]["rmse"],
                "test_transitions": split_info["transition_counts"]["test"],
            }
        )
        summary["variants"][variant] = {
            "split_info": split_info,
            "tabular": tabular,
            "gcn_metrics": gcn_eval,
            "rgcn_metrics": rgcn_metrics,
            "relations": relation_names,
        }
    save_v2_comparison(
        "ood_temporal_rf_metrics_v2",
        {f"RF ({variant})": summary["variants"][variant]["tabular"]["rf_local"] for variant in TEMPORAL_VARIANTS},
    )
    save_v2_comparison(
        "ood_temporal_ridge_metrics_v2",
        {f"Ridge ({variant})": summary["variants"][variant]["tabular"]["ridge_local"] for variant in TEMPORAL_VARIANTS},
    )
    save_v2_comparison(
        "ood_temporal_gcn_metrics_v2",
        {f"GCN ({variant})": summary["variants"][variant]["gcn_metrics"] for variant in TEMPORAL_VARIANTS},
    )
    save_v2_comparison(
        "ood_temporal_rgcn_metrics_v2",
        {f"R-GCN ({variant})": summary["variants"][variant]["rgcn_metrics"] for variant in TEMPORAL_VARIANTS},
    )
    return pd.DataFrame(rows), summary


class StaticMLP(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_static_with_indices(features: np.ndarray, targets: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    x_t = torch.tensor(features, dtype=torch.float32)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(targets)
    y_t = torch.tensor(y_scaled, dtype=torch.float32)
    model = StaticMLP(input_dim=features.shape[-1], hidden_dim=64, out_dim=targets.shape[1], dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    no_improve = 0
    patience = 25
    train_t = torch.tensor(train_idx, dtype=torch.long)
    val_t = torch.tensor(val_idx, dtype=torch.long)
    for epoch in range(1, 181):
        model.train()
        opt.zero_grad()
        pred = model(x_t)
        loss = F.mse_loss(pred[train_t], y_t[train_t])
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_loss = F.mse_loss(model(x_t)[val_t], y_t[val_t]).item()
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred_scaled = model(x_t).cpu().numpy()
    preds = scaler.inverse_transform(pred_scaled)
    ridge = Ridge(alpha=1.0)
    ridge.fit(features[train_idx], y_scaled[train_idx])
    ridge_preds = scaler.inverse_transform(ridge.predict(features))

    target_names = ["delta_vht_proxy", "connectivity_loss_pct", "disconnected_od_pct", "disruption_score"]
    metrics = {"targets": {}}
    for i, target_name in enumerate(target_names):
        y = targets[:, i]
        yp = preds[:, i]
        yr = ridge_preds[:, i]
        metrics["targets"][target_name] = {
            "mlp": {
                "train": metric_block(y[train_idx], yp[train_idx]),
                "val": metric_block(y[val_idx], yp[val_idx]),
                "test": metric_block(y[test_idx], yp[test_idx]),
            },
            "ridge": {
                "train": metric_block(y[train_idx], yr[train_idx]),
                "val": metric_block(y[val_idx], yr[val_idx]),
                "test": metric_block(y[test_idx], yr[test_idx]),
            },
        }
    return metrics, preds, ridge_preds


def build_static_state_holdout(scenarios: pd.DataFrame, node_state: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    dominant_states = []
    for row in scenarios.itertuples(index=False):
        states = [node_state.get(node) for node in str(row.closed_node_ids).split(";") if node]
        states = [state for state in states if state is not None]
        if not states:
            dominant_states.append("unknown")
            continue
        counts = pd.Series(states).value_counts()
        dominant_states.append(str(counts.index[0]))
    scenarios = scenarios.copy()
    scenarios["dominant_state"] = dominant_states
    counts = scenarios["dominant_state"].value_counts()
    val_states, test_states = greedy_holdout_groups(counts)
    train_states = set(counts.index.astype(str)) - val_states - test_states
    train_idx = scenarios.index[scenarios["dominant_state"].isin(train_states)].to_numpy()
    val_idx = scenarios.index[scenarios["dominant_state"].isin(val_states)].to_numpy()
    test_idx = scenarios.index[scenarios["dominant_state"].isin(test_states)].to_numpy()
    info = {
        "train_states": sorted(train_states),
        "val_states": sorted(val_states),
        "test_states": sorted(test_states),
        "scenario_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
    }
    return train_idx, val_idx, test_idx, info


def static_ood_experiment() -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    summary: dict[str, object] = {"variants": {}}
    node_state = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False).set_index("node_id")["state_code"].astype(str).to_dict()
    target_names = ["delta_vht_proxy", "connectivity_loss_pct", "disconnected_od_pct", "disruption_score"]
    for variant in STATIC_VARIANTS:
        log(f"Running static OOD validation for {variant}")
        scenarios = pd.read_csv(GRAPH_DIR / f"network_scenarios_{variant}.csv", low_memory=False)
        features = pd.read_csv(GRAPH_DIR / f"network_scenario_features_{variant}.csv", low_memory=False).to_numpy(dtype=np.float32)
        targets = scenarios[target_names].to_numpy(dtype=np.float32)
        train_idx, val_idx, test_idx, split_info = build_static_state_holdout(scenarios, node_state)
        metrics, _, _ = train_static_with_indices(features, targets, train_idx, val_idx, test_idx)
        row = {
            "graph_variant": variant,
            "train_states": ";".join(split_info["train_states"]),
            "val_states": ";".join(split_info["val_states"]),
            "test_states": ";".join(split_info["test_states"]),
            "test_scenarios": split_info["scenario_counts"]["test"],
        }
        for target_name in target_names:
            row[f"{target_name}_mlp_test_r2"] = metrics["targets"][target_name]["mlp"]["test"]["r2"]
            row[f"{target_name}_ridge_test_r2"] = metrics["targets"][target_name]["ridge"]["test"]["r2"]
        rows.append(row)
        summary["variants"][variant] = {"split_info": split_info, "metrics": metrics}
    save_v2_comparison(
        "ood_static_mlp_metrics_v2",
        {
            f"Static MLP {variant}::{target_name}": summary["variants"][variant]["metrics"]["targets"][target_name]["mlp"]
            for variant in STATIC_VARIANTS
            for target_name in target_names
        },
    )
    save_v2_comparison(
        "ood_static_ridge_metrics_v2",
        {
            f"Static Ridge {variant}::{target_name}": summary["variants"][variant]["metrics"]["targets"][target_name]["ridge"]
            for variant in STATIC_VARIANTS
            for target_name in target_names
        },
    )
    return pd.DataFrame(rows), summary


def fetch_osm_graph_for_bbox(north: float, south: float, east: float, west: float):
    """Fetch an OSM drive graph for a lat/lon bounding box.

    OSMnx 2.x expects a single bbox tuple ordered as (left, bottom, right, top),
    i.e. (west, south, east, north). Older project notes often used
    (north, south, east, west), which can silently request the wrong area and
    produce invalid snapping distances. Keep the function signature explicit in
    geographic terms, but pass the tuple in the order OSMnx expects.
    """
    ox.settings.requests_timeout = 60
    ox.settings.use_cache = True
    bbox = (west, south, east, north)
    return ox.graph_from_bbox(
        bbox,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )
def bbox_diagonal_km(north: float, south: float, east: float, west: float) -> float:
    return haversine_km(west, south, east, north)


def snap_share_within(distances: list[float], threshold_m: float) -> float:
    valid = [distance for distance in distances if np.isfinite(distance)]
    if not valid:
        return 0.0
    return float(np.mean(np.array(valid) <= threshold_m))


def edge_distance_bin(distance_km: float) -> str:
    if not np.isfinite(distance_km):
        return "unknown"
    if distance_km <= 1:
        return "0-1 km"
    if distance_km <= 5:
        return "1-5 km"
    if distance_km <= 10:
        return "5-10 km"
    if distance_km <= 25:
        return "10-25 km"
    if distance_km <= 50:
        return "25-50 km"
    return ">50 km"


def topology_level(status: str) -> str:
    if status in {"same_osm_edge", "short_connected", "supported_connected"}:
        return "supported"
    if status == "long_connected":
        return "weakly_connected"
    if status == "unreachable":
        return "not_connected"
    if status == "snap_too_far":
        return "invalid_snap"
    return "unknown"


class OSMTimeoutError(TimeoutError):
    pass


def _alarm_handler(signum, frame):  # pragma: no cover - signal callback
    del signum, frame
    raise OSMTimeoutError(f"OSM fetch exceeded {OSM_COMPONENT_TIMEOUT_SEC} seconds")


def fetch_osm_graph_with_timeout(north: float, south: float, east: float, west: float):
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(OSM_COMPONENT_TIMEOUT_SEC)
    try:
        return fetch_osm_graph_for_bbox(north, south, east, west)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def min_edge_to_edge_path_m(graph: nx.MultiDiGraph, src_edge: tuple[int, int, int], dst_edge: tuple[int, int, int]) -> float:
    if src_edge == dst_edge:
        return 0.0
    src_u, src_v, _ = src_edge
    dst_u, dst_v, _ = dst_edge
    candidates = []
    for left in (src_u, src_v):
        for right in (dst_u, dst_v):
            try:
                candidates.append(float(nx.shortest_path_length(graph, left, right, weight="length")))
            except Exception:
                continue
    return min(candidates) if candidates else math.nan


def classify_topology_status(snap_ok: bool, same_osm_edge: bool, path_len_m: float, detour_ratio: float) -> str:
    if not snap_ok:
        return "snap_too_far"
    if same_osm_edge:
        return "same_osm_edge"
    if not np.isfinite(path_len_m):
        return "unreachable"
    if np.isfinite(detour_ratio) and detour_ratio <= OSM_DIRECT_RATIO:
        return "short_connected"
    if np.isfinite(detour_ratio) and detour_ratio <= OSM_SUPPORTED_RATIO:
        return "supported_connected"
    return "long_connected"


def _tokenize_osm_attr(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return set()
        raw_items = text.replace("|", ";").replace("/", ";").split(";")
    out = set()
    for item in raw_items:
        token = str(item).strip().lower()
        if token and token != "nan":
            out.add(token)
    return out


def osm_edge_signature(graph: nx.MultiDiGraph, edge: tuple[int, int, int] | None) -> dict[str, set[str]]:
    if edge is None:
        return {"ref": set(), "name": set(), "highway": set()}
    data = graph.get_edge_data(edge[0], edge[1], edge[2], default={}) or {}
    return {
        "ref": _tokenize_osm_attr(data.get("ref")),
        "name": _tokenize_osm_attr(data.get("name")),
        "highway": _tokenize_osm_attr(data.get("highway")),
    }


def classify_real_axis_verdict(topology_status: str, ref_overlap: bool, name_overlap: bool, distance_km: float) -> str:
    if topology_status == "snap_too_far":
        return "map_match_invalid"
    if topology_status == "unreachable":
        return "not_supported"
    if topology_status == "same_osm_edge":
        return "same_real_axis_strong"
    if ref_overlap and topology_status in {"short_connected", "supported_connected"}:
        return "same_real_axis_supported"
    if name_overlap and topology_status in {"short_connected", "supported_connected"}:
        return "same_real_axis_supported"
    if (ref_overlap or name_overlap) and topology_status == "long_connected":
        return "same_axis_but_far"
    if topology_status in {"short_connected", "supported_connected"}:
        if np.isfinite(distance_km) and distance_km <= 10.0:
            return "connected_but_axis_unclear"
        return "connected_but_far"
    if topology_status == "long_connected":
        return "weak_support_only"
    return "not_supported"


def classify_local_neighbour_verdict(topology_status: str, distance_km: float) -> str:
    if topology_status == "snap_too_far":
        return "unknown"
    if topology_status == "unreachable":
        return "not_local_neighbours"
    if topology_status == "same_osm_edge":
        return "same_segment"
    if topology_status in {"short_connected", "supported_connected"}:
        if np.isfinite(distance_km) and distance_km <= 10.0:
            return "plausible_local_neighbours"
        return "same_corridor_but_not_local"
    if topology_status == "long_connected":
        return "same_corridor_but_not_local"
    return "unknown"


def same_route_real_axis_audit(
    min_distance_km: float = SAME_ROUTE_AUDIT_MIN_DISTANCE_KM,
    max_edges: int = SAME_ROUTE_AUDIT_MAX_EDGES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    same_route = edges[edges["edge_type"].eq("same_route")].copy()
    same_route["distance_km"] = pd.to_numeric(same_route["distance_km"], errors="coerce")
    same_route = same_route[same_route["distance_km"] >= float(min_distance_km)].copy()
    same_route = same_route.sort_values("distance_km", ascending=False).head(int(max_edges)).reset_index(drop=True)
    if same_route.empty:
        return pd.DataFrame(), pd.DataFrame(), {"candidate_edges": 0, "audited_edges": 0, "min_distance_km": float(min_distance_km)}

    node_lookup = nodes.set_index("node_id")[["latitude", "longitude", "route_key", "ROUTE_NO", "ROUTE_SIGNING", "functional_class"]].copy()
    rows: list[dict[str, object]] = []
    failure_examples: list[dict[str, object]] = []

    for idx, row in enumerate(same_route.itertuples(index=False), start=1):
        src = str(row.source)
        dst = str(row.target)
        src_lat = float(node_lookup.at[src, "latitude"])
        src_lon = float(node_lookup.at[src, "longitude"])
        dst_lat = float(node_lookup.at[dst, "latitude"])
        dst_lon = float(node_lookup.at[dst, "longitude"])
        north = max(src_lat, dst_lat) + SAME_ROUTE_OSM_BUFFER_DEG
        south = min(src_lat, dst_lat) - SAME_ROUTE_OSM_BUFFER_DEG
        east = max(src_lon, dst_lon) + SAME_ROUTE_OSM_BUFFER_DEG
        west = min(src_lon, dst_lon) - SAME_ROUTE_OSM_BUFFER_DEG
        diagonal_km = bbox_diagonal_km(north, south, east, west)
        if diagonal_km > OSM_MAX_COMPONENT_DIAGONAL_KM:
            rows.append({
                "source": src,
                "target": dst,
                "distance_km": float(row.distance_km),
                "route_key": node_lookup.at[src, "route_key"],
                "bbox_diagonal_km": diagonal_km,
                "topology_status": "bbox_too_large",
                "real_axis_verdict": "needs_manual_map_check",
                "local_neighbour_verdict": "unknown",
            })
            continue
        try:
            log(f"Same-route OSM audit {idx}/{len(same_route)}: {src} vs {dst}, distance_km={float(row.distance_km):.2f}")
            osm_graph = fetch_osm_graph_with_timeout(north, south, east, west)
            osm_graph_proj = ox.project_graph(osm_graph)
            transformer = Transformer.from_crs("EPSG:4326", osm_graph_proj.graph["crs"], always_xy=True)
            src_x, src_y = transformer.transform(src_lon, src_lat)
            dst_x, dst_y = transformer.transform(dst_lon, dst_lat)
            src_edge_arr, src_dist_arr = ox.distance.nearest_edges(osm_graph_proj, X=[src_x], Y=[src_y], return_dist=True)
            dst_edge_arr, dst_dist_arr = ox.distance.nearest_edges(osm_graph_proj, X=[dst_x], Y=[dst_y], return_dist=True)
            src_edge = tuple(src_edge_arr[0])
            dst_edge = tuple(dst_edge_arr[0])
            src_snap_m = float(src_dist_arr[0])
            dst_snap_m = float(dst_dist_arr[0])
            same_osm_edge = int(src_edge == dst_edge)
            snap_ok = bool(src_snap_m <= OSM_MAX_SNAP_M and dst_snap_m <= OSM_MAX_SNAP_M)
            path_len_m = min_edge_to_edge_path_m(osm_graph_proj, src_edge, dst_edge) if snap_ok else math.nan
            detour_ratio = float(path_len_m / (float(row.distance_km) * 1000.0)) if np.isfinite(path_len_m) and float(row.distance_km) > 0 else math.nan
            topology_status = classify_topology_status(snap_ok, bool(same_osm_edge), path_len_m, detour_ratio)
            src_sig = osm_edge_signature(osm_graph_proj, src_edge)
            dst_sig = osm_edge_signature(osm_graph_proj, dst_edge)
            ref_overlap = bool(src_sig["ref"] & dst_sig["ref"])
            name_overlap = bool(src_sig["name"] & dst_sig["name"])
            real_axis_verdict = classify_real_axis_verdict(topology_status, ref_overlap, name_overlap, float(row.distance_km))
            local_neighbour_verdict = classify_local_neighbour_verdict(topology_status, float(row.distance_km))
            rows.append({
                "source": src,
                "target": dst,
                "distance_km": float(row.distance_km),
                "route_key": node_lookup.at[src, "route_key"],
                "source_route_no": node_lookup.at[src, "ROUTE_NO"],
                "target_route_no": node_lookup.at[dst, "ROUTE_NO"],
                "source_route_signing": node_lookup.at[src, "ROUTE_SIGNING"],
                "target_route_signing": node_lookup.at[dst, "ROUTE_SIGNING"],
                "bbox_diagonal_km": diagonal_km,
                "src_snap_m": src_snap_m,
                "dst_snap_m": dst_snap_m,
                "same_osm_edge": same_osm_edge,
                "osm_path_km": float(path_len_m / 1000.0) if np.isfinite(path_len_m) else math.nan,
                "detour_ratio": detour_ratio,
                "topology_status": topology_status,
                "src_osm_ref": ";".join(sorted(src_sig["ref"])),
                "dst_osm_ref": ";".join(sorted(dst_sig["ref"])),
                "src_osm_name": ";".join(sorted(src_sig["name"])),
                "dst_osm_name": ";".join(sorted(dst_sig["name"])),
                "ref_overlap": int(ref_overlap),
                "name_overlap": int(name_overlap),
                "real_axis_verdict": real_axis_verdict,
                "local_neighbour_verdict": local_neighbour_verdict,
            })
        except Exception as exc:
            failure_examples.append({"source": src, "target": dst, "error": str(exc)})
            rows.append({
                "source": src,
                "target": dst,
                "distance_km": float(row.distance_km),
                "route_key": node_lookup.at[src, "route_key"],
                "bbox_diagonal_km": diagonal_km,
                "topology_status": "fetch_or_match_failed",
                "real_axis_verdict": "needs_manual_map_check",
                "local_neighbour_verdict": "unknown",
            })

    audit = pd.DataFrame(rows)
    summary = audit.groupby(["real_axis_verdict", "local_neighbour_verdict"], as_index=False).agg(
        edges=("source", "size"),
        mean_distance_km=("distance_km", "mean"),
        median_distance_km=("distance_km", "median"),
    )
    meta = {
        "candidate_edges": int(len(same_route)),
        "audited_edges": int(len(audit)),
        "min_distance_km": float(min_distance_km),
        "max_edges": int(max_edges),
        "failure_examples": failure_examples[:10],
    }
    return audit, summary, meta


def osm_comparison_experiment(max_components: int = OSM_MAX_COMPONENTS) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    edges = edges[edges["edge_type"].isin(["spatial", "same_route"])].copy()
    node_lookup = nodes.set_index("node_id")[["latitude", "longitude"]].copy()
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].astype(str).tolist())
    graph.add_edges_from(edges[["source", "target"]].astype(str).itertuples(index=False, name=None))
    components = sorted(nx.connected_components(graph), key=len, reverse=True)

    rows = []
    attempted_components = 0
    processed_components = 0
    failed_attempts = 0
    covered_nodes = set()
    failure_examples: list[dict[str, object]] = []
    meta_path = REPORT_DIR / "osm_comparison_summary.json"
    edge_csv_path = GRAPH_DIR / "osm_edge_comparison.csv"
    summary_csv_path = REPORT_DIR / "osm_comparison_summary.csv"
    debug_component_rows: list[dict[str, object]] = []
    snap_diagnostic_rows: list[dict[str, object]] = []
    component_debug_csv_path = REPORT_DIR / "osm_debug_component_summary.csv"
    snap_debug_csv_path = REPORT_DIR / "osm_snap_diagnostics.csv"
    topology_status_csv_path = REPORT_DIR / "osm_topology_status_summary.csv"
    distance_bin_csv_path = REPORT_DIR / "osm_distance_bin_summary.csv"
    component_edge_csv_path = REPORT_DIR / "osm_component_edge_summary.csv"

    def write_progress_outputs() -> None:
        comparison_now = pd.DataFrame(rows)
        comparison_now.to_csv(edge_csv_path, index=False)
        if comparison_now.empty:
            pd.DataFrame().to_csv(summary_csv_path, index=False)
            pd.DataFrame().to_csv(topology_status_csv_path, index=False)
            pd.DataFrame().to_csv(distance_bin_csv_path, index=False)
            pd.DataFrame().to_csv(component_edge_csv_path, index=False)
        else:
            summary_now = (
                comparison_now.groupby("edge_type", as_index=False)
                .agg(
                    edges=("edge_type", "size"),
                    osm_connected_share=("osm_connected", "mean"),
                    osm_supported_share=("osm_supported", "mean"),
                    same_osm_edge_share=("same_osm_edge", "mean"),
                    mean_src_snap_m=("src_snap_m", "mean"),
                    mean_dst_snap_m=("dst_snap_m", "mean"),
                    median_src_snap_m=("src_snap_m", "median"),
                    median_dst_snap_m=("dst_snap_m", "median"),
                    mean_detour_ratio=("detour_ratio", "mean"),
                    median_detour_ratio=("detour_ratio", "median"),
                )
            )
            summary_now.to_csv(summary_csv_path, index=False)
            topology_summary_now = (
                comparison_now.groupby(["edge_type", "topology_status", "topology_level"], as_index=False)
                .agg(
                    edges=("edge_type", "size"),
                    mean_distance_km=("distance_km", "mean"),
                    median_distance_km=("distance_km", "median"),
                    mean_detour_ratio=("detour_ratio", "mean"),
                    median_detour_ratio=("detour_ratio", "median"),
                )
            )
            topology_summary_now.to_csv(topology_status_csv_path, index=False)
            distance_summary_now = (
                comparison_now.groupby(["edge_type", "distance_bin"], as_index=False)
                .agg(
                    edges=("edge_type", "size"),
                    osm_connected_share=("osm_connected", "mean"),
                    osm_supported_share=("osm_supported", "mean"),
                    same_osm_edge_share=("same_osm_edge", "mean"),
                    mean_detour_ratio=("detour_ratio", "mean"),
                    median_detour_ratio=("detour_ratio", "median"),
                )
            )
            distance_summary_now.to_csv(distance_bin_csv_path, index=False)
            component_summary_now = (
                comparison_now.groupby(["component_rank", "edge_type"], as_index=False)
                .agg(
                    edges=("edge_type", "size"),
                    osm_connected_share=("osm_connected", "mean"),
                    osm_supported_share=("osm_supported", "mean"),
                    same_osm_edge_share=("same_osm_edge", "mean"),
                    mean_distance_km=("distance_km", "mean"),
                    median_distance_km=("distance_km", "median"),
                    mean_detour_ratio=("detour_ratio", "mean"),
                    median_detour_ratio=("detour_ratio", "median"),
                )
            )
            component_summary_now.to_csv(component_edge_csv_path, index=False)
        pd.DataFrame(debug_component_rows).to_csv(component_debug_csv_path, index=False)
        pd.DataFrame(snap_diagnostic_rows).to_csv(snap_debug_csv_path, index=False)

    for comp_rank, comp in enumerate(components, start=1):
        if processed_components >= max_components:
            break
        if failed_attempts >= OSM_MAX_FAILED_ATTEMPTS:
            log(f"Stopping OSM comparison after {failed_attempts} failed fetch attempts.")
            break
        if attempted_components >= OSM_MAX_ATTEMPTED_UNITS:
            log(f"Stopping OSM comparison after inspecting {attempted_components} candidate components.")
            break
        comp_nodes = sorted(comp)
        comp_df = nodes[nodes["node_id"].isin(comp_nodes)].dropna(subset=["latitude", "longitude"]).copy()
        if len(comp_df) < 2:
            continue
        north = float(comp_df["latitude"].max() + OSM_BUFFER_DEG)
        south = float(comp_df["latitude"].min() - OSM_BUFFER_DEG)
        east = float(comp_df["longitude"].max() + OSM_BUFFER_DEG)
        west = float(comp_df["longitude"].min() - OSM_BUFFER_DEG)
        diagonal_km = bbox_diagonal_km(north, south, east, west)
        debug_component_rows.append(
            {
                "attempt": attempted_components + 1,
                "component_rank": comp_rank,
                "component_size": int(len(comp_nodes)),
                "lat_min": float(comp_df["latitude"].min()),
                "lat_max": float(comp_df["latitude"].max()),
                "lon_min": float(comp_df["longitude"].min()),
                "lon_max": float(comp_df["longitude"].max()),
                "bbox_north": north,
                "bbox_south": south,
                "bbox_east": east,
                "bbox_west": west,
                "bbox_diagonal_km": diagonal_km,
                "status": "pending",
                "graph_nodes": 0,
                "graph_edges": 0,
                "median_snap_m": math.nan,
                "snap_share_lt_100m": math.nan,
                "snap_share_lt_500m": math.nan,
                "snap_share_lt_1000m": math.nan,
            }
        )
        if diagonal_km > OSM_MAX_COMPONENT_DIAGONAL_KM:
            message = f"component bbox diagonal {diagonal_km:.1f} km exceeds limit {OSM_MAX_COMPONENT_DIAGONAL_KM:.1f} km"
            log(
                f"Skipping OSM attempt {attempted_components + 1} "
                f"(component_rank={comp_rank}, size={len(comp_nodes)}): {message}"
            )
            debug_component_rows[-1]["status"] = "skipped_bbox_too_large"
            failure_examples.append(
                {
                    "attempt": attempted_components + 1,
                    "component_rank": comp_rank,
                    "component_size": int(len(comp_nodes)),
                    "error": message,
                }
            )
            attempted_components += 1
            meta_snapshot = {
                "attempted_components": attempted_components,
                "processed_components": processed_components,
                "failed_attempts": failed_attempts,
                "skipped_large_components": int(sum(1 for row in debug_component_rows if row.get("status") == "skipped_bbox_too_large")),
                "covered_nodes": int(len(covered_nodes)),
                "total_nodes": int(len(nodes)),
                "coverage_pct": float(100.0 * len(covered_nodes) / len(nodes)) if len(nodes) else 0.0,
                "variant": OSM_VARIANT,
                "failure_examples": failure_examples[-5:],
            }
            meta_path.write_text(json.dumps(meta_snapshot, indent=2), encoding="utf-8")
            write_progress_outputs()
            continue
        attempted_components += 1
        log(
            f"OSM attempt {attempted_components}: component_rank={comp_rank}, size={len(comp_nodes)}, "
            f"bbox=({north:.4f},{south:.4f},{east:.4f},{west:.4f})"
        )
        start_time = time.time()
        try:
            osm_graph = fetch_osm_graph_with_timeout(north, south, east, west)
        except Exception as exc:
            failed_attempts += 1
            message = str(exc)
            log(
                f"OSM fetch failed on attempt {attempted_components} "
                f"(component_rank={comp_rank}, size={len(comp_nodes)}): {message}"
            )
            failure_examples.append(
                {
                    "attempt": attempted_components,
                    "component_rank": comp_rank,
                    "component_size": int(len(comp_nodes)),
                    "error": message,
                }
            )
            debug_component_rows[-1]["status"] = "fetch_failed"
            meta_snapshot = {
                "attempted_components": attempted_components,
                "processed_components": processed_components,
                "failed_attempts": failed_attempts,
                "skipped_large_components": int(sum(1 for row in debug_component_rows if row.get("status") == "skipped_bbox_too_large")),
                "covered_nodes": int(len(covered_nodes)),
                "total_nodes": int(len(nodes)),
                "coverage_pct": float(100.0 * len(covered_nodes) / len(nodes)) if len(nodes) else 0.0,
                "variant": OSM_VARIANT,
                "failure_examples": failure_examples[-5:],
            }
            meta_path.write_text(json.dumps(meta_snapshot, indent=2), encoding="utf-8")
            write_progress_outputs()
            continue
        elapsed = time.time() - start_time
        log(
            f"OSM fetch succeeded on attempt {attempted_components}: "
            f"graph_nodes={osm_graph.number_of_nodes()}, graph_edges={osm_graph.number_of_edges()}, "
            f"elapsed={elapsed:.1f}s"
        )
        debug_component_rows[-1]["status"] = "fetched"
        debug_component_rows[-1]["graph_nodes"] = int(osm_graph.number_of_nodes())
        debug_component_rows[-1]["graph_edges"] = int(osm_graph.number_of_edges())
        osm_graph_proj = ox.project_graph(osm_graph)
        transformer = Transformer.from_crs("EPSG:4326", osm_graph_proj.graph["crs"], always_xy=True)
        comp_edges = edges[edges["source"].isin(comp_nodes) & edges["target"].isin(comp_nodes)].copy()
        if comp_edges.empty:
            continue
        # Snapping preflight
        preflight_snap_distances: list[float] = []
        preflight_snap_rows: list[dict[str, object]] = []
        for node_row in comp_df.itertuples(index=False):
            node_id = str(node_row.node_id)
            lon = float(node_row.longitude)
            lat = float(node_row.latitude)
            try:
                x_coord, y_coord = transformer.transform(lon, lat)
                edge_arr, dist_arr = ox.distance.nearest_edges(
                    osm_graph_proj,
                    X=[x_coord],
                    Y=[y_coord],
                    return_dist=True,
                )
                snap_m = float(dist_arr[0])
                snapped_edge = str(tuple(edge_arr[0]))
            except Exception as exc:
                snap_m = math.nan
                snapped_edge = ""
                preflight_snap_rows.append(
                    {
                        "component_rank": comp_rank,
                        "node_id": node_id,
                        "latitude": lat,
                        "longitude": lon,
                        "snap_m": snap_m,
                        "snapped_edge": snapped_edge,
                        "status": f"snap_failed: {exc}",
                    }
                )
                continue
            preflight_snap_distances.append(snap_m)
            preflight_snap_rows.append(
                {
                    "component_rank": comp_rank,
                    "node_id": node_id,
                    "latitude": lat,
                    "longitude": lon,
                    "snap_m": snap_m,
                    "snapped_edge": snapped_edge,
                    "status": "ok" if snap_m <= OSM_MAX_SNAP_M else "snap_too_far",
                }
            )
        snap_diagnostic_rows.extend(preflight_snap_rows)
        median_snap_m = float(np.nanmedian(preflight_snap_distances)) if preflight_snap_distances else math.nan
        share_100m = snap_share_within(preflight_snap_distances, 100.0)
        share_500m = snap_share_within(preflight_snap_distances, 500.0)
        share_1000m = snap_share_within(preflight_snap_distances, OSM_MAX_SNAP_M)
        debug_component_rows[-1]["median_snap_m"] = median_snap_m
        debug_component_rows[-1]["snap_share_lt_100m"] = share_100m
        debug_component_rows[-1]["snap_share_lt_500m"] = share_500m
        debug_component_rows[-1]["snap_share_lt_1000m"] = share_1000m
        if not np.isfinite(median_snap_m) or share_1000m < OSM_MIN_VALID_SNAP_SHARE:
            failed_attempts += 1
            message = (
                f"invalid OSM snapping: median_snap_m={median_snap_m:.1f}, "
                f"share_within_{OSM_MAX_SNAP_M:.0f}m={share_1000m:.2f}"
            )
            log(
                f"Discarding OSM attempt {attempted_components} "
                f"(component_rank={comp_rank}, size={len(comp_nodes)}): {message}"
            )
            debug_component_rows[-1]["status"] = "discarded_invalid_snapping"
            failure_examples.append(
                {
                    "attempt": attempted_components,
                    "component_rank": comp_rank,
                    "component_size": int(len(comp_nodes)),
                    "error": message,
                }
            )
            meta_snapshot = {
                "attempted_components": attempted_components,
                "processed_components": processed_components,
                "failed_attempts": failed_attempts,
                "skipped_large_components": int(sum(1 for row in debug_component_rows if row.get("status") == "skipped_bbox_too_large")),
                "covered_nodes": int(len(covered_nodes)),
                "total_nodes": int(len(nodes)),
                "coverage_pct": float(100.0 * len(covered_nodes) / len(nodes)) if len(nodes) else 0.0,
                "variant": OSM_VARIANT,
                "failure_examples": failure_examples[-5:],
            }
            meta_path.write_text(json.dumps(meta_snapshot, indent=2), encoding="utf-8")
            write_progress_outputs()
            continue
        debug_component_rows[-1]["status"] = "accepted"
        covered_nodes.update(comp_nodes)
        processed_components += 1
        for row in comp_edges.itertuples(index=False):
            src = str(row.source)
            dst = str(row.target)
            src_lon = float(node_lookup.at[src, "longitude"])
            src_lat = float(node_lookup.at[src, "latitude"])
            dst_lon = float(node_lookup.at[dst, "longitude"])
            dst_lat = float(node_lookup.at[dst, "latitude"])
            geodesic_km = float(pd.to_numeric(getattr(row, "distance_km"), errors="coerce"))
            src_edge = None
            dst_edge = None
            src_snap_m = math.nan
            dst_snap_m = math.nan
            same_osm_edge = 0
            path_len_m = math.nan
            osm_connected = 0
            supported = 0
            topology_status = "match_failed"
            try:
                src_x, src_y = transformer.transform(src_lon, src_lat)
                dst_x, dst_y = transformer.transform(dst_lon, dst_lat)
                src_edge_arr, src_dist_arr = ox.distance.nearest_edges(osm_graph_proj, X=[src_x], Y=[src_y], return_dist=True)
                dst_edge_arr, dst_dist_arr = ox.distance.nearest_edges(osm_graph_proj, X=[dst_x], Y=[dst_y], return_dist=True)
                src_edge = tuple(src_edge_arr[0])
                dst_edge = tuple(dst_edge_arr[0])
                src_snap_m = float(src_dist_arr[0])
                dst_snap_m = float(dst_dist_arr[0])
                same_osm_edge = int(src_edge == dst_edge)
                snap_ok = bool(src_snap_m <= OSM_MAX_SNAP_M and dst_snap_m <= OSM_MAX_SNAP_M)
                if snap_ok:
                    path_len_m = min_edge_to_edge_path_m(osm_graph_proj, src_edge, dst_edge)
                    osm_connected = int(np.isfinite(path_len_m))
                detour_ratio = float(path_len_m / (geodesic_km * 1000.0)) if np.isfinite(path_len_m) and geodesic_km > 0 else math.nan
                topology_status = classify_topology_status(snap_ok, bool(same_osm_edge), path_len_m, detour_ratio)
                supported = int(topology_status in {"same_osm_edge", "short_connected", "supported_connected"})
            except Exception:
                detour_ratio = math.nan
            rows.append(
                {
                    "source": src,
                    "target": dst,
                    "edge_type": str(row.edge_type),
                    "distance_km": geodesic_km,
                    "component_rank": comp_rank,
                    "distance_bin": edge_distance_bin(geodesic_km),
                    "topology_level": topology_level(topology_status),
                    "osm_path_km": float(path_len_m / 1000.0) if np.isfinite(path_len_m) else math.nan,
                    "detour_ratio": detour_ratio,
                    "osm_connected": osm_connected,
                    "osm_supported": supported,
                    "same_osm_edge": same_osm_edge,
                    "src_osm_edge": str(src_edge) if src_edge is not None else "",
                    "dst_osm_edge": str(dst_edge) if dst_edge is not None else "",
                    "src_snap_m": src_snap_m,
                    "dst_snap_m": dst_snap_m,
                    "topology_status": topology_status,
                    "component_size": int(len(comp_nodes)),
                }
            )
        write_progress_outputs()
        meta_snapshot = {
            "attempted_components": attempted_components,
            "processed_components": processed_components,
            "failed_attempts": failed_attempts,
            "skipped_large_components": int(sum(1 for row in debug_component_rows if row.get("status") == "skipped_bbox_too_large")),
            "covered_nodes": int(len(covered_nodes)),
            "total_nodes": int(len(nodes)),
            "coverage_pct": float(100.0 * len(covered_nodes) / len(nodes)) if len(nodes) else 0.0,
            "variant": OSM_VARIANT,
            "failure_examples": failure_examples[-5:],
        }
        meta_path.write_text(json.dumps(meta_snapshot, indent=2), encoding="utf-8")
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison, pd.DataFrame(), {
            "attempted_components": attempted_components,
            "processed_components": processed_components,
            "failed_attempts": failed_attempts,
            "skipped_large_components": int(sum(1 for row in debug_component_rows if row.get("status") == "skipped_bbox_too_large")),
            "covered_nodes": len(covered_nodes),
            "total_nodes": int(len(nodes)),
            "coverage_pct": float(100.0 * len(covered_nodes) / len(nodes)) if len(nodes) else 0.0,
            "variant": OSM_VARIANT,
            "failure_examples": failure_examples[-5:],
        }
    summary = (
        comparison.groupby("edge_type", as_index=False)
        .agg(
            edges=("edge_type", "size"),
            osm_connected_share=("osm_connected", "mean"),
            osm_supported_share=("osm_supported", "mean"),
            same_osm_edge_share=("same_osm_edge", "mean"),
            mean_src_snap_m=("src_snap_m", "mean"),
            mean_dst_snap_m=("dst_snap_m", "mean"),
            median_src_snap_m=("src_snap_m", "median"),
            median_dst_snap_m=("dst_snap_m", "median"),
            mean_detour_ratio=("detour_ratio", "mean"),
            median_detour_ratio=("detour_ratio", "median"),
        )
    )
    topology_summary = (
        comparison.groupby(["edge_type", "topology_status", "topology_level"], as_index=False)
        .agg(
            edges=("edge_type", "size"),
            mean_distance_km=("distance_km", "mean"),
            median_distance_km=("distance_km", "median"),
            mean_detour_ratio=("detour_ratio", "mean"),
            median_detour_ratio=("detour_ratio", "median"),
        )
    )
    topology_summary.to_csv(REPORT_DIR / "osm_topology_status_summary.csv", index=False)
    distance_summary = (
        comparison.groupby(["edge_type", "distance_bin"], as_index=False)
        .agg(
            edges=("edge_type", "size"),
            osm_connected_share=("osm_connected", "mean"),
            osm_supported_share=("osm_supported", "mean"),
            same_osm_edge_share=("same_osm_edge", "mean"),
            mean_detour_ratio=("detour_ratio", "mean"),
            median_detour_ratio=("detour_ratio", "median"),
        )
    )
    distance_summary.to_csv(REPORT_DIR / "osm_distance_bin_summary.csv", index=False)
    component_summary = (
        comparison.groupby(["component_rank", "edge_type"], as_index=False)
        .agg(
            edges=("edge_type", "size"),
            osm_connected_share=("osm_connected", "mean"),
            osm_supported_share=("osm_supported", "mean"),
            same_osm_edge_share=("same_osm_edge", "mean"),
            mean_distance_km=("distance_km", "mean"),
            median_distance_km=("distance_km", "median"),
            mean_detour_ratio=("detour_ratio", "mean"),
            median_detour_ratio=("detour_ratio", "median"),
        )
    )
    component_summary.to_csv(REPORT_DIR / "osm_component_edge_summary.csv", index=False)
    meta = {
        "attempted_components": attempted_components,
        "processed_components": processed_components,
        "failed_attempts": failed_attempts,
        "skipped_large_components": int(sum(1 for row in debug_component_rows if row.get("status") == "skipped_bbox_too_large")),
        "covered_nodes": int(len(covered_nodes)),
        "total_nodes": int(len(nodes)),
        "coverage_pct": float(100.0 * len(covered_nodes) / len(nodes)) if len(nodes) else 0.0,
        "variant": OSM_VARIANT,
        "failure_examples": failure_examples[-5:],
    }
    return comparison, summary, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 1 reinforcement experiments.")
    parser.add_argument(
        "--stage",
        choices=["all", "rgcn", "ood_temporal", "ood_static", "osm", "same_route_osm"],
        default="all",
    )
    parser.add_argument("--osm-max-components", type=int, default=OSM_MAX_COMPONENTS)
    parser.add_argument("--same-route-min-distance-km", type=float, default=SAME_ROUTE_AUDIT_MIN_DISTANCE_KM)
    parser.add_argument("--same-route-max-edges", type=int, default=SAME_ROUTE_AUDIT_MAX_EDGES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    REPORT_DIR.mkdir(exist_ok=True)

    overall: dict[str, object] = {}

    if args.stage in {"all", "rgcn"}:
        rgcn_df, rgcn_json = temporal_rgcn_experiment()
        rgcn_df.to_csv(REPORT_DIR / "part1_rgcn_temporal.csv", index=False)
        (REPORT_DIR / "part1_rgcn_temporal.json").write_text(json.dumps(rgcn_json, indent=2), encoding="utf-8")
        overall["rgcn_temporal"] = rgcn_json
        print("R-GCN temporal")
        print(rgcn_df.to_string(index=False))

    if args.stage in {"all", "ood_temporal"}:
        temp_ood_df, temp_ood_json = temporal_ood_experiment()
        temp_ood_df.to_csv(REPORT_DIR / "part1_ood_temporal.csv", index=False)
        (REPORT_DIR / "part1_ood_temporal.json").write_text(json.dumps(temp_ood_json, indent=2), encoding="utf-8")
        overall["ood_temporal"] = temp_ood_json
        print("\nTemporal OOD")
        print(temp_ood_df.to_string(index=False))

    if args.stage in {"all", "ood_static"}:
        static_ood_df, static_ood_json = static_ood_experiment()
        static_ood_df.to_csv(REPORT_DIR / "part1_ood_static.csv", index=False)
        (REPORT_DIR / "part1_ood_static.json").write_text(json.dumps(static_ood_json, indent=2), encoding="utf-8")
        overall["ood_static"] = static_ood_json
        print("\nStatic OOD")
        print(static_ood_df.to_string(index=False))

    if args.stage in {"all", "osm"}:
        osm_edge_df, osm_summary_df, osm_meta = osm_comparison_experiment(max_components=args.osm_max_components)
        osm_edge_df.to_csv(GRAPH_DIR / "osm_edge_comparison.csv", index=False)
        if not osm_edge_df.empty:
            supported = osm_edge_df[osm_edge_df["osm_supported"] == 1].copy()
            supported.to_csv(GRAPH_DIR / "edges_osm_supported.csv", index=False)
        osm_summary_df.to_csv(REPORT_DIR / "osm_comparison_summary.csv", index=False)
        (REPORT_DIR / "osm_comparison_summary.json").write_text(json.dumps(osm_meta, indent=2), encoding="utf-8")
        overall["osm_meta"] = osm_meta
        print("\nOSM meta")
        print(json.dumps(osm_meta, indent=2))
        if (REPORT_DIR / "osm_debug_component_summary.csv").exists():
            log("OSM debug component summary written to reports/osm_debug_component_summary.csv")
        if (REPORT_DIR / "osm_snap_diagnostics.csv").exists():
            log("OSM snap diagnostics written to reports/osm_snap_diagnostics.csv")
        if (REPORT_DIR / "osm_topology_status_summary.csv").exists():
            log("OSM topology status summary written to reports/osm_topology_status_summary.csv")
        if (REPORT_DIR / "osm_distance_bin_summary.csv").exists():
            log("OSM distance-bin summary written to reports/osm_distance_bin_summary.csv")
        if (REPORT_DIR / "osm_component_edge_summary.csv").exists():
            log("OSM component-edge summary written to reports/osm_component_edge_summary.csv")

    if args.stage in {"all", "same_route_osm"}:
        audit_df, audit_summary_df, audit_meta = same_route_real_axis_audit(
            min_distance_km=args.same_route_min_distance_km,
            max_edges=args.same_route_max_edges,
        )
        audit_df.to_csv(REPORT_DIR / "same_route_real_axis_audit.csv", index=False)
        audit_summary_df.to_csv(REPORT_DIR / "same_route_real_axis_summary.csv", index=False)
        (REPORT_DIR / "same_route_real_axis_meta.json").write_text(json.dumps(audit_meta, indent=2), encoding="utf-8")
        overall["same_route_real_axis_meta"] = audit_meta
        print("\nSame-route real-axis audit")
        if not audit_summary_df.empty:
            print(audit_summary_df.to_string(index=False))
        print(json.dumps(audit_meta, indent=2))

    if overall:
        (REPORT_DIR / "part1_extensions_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
