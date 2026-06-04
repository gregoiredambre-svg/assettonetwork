"""Utility helpers for interim-presentation benchmark scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble import StackedEnsemble, build_meta_feature_matrix, fit_climate_pc1
from evaluation import compute_metrics
import graph_model_temporal as temporal_model
import part1_extensions as ext_model

SEED = 42


def load_target_data(
    target: str,
    graph_variant: str = "full_refined",
    treatment_mode: str = "experiment",
) -> tuple[temporal_model.MultiTaskTemporalData, np.ndarray, dict[str, np.ndarray]]:
    requested = [temporal_model.TARGET_COL]
    if target not in requested:
        requested.append(target)
    data, _ = temporal_model.prepare_multitask_data(
        graph_variant=graph_variant,
        treatment_mode=treatment_mode,
        target_cols=requested,
    )
    target_index = data.target_cols.index(target)
    y_full = data.y[:, :, target_index]
    y_mask = data.y_mask[:, :, target_index]
    split_masks = {
        split_name: data.split_masks[split_name] & y_mask for split_name in ["train", "val", "test"]
    }
    return data, y_full, split_masks


def build_target_rows(
    data: temporal_model.MultiTaskTemporalData,
    target: str,
    split_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    future_col = f"{target}_t1"
    rows = data.panel.loc[data.panel[future_col].notna()].copy()
    year_index = {year: idx for idx, year in enumerate(data.years)}
    node_index = {node_id: idx for idx, node_id in enumerate(data.node_ids)}
    split_labels: list[str] = []
    year_orders: list[int] = []
    node_orders: list[int] = []
    for row in rows.itertuples(index=False):
        yi = year_index[int(row.YEAR)]
        ni = node_index[str(row.node_id)]
        year_orders.append(yi)
        node_orders.append(ni)
        if split_masks["train"][yi, ni]:
            split_labels.append("train")
        elif split_masks["val"][yi, ni]:
            split_labels.append("val")
        elif split_masks["test"][yi, ni]:
            split_labels.append("test")
        else:
            split_labels.append("drop")
    rows["split"] = split_labels
    rows["year_order"] = year_orders
    rows["node_order"] = node_orders
    rows["target_value"] = pd.to_numeric(rows[future_col], errors="coerce")
    rows = rows[rows["split"] != "drop"].copy().reset_index(drop=True)
    return rows


def fit_rf_predictions(
    rows: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, object]]:
    train = rows[rows["split"] == "train"].copy()
    val = rows[rows["split"] == "val"].copy()
    test = rows[rows["split"] == "test"].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[feature_cols].to_numpy()))
    x_val = scaler.transform(imputer.transform(val[feature_cols].to_numpy()))
    x_test = scaler.transform(imputer.transform(test[feature_cols].to_numpy()))

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(x_train, train["target_value"].to_numpy(dtype=float))

    train_pred = rf.predict(x_train)
    val_pred = rf.predict(x_val)
    test_pred = rf.predict(x_test)

    out = rows.copy()
    out.loc[out["split"] == "train", "rf_pred"] = train_pred
    out.loc[out["split"] == "val", "rf_pred"] = val_pred
    out.loc[out["split"] == "test", "rf_pred"] = test_pred

    metrics = {
        "train": compute_metrics(train["target_value"].to_numpy(dtype=float), train_pred),
        "val": compute_metrics(val["target_value"].to_numpy(dtype=float), val_pred),
        "test": compute_metrics(test["target_value"].to_numpy(dtype=float), test_pred),
    }
    bundle = {"model": rf, "imputer": imputer, "scaler": scaler, "feature_cols": feature_cols}
    return out, metrics, bundle


def _predict_rgcn_cube(model: torch.nn.Module, x: np.ndarray, relation_adjs: list[np.ndarray]) -> np.ndarray:
    adjs = [torch.tensor(adj, dtype=torch.float32) for adj in relation_adjs]
    preds = []
    model.eval()
    with torch.no_grad():
        for yi in range(x.shape[0]):
            x_t = torch.tensor(x[yi], dtype=torch.float32)
            preds.append(model(x_t, adjs).cpu().numpy())
    return np.stack(preds, axis=0)


def train_single_target_rgcn(
    data: temporal_model.MultiTaskTemporalData,
    target: str,
    y_full: np.ndarray,
    split_masks: dict[str, np.ndarray],
    graph_variant: str = "full_refined",
    edges_df: pd.DataFrame | None = None,
    max_epochs: int = 180,
    patience: int = 20,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, object]]:
    train_vals = y_full[split_masks["train"]]
    if train_vals.size < 50:
        raise ValueError(f"Not enough train samples for {target}: {train_vals.size}")
    mu = float(np.mean(train_vals))
    sigma = float(np.std(train_vals))
    if sigma <= 1e-8:
        sigma = 1.0
    y_z = (y_full - mu) / sigma
    relation_names, relation_adjs = ext_model.build_relation_adjacencies(data.node_ids, graph_variant, edges_df=edges_df)
    model, _ = ext_model.train_temporal_rgcn(
        data.x_with_maint,
        y_z,
        split_masks,
        relation_adjs,
        max_epochs=max_epochs,
        patience=patience,
    )
    pred_z = _predict_rgcn_cube(model, data.x_with_maint, relation_adjs)
    pred = pred_z * sigma + mu

    rows = build_target_rows(data, target, split_masks)
    rows["rgcn_pred"] = [pred[int(r.year_order), int(r.node_order)] for r in rows.itertuples(index=False)]
    metrics = {
        split: compute_metrics(
            rows.loc[rows["split"] == split, "target_value"].to_numpy(dtype=float),
            rows.loc[rows["split"] == split, "rgcn_pred"].to_numpy(dtype=float),
        )
        for split in ["train", "val", "test"]
    }
    bundle = {
        "model": model,
        "relation_names": relation_names,
        "relation_adjs": relation_adjs,
        "z_mu": mu,
        "z_sigma": sigma,
        "hidden_dim": 64,
        "graph_variant": graph_variant,
    }
    return rows, metrics, bundle


def predict_single_target_rgcn_from_checkpoint(
    data: temporal_model.MultiTaskTemporalData,
    target: str,
    checkpoint_path: Path,
    split_masks: dict[str, np.ndarray],
    graph_variant: str = "full_refined",
    edges_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    relation_names, relation_adjs = ext_model.build_relation_adjacencies(data.node_ids, graph_variant, edges_df=edges_df)
    if list(payload["relation_names"]) != list(relation_names):
        raise ValueError("Checkpoint relation order does not match current graph build.")
    model = ext_model.RelationSnapshotGCN(
        input_dim=len(data.local_feature_cols),
        num_relations=len(relation_names),
        hidden_dim=int(payload.get("hidden_dim", 64)),
        dropout=0.2,
    )
    model.load_state_dict(payload["state_dict"])
    pred_z = _predict_rgcn_cube(model, data.x_with_maint, relation_adjs)
    pred = pred_z * float(payload["z_sigma"]) + float(payload["z_mu"])

    rows = build_target_rows(data, target, split_masks)
    rows["rgcn_pred"] = [pred[int(r.year_order), int(r.node_order)] for r in rows.itertuples(index=False)]
    metrics = {
        split: compute_metrics(
            rows.loc[rows["split"] == split, "target_value"].to_numpy(dtype=float),
            rows.loc[rows["split"] == split, "rgcn_pred"].to_numpy(dtype=float),
        )
        for split in ["train", "val", "test"]
    }
    bundle = {
        "model": model,
        "relation_names": relation_names,
        "relation_adjs": relation_adjs,
        "z_mu": float(payload["z_mu"]),
        "z_sigma": float(payload["z_sigma"]),
        "hidden_dim": int(payload.get("hidden_dim", 64)),
        "graph_variant": graph_variant,
    }
    return rows, metrics, bundle


def attach_meta_columns(
    data: temporal_model.MultiTaskTemporalData,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    out = rows.copy()
    climate_cols = [
        col for col in data.local_feature_cols if col.startswith(("humid_", "precip_", "wind_", "solar_", "temp_year_"))
    ]
    train_rows = out[out["split"] == "train"].copy()
    train_climate = train_rows[climate_cols].to_numpy(dtype=float) if climate_cols else np.empty((len(train_rows), 0))
    apply_climate = out[climate_cols].to_numpy(dtype=float) if climate_cols else np.empty((len(out), 0))
    _, climate_pc1 = fit_climate_pc1(train_climate, apply_climate)
    out["climate_pc1"] = climate_pc1

    if "years_since_last_treatment_event" in out.columns:
        node_age_proxy = pd.to_numeric(out["years_since_last_treatment_event"], errors="coerce")
    else:
        first_year = out.groupby("node_id")["YEAR"].transform("min")
        node_age_proxy = out["YEAR"] - first_year
    fallback_age = float(node_age_proxy.median()) if node_age_proxy.notna().any() else 0.0
    out["node_age_proxy"] = node_age_proxy.fillna(fallback_age)

    traffic_candidates = [
        "traffic_trf_trend_1_annual_truck_volume_trend",
        "traffic_trf_trend_1_aadtt_all_trucks_trend",
        "traffic_trf_trend_annual_esal_trend",
    ]
    traffic_col = next((col for col in traffic_candidates if col in out.columns), None)
    if traffic_col is None:
        out["traffic_log"] = 0.0
    else:
        traffic_values = pd.to_numeric(out[traffic_col], errors="coerce").fillna(0.0)
        out["traffic_log"] = np.log1p(traffic_values.clip(lower=0.0))
    return out


def fit_stacked_mlp(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows = attach_meta_columns(None if False else rows, rows)
    raise RuntimeError('Do not call fit_stacked_mlp directly; use fit_stacked_mlp_with_data.')


def fit_stacked_mlp_with_data(
    data: temporal_model.MultiTaskTemporalData,
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows = attach_meta_columns(data, rows)
    val_rows = rows[rows["split"] == "val"].copy()
    test_rows = rows[rows["split"] == "test"].copy()
    train_rows = rows[rows["split"] == "train"].copy()

    x_train_meta, feature_names = build_meta_feature_matrix(
        train_rows["rf_pred"].to_numpy(),
        train_rows["rgcn_pred"].to_numpy(),
        train_rows["node_age_proxy"].to_numpy(),
        train_rows["climate_pc1"].to_numpy(),
        train_rows["traffic_log"].to_numpy(),
    )
    x_val_meta, _ = build_meta_feature_matrix(
        val_rows["rf_pred"].to_numpy(),
        val_rows["rgcn_pred"].to_numpy(),
        val_rows["node_age_proxy"].to_numpy(),
        val_rows["climate_pc1"].to_numpy(),
        val_rows["traffic_log"].to_numpy(),
    )
    x_test_meta, _ = build_meta_feature_matrix(
        test_rows["rf_pred"].to_numpy(),
        test_rows["rgcn_pred"].to_numpy(),
        test_rows["node_age_proxy"].to_numpy(),
        test_rows["climate_pc1"].to_numpy(),
        test_rows["traffic_log"].to_numpy(),
    )

    ensemble = StackedEnsemble(meta_model="mlp", random_state=SEED).fit(
        x_val_meta,
        val_rows["target_value"].to_numpy(dtype=float),
        feature_names=feature_names,
    )
    rows = rows.copy()
    rows.loc[rows["split"] == "train", "ensemble_pred"] = ensemble.predict(x_train_meta)
    rows.loc[rows["split"] == "val", "ensemble_pred"] = ensemble.predict(x_val_meta)
    rows.loc[rows["split"] == "test", "ensemble_pred"] = ensemble.predict(x_test_meta)
    metrics = {
        split: compute_metrics(
            rows.loc[rows["split"] == split, "target_value"].to_numpy(dtype=float),
            rows.loc[rows["split"] == split, "ensemble_pred"].to_numpy(dtype=float),
        )
        for split in ["train", "val", "test"]
    }
    return rows, metrics
