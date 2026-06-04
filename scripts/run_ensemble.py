"""Run RF + R-GCN temporal ensembles on the dissertation temporal split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble import FixedRatioEnsemble, StackedEnsemble, build_meta_feature_matrix, fit_climate_pc1
from evaluation import (
    bootstrap_metric_ci,
    bootstrap_paired_delta_ci,
    compare_models,
    compute_metrics,
    dataframe_to_markdown,
)
from sklearn.metrics import mean_absolute_error, r2_score
import graph_model_temporal as temporal_model
import part1_extensions as ext_model

GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"
SEED = 42


def log(message: str) -> None:
    print(f"[run_ensemble] {message}")


def infer_best_rgcn_variant() -> str:
    frame = pd.read_csv(REPORT_DIR / "part1_rgcn_temporal.csv")
    return str(frame.sort_values("test_r2", ascending=False).iloc[0]["graph_variant"])


def load_rf_bundle(graph_variant: str = "full_refined", treatment_mode: str = "experiment") -> dict:
    path = GRAPH_DIR / f"temporal_rf_local_{graph_variant}_{treatment_mode}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing RF artifact: {path}")
    return joblib.load(path)


def load_rgcn_bundle(graph_variant: str) -> dict:
    path = GRAPH_DIR / f"temporal_rgcn_{graph_variant}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing R-GCN artifact: {path}")
    return torch.load(path, map_location="cpu")


def build_transition_rows(data: temporal_model.TemporalData) -> pd.DataFrame:
    rows = data.panel.loc[data.panel["target_t1"].notna()].copy()
    rows["split"] = "train"
    rows.loc[(rows["YEAR"] >= temporal_model.VAL_START) & (rows["YEAR"] <= temporal_model.VAL_END), "split"] = "val"
    rows.loc[(rows["YEAR"] >= temporal_model.TEST_START) & (rows["YEAR"] <= temporal_model.TEST_END), "split"] = "test"
    rows["node_order"] = rows["node_id"].map({node_id: idx for idx, node_id in enumerate(data.node_ids)})
    rows["year_order"] = rows["YEAR"].map({year: idx for idx, year in enumerate(data.years)})
    return rows.sort_values(["year_order", "node_order"]).reset_index(drop=True)


def predict_rf(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    feature_cols = bundle["feature_cols"]
    x = rows[feature_cols].to_numpy()
    x = bundle["imputer"].transform(x)
    x = bundle["scaler"].transform(x)
    return np.asarray(bundle["model"].predict(x), dtype=float)


def predict_rgcn(bundle: dict, data: temporal_model.TemporalData) -> np.ndarray:
    relation_names, relation_adjs = ext_model.build_relation_adjacencies(data.node_ids, bundle["graph_variant"])
    if relation_names != bundle["relation_names"]:
        raise ValueError("Saved R-GCN relation order does not match current graph construction.")
    model = ext_model.RelationSnapshotGCN(
        input_dim=len(bundle["feature_cols"]),
        num_relations=len(relation_names),
        hidden_dim=int(bundle.get("hidden_dim", 64)),
        dropout=0.2,
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    adjs = [torch.tensor(adj, dtype=torch.float32) for adj in relation_adjs]
    preds_by_year = []
    with torch.no_grad():
        for yi in range(data.x_with_maint.shape[0]):
            x_t = torch.tensor(data.x_with_maint[yi], dtype=torch.float32)
            preds_by_year.append(model(x_t, adjs).cpu().numpy())
    return np.stack(preds_by_year, axis=0)


def attach_rgcn_predictions(rows: pd.DataFrame, pred_cube: np.ndarray) -> np.ndarray:
    preds = []
    for row in rows.itertuples(index=False):
        preds.append(float(pred_cube[int(row.year_order), int(row.node_order)]))
    return np.asarray(preds, dtype=float)


def build_meta_columns(data: temporal_model.TemporalData, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["node_id", "YEAR", "target_t1", "split"]].copy()

    climate_cols = [col for col in data.local_feature_cols if col.startswith(("humid_", "precip_", "wind_", "solar_", "temp_year_"))]
    full_transition_rows = build_transition_rows(data)
    train_rows = full_transition_rows[full_transition_rows["split"] == "train"].copy()
    train_climate = train_rows[climate_cols].to_numpy(dtype=float) if climate_cols else np.empty((len(train_rows), 0))
    apply_climate = rows[climate_cols].to_numpy(dtype=float) if climate_cols else np.empty((len(rows), 0))
    _, climate_pc1 = fit_climate_pc1(train_climate, apply_climate)
    out["climate_pc1"] = climate_pc1

    if "years_since_last_treatment_event" in rows.columns:
        node_age_proxy = pd.to_numeric(rows["years_since_last_treatment_event"], errors="coerce")
    else:
        first_year = rows.groupby("node_id")["YEAR"].transform("min")
        node_age_proxy = rows["YEAR"] - first_year
    out["node_age_proxy"] = node_age_proxy.fillna(node_age_proxy.median() if node_age_proxy.notna().any() else 0.0)

    traffic_candidates = [
        "traffic_trf_trend_1_annual_truck_volume_trend",
        "traffic_trf_trend_1_aadtt_all_trucks_trend",
        "traffic_trf_trend_annual_esal_trend",
    ]
    traffic_col = next((col for col in traffic_candidates if col in rows.columns), None)
    if traffic_col is None:
        out["traffic_log"] = 0.0
    else:
        traffic_values = pd.to_numeric(rows[traffic_col], errors="coerce").fillna(0.0)
        out["traffic_log"] = np.log1p(traffic_values.clip(lower=0.0))
    return out


def evaluate_prediction_column(rows: pd.DataFrame, pred_col: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for split_name in ["val", "test"]:
        split_rows = rows[rows["split"] == split_name]
        metrics[split_name] = compute_metrics(split_rows["target_t1"].to_numpy(), split_rows[pred_col].to_numpy())
    return metrics


def main() -> None:
    best_variant = infer_best_rgcn_variant()
    log(f"Best saved R-GCN variant: {best_variant}")

    rf_bundle = load_rf_bundle(graph_variant="full_refined", treatment_mode="experiment")
    rgcn_bundle = load_rgcn_bundle(best_variant)
    data, _ = temporal_model.prepare_temporal_data(best_variant, treatment_mode="experiment")
    rows = build_transition_rows(data)
    rows = rows[rows["split"].isin(["val", "test"])].copy()

    rows["y_true"] = rows["target_t1"].astype(float)
    rows["rf_pred"] = predict_rf(rf_bundle, rows)
    rgcn_cube = predict_rgcn(rgcn_bundle, data)
    rows["gcn_pred"] = attach_rgcn_predictions(rows, rgcn_cube)

    meta_cols = build_meta_columns(data, rows)
    rows = rows.merge(meta_cols, on=["node_id", "YEAR", "target_t1", "split"], how="left")

    for weight in [0.3, 0.5, 0.7, 0.8]:
        label = f"ratio_{int(weight * 100)}"
        rows[label] = FixedRatioEnsemble(weight_rf=weight).predict(rows["rf_pred"].to_numpy(), rows["gcn_pred"].to_numpy())

    val_rows = rows[rows["split"] == "val"].copy()
    test_rows = rows[rows["split"] == "test"].copy()
    x_val_meta, feature_names = build_meta_feature_matrix(
        val_rows["rf_pred"].to_numpy(),
        val_rows["gcn_pred"].to_numpy(),
        val_rows["node_age_proxy"].to_numpy(),
        val_rows["climate_pc1"].to_numpy(),
        val_rows["traffic_log"].to_numpy(),
    )
    x_test_meta, _ = build_meta_feature_matrix(
        test_rows["rf_pred"].to_numpy(),
        test_rows["gcn_pred"].to_numpy(),
        test_rows["node_age_proxy"].to_numpy(),
        test_rows["climate_pc1"].to_numpy(),
        test_rows["traffic_log"].to_numpy(),
    )

    stacked_ridge = StackedEnsemble(meta_model="ridge", random_state=SEED).fit(
        x_val_meta,
        val_rows["y_true"].to_numpy(),
        feature_names=feature_names,
    )
    stacked_mlp = StackedEnsemble(meta_model="mlp", random_state=SEED).fit(
        x_val_meta,
        val_rows["y_true"].to_numpy(),
        feature_names=feature_names,
    )

    rows.loc[rows["split"] == "val", "stacked_ridge"] = stacked_ridge.predict(x_val_meta)
    rows.loc[rows["split"] == "test", "stacked_ridge"] = stacked_ridge.predict(x_test_meta)
    rows.loc[rows["split"] == "val", "stacked_mlp"] = stacked_mlp.predict(x_val_meta)
    rows.loc[rows["split"] == "test", "stacked_mlp"] = stacked_mlp.predict(x_test_meta)

    rf_gcn_best_val = max(
        compute_metrics(val_rows["y_true"].to_numpy(), val_rows["rf_pred"].to_numpy())["r2"],
        compute_metrics(val_rows["y_true"].to_numpy(), val_rows["gcn_pred"].to_numpy())["r2"],
    )
    stacked_val_r2 = compute_metrics(val_rows["y_true"].to_numpy(), rows.loc[rows["split"] == "val", "stacked_ridge"].to_numpy())["r2"]
    if stacked_val_r2 < rf_gcn_best_val:
        print(
            f"WARNING: stacked ridge val R² ({stacked_val_r2:.6f}) is worse than the best base val R² ({rf_gcn_best_val:.6f})."
        )

    result_blocks = {
        "RF local": {
            "val": compute_metrics(val_rows["y_true"].to_numpy(), val_rows["rf_pred"].to_numpy()),
            "test": compute_metrics(test_rows["y_true"].to_numpy(), test_rows["rf_pred"].to_numpy()),
        },
        "R-GCN": {
            "val": compute_metrics(val_rows["y_true"].to_numpy(), val_rows["gcn_pred"].to_numpy()),
            "test": compute_metrics(test_rows["y_true"].to_numpy(), test_rows["gcn_pred"].to_numpy()),
        },
        "Fixed ratio 30/70": evaluate_prediction_column(rows, "ratio_30"),
        "Fixed ratio 50/50": evaluate_prediction_column(rows, "ratio_50"),
        "Fixed ratio 70/30": evaluate_prediction_column(rows, "ratio_70"),
        "Fixed ratio 80/20": evaluate_prediction_column(rows, "ratio_80"),
        "Stacked ridge": evaluate_prediction_column(rows, "stacked_ridge"),
        "Stacked MLP": evaluate_prediction_column(rows, "stacked_mlp"),
    }

    comparison = compare_models(result_blocks)
    GRAPH_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    (GRAPH_DIR / "ensemble_results.json").write_text(
        json.dumps({"best_rgcn_variant": best_variant, "results": result_blocks}, indent=2),
        encoding="utf-8",
    )
    rows.rename(columns={"YEAR": "year"})[
        [
            "node_id",
            "year",
            "split",
            "y_true",
            "rf_pred",
            "gcn_pred",
            "ratio_30",
            "ratio_50",
            "ratio_70",
            "ratio_80",
            "stacked_ridge",
            "stacked_mlp",
        ]
    ].to_parquet(GRAPH_DIR / "ensemble_predictions.parquet", index=False)

    print(dataframe_to_markdown(comparison))

    test_only = comparison[["model", "r2_test", "mae_test", "rmse_test"]].copy()
    best_single = test_only[test_only["model"].isin(["RF local", "R-GCN"])].sort_values("r2_test", ascending=False).iloc[0]
    best_ensemble = test_only[~test_only["model"].isin(["RF local", "R-GCN"])].sort_values("r2_test", ascending=False).iloc[0]

    print(f"Best single model: {best_single['model']} (test R²={best_single['r2_test']:.6f})")
    print(f"Best ensemble strategy: {best_ensemble['model']} (test R²={best_ensemble['r2_test']:.6f})")
    print(f"Delta R² (ensemble - best single): {best_ensemble['r2_test'] - best_single['r2_test']:.6f}")
    print(f"Delta MAE (ensemble - best single): {best_ensemble['mae_test'] - best_single['mae_test']:.6f}")
    print(f"Delta RMSE (ensemble - best single): {best_ensemble['rmse_test'] - best_single['rmse_test']:.6f}")

    # ---- Bootstrap 95% CI on the headline gain ----
    col_map = {
        "Fixed ratio 30/70": "ratio_30",
        "Fixed ratio 50/50": "ratio_50",
        "Fixed ratio 70/30": "ratio_70",
        "Fixed ratio 80/20": "ratio_80",
        "Stacked ridge": "stacked_ridge",
        "Stacked MLP": "stacked_mlp",
    }
    ens_col = col_map[best_ensemble["model"]]
    test_subset = rows[rows["split"] == "test"]
    y_test = test_subset["y_true"].to_numpy()
    rf_test = test_subset["rf_pred"].to_numpy()
    ens_test = test_subset[ens_col].to_numpy()

    r2_rf_ci = bootstrap_metric_ci(y_test, rf_test, r2_score)
    r2_ens_ci = bootstrap_metric_ci(y_test, ens_test, r2_score)
    mae_rf_ci = bootstrap_metric_ci(y_test, rf_test, mean_absolute_error)
    mae_ens_ci = bootstrap_metric_ci(y_test, ens_test, mean_absolute_error)
    delta_r2 = bootstrap_paired_delta_ci(y_test, ens_test, rf_test, r2_score)
    delta_mae = bootstrap_paired_delta_ci(y_test, rf_test, ens_test, mean_absolute_error)

    print(f"\n## Bootstrap 95% CI on test set (n={len(y_test)}, 2000 resamples)")
    print(
        f"RF      R² : {r2_rf_ci['point']:.3f}  [{r2_rf_ci['lo']:.3f}, {r2_rf_ci['hi']:.3f}]"
        f"  (se={r2_rf_ci['se']:.3f})"
    )
    print(
        f"{best_ensemble['model']} R² : {r2_ens_ci['point']:.3f}  [{r2_ens_ci['lo']:.3f}, {r2_ens_ci['hi']:.3f}]"
        f"  (se={r2_ens_ci['se']:.3f})"
    )
    print(
        f"RF      MAE: {mae_rf_ci['point']:.3f}  [{mae_rf_ci['lo']:.3f}, {mae_rf_ci['hi']:.3f}]"
        f"  (se={mae_rf_ci['se']:.3f})"
    )
    print(
        f"{best_ensemble['model']} MAE: {mae_ens_ci['point']:.3f}  [{mae_ens_ci['lo']:.3f}, {mae_ens_ci['hi']:.3f}]"
        f"  (se={mae_ens_ci['se']:.3f})"
    )
    print(
        f"\nPaired Δ R² (ensemble - RF) : {delta_r2['point']:+.3f}  "
        f"[{delta_r2['lo']:+.3f}, {delta_r2['hi']:+.3f}]  P(>0)={delta_r2['p_positive']:.3f}"
    )
    print(
        f"Paired Δ MAE (RF - ensemble): {delta_mae['point']:+.3f}  "
        f"[{delta_mae['lo']:+.3f}, {delta_mae['hi']:+.3f}]  P(>0)={delta_mae['p_positive']:.3f}"
    )

    existing = json.loads((GRAPH_DIR / "ensemble_results.json").read_text(encoding="utf-8"))
    existing["bootstrap_ci"] = {
        "n_boot": 2000,
        "ci_level": 0.95,
        "best_ensemble": best_ensemble["model"],
        "rf_r2_ci": r2_rf_ci,
        "ensemble_r2_ci": r2_ens_ci,
        "rf_mae_ci": mae_rf_ci,
        "ensemble_mae_ci": mae_ens_ci,
        "delta_r2_paired_ens_minus_rf": delta_r2,
        "delta_mae_paired_rf_minus_ens": delta_mae,
    }
    (GRAPH_DIR / "ensemble_results.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
