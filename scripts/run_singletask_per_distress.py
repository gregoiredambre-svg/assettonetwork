"""Train one single-task R-GCN per distress type with target-specific transforms."""

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
import graph_model_temporal as temporal_model
import part1_extensions as ext_model

GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"
SEED = 42

TARGET_COLS = [
    "HPMS16_CRACKING_PERCENT_AC",
    "MEPDG_CRACKING_PERCENT_AC",
    "MEPDG_TRANS_CRACK_LENGTH_AC",
    "PATCH_A",
    "POTHOLES_A",
]

HEAVY_TAILED = {"MEPDG_TRANS_CRACK_LENGTH_AC", "PATCH_A", "POTHOLES_A"}


def log(msg: str) -> None:
    print(f"[run_singletask] {msg}")


def apply_transform(
    y: np.ndarray,
    mask: np.ndarray,
    target_name: str,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Apply a target-specific transform using train-only winsorisation when needed."""

    y = np.array(y, dtype=float, copy=True)
    if target_name in HEAVY_TAILED:
        train_values = y[train_mask & mask]
        train_values = train_values[~np.isnan(train_values)]
        train_values = train_values[train_values >= 0]
        if train_values.size == 0:
            cap = 1.0
        else:
            cap = float(np.percentile(train_values, 99.0))
        cap = max(cap, 1e-6)
        y_clipped = np.clip(y, 0.0, cap)
        y_t = np.log1p(y_clipped)
        return y_t, {"transform": "log1p_winsor99", "cap": cap}
    return y, {"transform": "identity"}


def inverse_transform(y_t: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    """Map transformed predictions back to the original target scale."""

    if params["transform"] == "log1p_winsor99":
        cap = float(params["cap"])
        y = np.expm1(np.clip(y_t, 0.0, None))
        return np.clip(y, 0.0, cap)
    return y_t


def main() -> None:
    log("Preparing multitask data once to extract all targets ...")
    data, _ = temporal_model.prepare_multitask_data(
        graph_variant="full_refined",
        treatment_mode="experiment",
        target_cols=TARGET_COLS,
    )
    relation_names, relation_adjs = ext_model.build_relation_adjacencies(data.node_ids, "full_refined")
    split_masks = data.split_masks

    all_results: list[dict[str, object]] = []
    per_target_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for ti, target_name in enumerate(data.target_cols):
        log(f"=== Single-task training for {target_name} ===")
        y_full = data.y[:, :, ti]
        y_mask_full = data.y_mask[:, :, ti]

        train_mask_2d = split_masks["train"] & y_mask_full
        y_transformed, params = apply_transform(y_full, y_mask_full, target_name, train_mask_2d)
        log(f"  transform: {params}")

        train_vals = y_transformed[train_mask_2d]
        if train_vals.size < 50:
            log(f"  SKIPPING (only {train_vals.size} train samples)")
            continue
        mu = float(np.mean(train_vals))
        sigma = float(np.std(train_vals))
        if sigma <= 1e-8:
            sigma = 1.0

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        y_z = (y_transformed - mu) / sigma

        masks_for_train = {
            "train": split_masks["train"] & y_mask_full,
            "val": split_masks["val"] & y_mask_full,
            "test": split_masks["test"] & y_mask_full,
        }

        model, _metrics_z = ext_model.train_temporal_rgcn(
            data.x_with_maint,
            y_z,
            masks_for_train,
            relation_adjs,
        )

        adjs = [torch.tensor(a, dtype=torch.float32) for a in relation_adjs]
        model.eval()
        preds_by_year: list[np.ndarray] = []
        with torch.no_grad():
            for yi in range(data.x_with_maint.shape[0]):
                x_t = torch.tensor(data.x_with_maint[yi], dtype=torch.float32)
                preds_by_year.append(model(x_t, adjs).cpu().numpy())
        pred_z = np.stack(preds_by_year, axis=0)
        pred_transformed = pred_z * sigma + mu
        pred_original = inverse_transform(pred_transformed, params)

        results_for_target: dict[str, dict[str, float]] = {}
        for split_name, mask in masks_for_train.items():
            true_vals = y_full[mask]
            pred_vals = pred_original[mask]
            results_for_target[split_name] = compute_metrics(true_vals, pred_vals)
        per_target_metrics[target_name] = results_for_target

        all_results.append(
            {
                "target": target_name,
                "transform": params["transform"],
                "winsor_cap": params.get("cap"),
                "r2_train": results_for_target["train"]["r2"],
                "r2_val": results_for_target["val"]["r2"],
                "r2_test": results_for_target["test"]["r2"],
                "mae_test": results_for_target["test"]["mae"],
                "rmse_test": results_for_target["test"]["rmse"],
                "smape_test": results_for_target["test"]["smape"],
                "n_train": int(masks_for_train["train"].sum()),
                "n_test": results_for_target["test"]["n_samples"],
            }
        )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "target_col": target_name,
                "transform": params["transform"],
                "winsor_cap": params.get("cap"),
                "z_mu": mu,
                "z_sigma": sigma,
                "relation_names": relation_names,
                "graph_variant": "full_refined",
                "hidden_dim": 64,
            },
            GRAPH_DIR / f"temporal_rgcn_singletask_{target_name}.pt",
        )

    multitask_path = GRAPH_DIR / "multitask_results.json"
    multitask_lookup: dict[str, dict[str, float]] = {}
    if multitask_path.exists():
        multitask_payload = json.loads(multitask_path.read_text(encoding="utf-8"))
        for tc, splits in multitask_payload.get("metrics", {}).get("test", {}).items():
            multitask_lookup[tc] = splits

    for row in all_results:
        mt = multitask_lookup.get(str(row["target"]), {})
        row["multitask_r2_test"] = mt.get("r2")
        row["delta_r2_vs_multitask"] = (
            float(row["r2_test"]) - float(mt["r2"])
            if mt and "r2" in mt and pd.notna(row["r2_test"])
            else None
        )

    df = pd.DataFrame(all_results)
    GRAPH_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    (GRAPH_DIR / "singletask_per_distress_results.json").write_text(
        json.dumps({"results": all_results, "per_target_metrics": per_target_metrics}, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "singletask_per_distress.md").write_text(dataframe_to_markdown(df), encoding="utf-8")
    print(dataframe_to_markdown(df))

    test_r2 = {str(row["target"]): float(row["r2_test"]) for row in all_results}
    best = max(test_r2, key=test_r2.get)
    worst = min(test_r2, key=test_r2.get)
    mean_r2 = float(np.mean(list(test_r2.values())))
    deltas = [
        f"{row['target']}: {row['delta_r2_vs_multitask']:+.3f}"
        for row in all_results
        if row.get("delta_r2_vs_multitask") is not None
    ]
    print(f"\nBest single-task target: {best} (R²={test_r2[best]:.3f})")
    print(f"Worst single-task target: {worst} (R²={test_r2[worst]:.3f})")
    print(f"Macro mean R² across {len(test_r2)} single-task targets: {mean_r2:.3f}")
    print(f"Δ vs multi-task (per-target deltas): {deltas}")


if __name__ == "__main__":
    main()
