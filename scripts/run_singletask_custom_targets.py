"""Train single-task R-GCN models for a custom list of distress targets."""

from __future__ import annotations

import argparse
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


def log(msg: str) -> None:
    print(f"[run_singletask_custom] {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-task R-GCN models for custom distress targets.")
    parser.add_argument("--graph-variant", default="full_refined", choices=["spatial", "spatial_route", "full_refined"])
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Target column names from ANALYSIS_DIS_AC.",
    )
    parser.add_argument(
        "--output-stem",
        default="singletask_custom_targets",
        help="Stem used for report and json outputs.",
    )
    return parser.parse_args()


def choose_transform(target_name: str, target_profile: pd.DataFrame) -> tuple[str, float | None]:
    row = target_profile.loc[target_profile["target"] == target_name]
    if row.empty:
        return "identity", None
    transform = str(row.iloc[0]["recommended_transform"])
    return transform, None


def apply_transform(
    y: np.ndarray,
    mask: np.ndarray,
    target_name: str,
    train_mask: np.ndarray,
    target_profile: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float | str]]:
    y = np.array(y, dtype=float, copy=True)
    transform, _ = choose_transform(target_name, target_profile)
    if transform.startswith("log1p"):
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
    if params["transform"] == "log1p_winsor99":
        cap = float(params["cap"])
        y = np.expm1(np.clip(y_t, 0.0, None))
        return np.clip(y, 0.0, cap)
    return y_t


def load_target_profile() -> pd.DataFrame:
    profile_path = REPORT_DIR / "distress_target_profile.csv"
    if profile_path.exists():
        return pd.read_csv(profile_path)
    raise FileNotFoundError(profile_path)


def main() -> None:
    args = parse_args()
    target_profile = load_target_profile()

    log("Preparing multitask data once to extract all requested targets...")
    data, prep = temporal_model.prepare_multitask_data(
        graph_variant=args.graph_variant,
        treatment_mode="experiment",
        target_cols=args.targets,
    )
    relation_names, relation_adjs = ext_model.build_relation_adjacencies(data.node_ids, args.graph_variant)
    split_masks = data.split_masks

    all_results: list[dict[str, object]] = []
    per_target_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for ti, target_name in enumerate(data.target_cols):
        log(f"=== Single-task training for {target_name} ===")
        y_full = data.y[:, :, ti]
        y_mask_full = data.y_mask[:, :, ti]

        train_mask_2d = split_masks["train"] & y_mask_full
        y_transformed, params = apply_transform(y_full, y_mask_full, target_name, train_mask_2d, target_profile)
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

        model, _ = ext_model.train_temporal_rgcn(
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

        profile_row = target_profile.loc[target_profile["target"] == target_name]
        coverage_pct = float(profile_row.iloc[0]["coverage_pct"]) if not profile_row.empty else np.nan
        zero_pct = float(profile_row.iloc[0]["zero_pct_observed"]) if not profile_row.empty else np.nan
        note = str(profile_row.iloc[0]["modelling_note"]) if not profile_row.empty else ""

        all_results.append(
            {
                "target": target_name,
                "transform": params["transform"],
                "winsor_cap": params.get("cap"),
                "coverage_pct": coverage_pct,
                "zero_pct_observed": zero_pct,
                "r2_train": results_for_target["train"]["r2"],
                "r2_val": results_for_target["val"]["r2"],
                "r2_test": results_for_target["test"]["r2"],
                "mae_test": results_for_target["test"]["mae"],
                "rmse_test": results_for_target["test"]["rmse"],
                "smape_test": results_for_target["test"]["smape"],
                "n_train": int(masks_for_train["train"].sum()),
                "n_val": int(masks_for_train["val"].sum()),
                "n_test": int(masks_for_train["test"].sum()),
                "modelling_note": note,
            }
        )

    df = pd.DataFrame(all_results).sort_values("r2_test", ascending=False).reset_index(drop=True)
    output = {
        "graph_variant": args.graph_variant,
        "requested_targets": args.targets,
        "kept_targets": data.target_cols,
        "relations": relation_names,
        "prep_info": prep,
        "results": all_results,
        "per_target_metrics": per_target_metrics,
    }
    (GRAPH_DIR / f"{args.output_stem}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (REPORT_DIR / f"{args.output_stem}.md").write_text(dataframe_to_markdown(df), encoding="utf-8")
    (REPORT_DIR / f"{args.output_stem}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(dataframe_to_markdown(df))


if __name__ == "__main__":
    main()
