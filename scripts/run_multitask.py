"""Multi-task R-GCN: joint prediction of 5 distress types."""

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

from evaluation import dataframe_to_markdown
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


def log(msg) -> None:
    """Emit a standard multi-task runner log line."""

    print(f"[run_multitask] {msg}")


def main() -> None:
    """Prepare data, train the multi-task R-GCN, evaluate it, and save artifacts."""

    log(f"Preparing multi-task data with {len(TARGET_COLS)} targets ...")
    data, prep_info = temporal_model.prepare_multitask_data(
        graph_variant="full_refined",
        treatment_mode="experiment",
        target_cols=TARGET_COLS,
    )
    log(f"n_nodes={len(data.node_ids)}, n_years={len(data.years)}, targets kept after coverage check: {data.target_cols}")
    log(f"Coverage per target (train/val/test): {prep_info.get('coverage_summary', {})}")

    relation_names, relation_adjs = ext_model.build_relation_adjacencies(data.node_ids, "full_refined")

    log("Training multi-task R-GCN ...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, best_val, full_metrics = ext_model.train_multitask_rgcn(
        data.x_with_maint,
        data.y,
        data.y_mask,
        data.split_masks,
        relation_adjs,
        data.target_means,
        data.target_stds,
    )
    log(f"Best validation z-MSE: {best_val:.6f}")

    log("Evaluating ...")
    adjs = [torch.tensor(a, dtype=torch.float32) for a in relation_adjs]
    metrics = ext_model.evaluate_multitask_model(
        model,
        data.x_with_maint,
        data.y,
        data.y_mask,
        data.split_masks,
        adjs,
        data.target_cols,
        data.target_means,
        data.target_stds,
    )

    GRAPH_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "target_cols": data.target_cols,
            "target_means": data.target_means,
            "target_stds": data.target_stds,
            "relation_names": relation_names,
            "graph_variant": "full_refined",
            "hidden_dim": 64,
            "feature_cols": data.local_feature_cols if hasattr(data, "local_feature_cols") else [],
        },
        GRAPH_DIR / "temporal_rgcn_multitask.pt",
    )

    rows = []
    for target_col in data.target_cols:
        row = {"target": target_col}
        for split in ("train", "val", "test"):
            row[f"r2_{split}"] = metrics[split][target_col]["r2"]
        row["mae_test"] = metrics["test"][target_col]["mae"]
        row["rmse_test"] = metrics["test"][target_col]["rmse"]
        row["smape_test"] = metrics["test"][target_col]["smape"]
        row["n_test"] = metrics["test"][target_col]["n_samples"]
        rows.append(row)

    macro = {"target": "MACRO_MEAN"}
    valid_targets = [target for target in data.target_cols if metrics["test"][target]["n_samples"] >= 30]
    for split in ("train", "val", "test"):
        macro[f"r2_{split}"] = float(np.mean([metrics[split][target]["r2"] for target in valid_targets]))
    macro["mae_test"] = float(np.mean([metrics["test"][target]["mae"] for target in valid_targets]))
    macro["rmse_test"] = float(np.mean([metrics["test"][target]["rmse"] for target in valid_targets]))
    macro["smape_test"] = float(np.mean([metrics["test"][target]["smape"] for target in valid_targets]))
    macro["n_test"] = int(sum(metrics["test"][target]["n_samples"] for target in data.target_cols))
    rows.append(macro)

    df = pd.DataFrame(rows)
    (GRAPH_DIR / "multitask_results.json").write_text(
        json.dumps({"metrics": metrics, "summary": rows, "prep_info": prep_info, "best_val_z_mse": best_val}, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "multitask_results.md").write_text(dataframe_to_markdown(df), encoding="utf-8")
    print(dataframe_to_markdown(df))

    test_r2_per_target = {target: metrics["test"][target]["r2"] for target in data.target_cols}
    best_target = max(test_r2_per_target, key=test_r2_per_target.get)
    worst_target = min(test_r2_per_target, key=test_r2_per_target.get)
    print(f"\nBest predicted distress: {best_target} (test R²={test_r2_per_target[best_target]:.3f})")
    print(f"Worst predicted distress: {worst_target} (test R²={test_r2_per_target[worst_target]:.3f})")
    print(f"Macro mean test R² across {len(data.target_cols)} targets: {macro['r2_test']:.3f}")


if __name__ == "__main__":
    main()
