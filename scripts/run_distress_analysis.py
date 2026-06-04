"""Analyse distress targets and compare single-task versus multi-task graph models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "Research Data"
GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"

TARGET_COLS = [
    "HPMS16_CRACKING_PERCENT_AC",
    "MEPDG_CRACKING_PERCENT_AC",
    "MEPDG_TRANS_CRACK_LENGTH_AC",
    "PATCH_A",
    "POTHOLES_A",
]

HEAVY_TAILED = {"MEPDG_TRANS_CRACK_LENGTH_AC", "PATCH_A", "POTHOLES_A"}
TARGET_LABELS = {
    "HPMS16_CRACKING_PERCENT_AC": "HPMS cracking (%)",
    "MEPDG_CRACKING_PERCENT_AC": "MEPDG cracking (%)",
    "MEPDG_TRANS_CRACK_LENGTH_AC": "Transverse cracking length",
    "PATCH_A": "Patched area",
    "POTHOLES_A": "Pothole area",
}
TARGET_NOTES = {
    "HPMS16_CRACKING_PERCENT_AC": "Stable headline cracking target with broad coverage; suitable for standard regression.",
    "MEPDG_CRACKING_PERCENT_AC": "Broader alligator-cracking measure; also strong enough for direct regression.",
    "MEPDG_TRANS_CRACK_LENGTH_AC": "Right-skewed length metric; needs log-style transform and careful error interpretation.",
    "PATCH_A": "Highly zero-inflated and heavy-tailed; regression alone is difficult and event-style framing may help.",
    "POTHOLES_A": "Rare-event distress with extreme sparsity; strongest candidate for two-stage or hurdle modelling.",
}


def log(message: str) -> None:
    print(f"[run_distress_analysis] {message}")


def load_target_metadata() -> pd.DataFrame:
    field_ref = pd.read_excel(DATA_DIR / "Analysis Ready Distress.xlsx", sheet_name="Field Reference")
    field_ref = field_ref[field_ref["TABLE_NAME"] == "ANALYSIS_DIS_AC"].copy()
    field_ref = field_ref[field_ref["FIELD_NAME"].isin(TARGET_COLS)].copy()
    field_ref = field_ref.rename(
        columns={
            "FIELD_NAME": "target",
            "FIELD_ALIAS": "field_alias",
            "FIELD_DESCRIPTION": "field_description",
            "FIELD_UNIT": "unit",
        }
    )
    return field_ref[["target", "field_alias", "field_description", "unit"]].drop_duplicates(subset=["target"])


def build_target_profile() -> pd.DataFrame:
    distress = pd.read_excel(DATA_DIR / "Analysis Ready Distress.xlsx", sheet_name="ANALYSIS_DIS_AC")
    distress["SURVEY_DATE"] = pd.to_datetime(distress["SURVEY_DATE"], errors="coerce")
    distress["YEAR"] = distress["SURVEY_DATE"].dt.year
    distress["node_id"] = distress[["STATE_CODE", "SHRP_ID"]].astype(str).agg("_".join, axis=1)
    metadata = load_target_metadata()

    rows: list[dict[str, object]] = []
    for target in TARGET_COLS:
        series = pd.to_numeric(distress[target], errors="coerce")
        obs = series.dropna()
        zeros = int((obs == 0).sum()) if not obs.empty else 0
        transform = "log1p_winsor99" if target in HEAVY_TAILED else "identity"
        row = {
            "target": target,
            "target_label": TARGET_LABELS[target],
            "recommended_transform": transform,
            "recommended_family": "single-task R-GCN" if target not in {"PATCH_A", "POTHOLES_A"} else "single-task R-GCN or two-stage model",
            "coverage_rows": int(obs.size),
            "coverage_pct": float(obs.size / len(distress) * 100.0),
            "zero_pct_observed": float((zeros / obs.size * 100.0) if obs.size else np.nan),
            "n_sections": int(distress.loc[series.notna(), "node_id"].nunique()),
            "n_years": int(distress.loc[series.notna(), "YEAR"].nunique()),
            "mean": float(obs.mean()) if not obs.empty else np.nan,
            "median": float(obs.median()) if not obs.empty else np.nan,
            "p90": float(obs.quantile(0.9)) if not obs.empty else np.nan,
            "p99": float(obs.quantile(0.99)) if not obs.empty else np.nan,
            "max": float(obs.max()) if not obs.empty else np.nan,
            "modelling_note": TARGET_NOTES[target],
        }
        rows.append(row)

    frame = pd.DataFrame(rows).merge(metadata, on="target", how="left")
    return frame


def build_model_comparison() -> pd.DataFrame:
    singletask_path = GRAPH_DIR / "singletask_per_distress_results.json"
    multitask_path = GRAPH_DIR / "multitask_results.json"
    if not singletask_path.exists():
        raise FileNotFoundError(singletask_path)
    singletask_payload = json.loads(singletask_path.read_text(encoding="utf-8"))
    singletask = pd.DataFrame(singletask_payload["results"])

    multitask_lookup: dict[str, dict[str, float]] = {}
    if multitask_path.exists():
        multitask_payload = json.loads(multitask_path.read_text(encoding="utf-8"))
        multitask_lookup = multitask_payload.get("metrics", {}).get("test", {})

    singletask["target_label"] = singletask["target"].map(TARGET_LABELS)
    singletask["multitask_r2_test"] = singletask["target"].map(lambda t: multitask_lookup.get(t, {}).get("r2"))
    singletask["multitask_mae_test"] = singletask["target"].map(lambda t: multitask_lookup.get(t, {}).get("mae"))
    singletask["multitask_rmse_test"] = singletask["target"].map(lambda t: multitask_lookup.get(t, {}).get("rmse"))
    singletask["delta_r2_vs_multitask"] = singletask["r2_test"] - singletask["multitask_r2_test"]
    singletask["delta_mae_vs_multitask"] = singletask["mae_test"] - singletask["multitask_mae_test"]
    singletask["delta_rmse_vs_multitask"] = singletask["rmse_test"] - singletask["multitask_rmse_test"]
    singletask["beats_multitask"] = singletask["delta_r2_vs_multitask"] > 0
    return singletask


def write_markdown(target_profile: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = [
        "# Distress Analysis",
        "",
        "## Target profiles",
        "",
        "| Target | Transform | Coverage % | Zero % | Median | P99 | Max | Modelling note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in target_profile.itertuples(index=False):
        lines.append(
            f"| {row.target_label} | {row.recommended_transform} | {row.coverage_pct:.1f} | "
            f"{row.zero_pct_observed:.1f} | {row.median:.3f} | {row.p99:.3f} | {row.max:.3f} | {row.modelling_note} |"
        )
    lines.extend(
        [
            "",
            "## Single-task versus multi-task",
            "",
            "| Target | Single-task test R² | Multi-task test R² | ΔR² | Transform | n_test |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in comparison.itertuples(index=False):
        mt = "" if pd.isna(row.multitask_r2_test) else f"{row.multitask_r2_test:.3f}"
        delta = "" if pd.isna(row.delta_r2_vs_multitask) else f"{row.delta_r2_vs_multitask:+.3f}"
        lines.append(
            f"| {row.target_label} | {row.r2_test:.3f} | {mt} | {delta} | {row.transform} | {int(row.n_test)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    GRAPH_DIR.mkdir(exist_ok=True)

    log("Building target profile summary...")
    target_profile = build_target_profile()
    log("Building single-task vs multi-task comparison...")
    comparison = build_model_comparison()

    (REPORT_DIR / "distress_target_profile.csv").write_text(target_profile.to_csv(index=False), encoding="utf-8")
    (REPORT_DIR / "distress_model_comparison.csv").write_text(comparison.to_csv(index=False), encoding="utf-8")
    (GRAPH_DIR / "distress_analysis.json").write_text(
        json.dumps(
            {
                "target_profile": target_profile.to_dict(orient="records"),
                "model_comparison": comparison.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown = write_markdown(target_profile, comparison)
    (REPORT_DIR / "distress_analysis.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
