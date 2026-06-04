"""Sensitivity analysis: do maintenance events show in annual traffic patterns?

The script tries to use a true AADT field first. If the annual traffic workbook
contains no overall-traffic AADT column, it falls back to the best available
annual traffic proxy and reports that choice explicitly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "Research Data" / "Annual Traffic Inputs Over Time.xlsx"
GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"

PRIMARY_CANDIDATES = [
    "AADT_ALL_VEHICS",
    "AADT_ALL_VEHICLES",
    "AADT_TOTAL",
    "AADT",
]
PROXY_CANDIDATES = [
    "ANNUAL_TRUCK_VOLUME_TREND",
    "AADTT_ALL_TRUCKS_TREND",
    "ANNUAL_GESAL_TREND",
    "ANNUAL_ESAL_TREND",
]


def log(msg: str) -> None:
    print(f"[traffic_sensitivity] {msg}")


def normalize_node_id(state_series: pd.Series, shrp_series: pd.Series) -> pd.Series:
    """Build node IDs in the same zero-padded format as graph exports."""

    state = pd.to_numeric(state_series, errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
    shrp = shrp_series.astype(str).str.strip()
    return state + "_" + shrp


def discover_metric_column(columns: list[str]) -> tuple[str | None, str]:
    """Return the preferred traffic column and whether it is exact or proxy."""

    normalized = {col.strip().upper(): col for col in columns}
    for name in PRIMARY_CANDIDATES:
        if name in normalized:
            return normalized[name], "aadt"
    for name in PRIMARY_CANDIDATES:
        hits = [col for upper, col in normalized.items() if name in upper and "TT" not in upper]
        if hits:
            return hits[0], "aadt"
    for name in PROXY_CANDIDATES:
        if name in normalized:
            return normalized[name], "proxy"
    return None, "missing"


def load_traffic_panel() -> tuple[pd.DataFrame, str, str]:
    """Load a section-year traffic panel using the best available annual metric."""

    log("Loading traffic panel from Excel ...")
    xls = pd.ExcelFile(DATA)
    sheets = [sheet for sheet in xls.sheet_names if "TRF_TREND" in sheet.upper()]
    log(f"Found sheets: {sheets}")

    chosen_metric = None
    chosen_kind = None
    frames: list[pd.DataFrame] = []
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = [str(col).strip() for col in df.columns]
        if not {"STATE_CODE", "SHRP_ID", "YEAR"}.issubset(df.columns):
            continue
        metric_col, metric_kind = discover_metric_column(df.columns.tolist())
        if metric_col is None:
            log(f"  Skip {sheet}: no annual AADT or traffic proxy column")
            continue
        if chosen_metric is None:
            chosen_metric = metric_col
            chosen_kind = metric_kind
            log(f"  Selected {metric_col} from {sheet} ({metric_kind})")
        if metric_col != chosen_metric:
            continue
        df["node_id"] = normalize_node_id(df["STATE_CODE"], df["SHRP_ID"])
        frame = df[["node_id", "YEAR", metric_col]].rename(columns={metric_col: "traffic_metric"})
        frames.append(frame)

    if not frames or chosen_metric is None or chosen_kind is None:
        raise ValueError("No usable AADT or annual traffic proxy column found in TRF_TREND sheets.")

    panel = pd.concat(frames, ignore_index=True)
    panel["YEAR"] = pd.to_numeric(panel["YEAR"], errors="coerce")
    panel["traffic_metric"] = pd.to_numeric(panel["traffic_metric"], errors="coerce")
    panel = panel.dropna(subset=["YEAR", "traffic_metric"]).copy()
    panel["YEAR"] = panel["YEAR"].astype(int)
    panel = panel.groupby(["node_id", "YEAR"], as_index=False)["traffic_metric"].mean()
    log(
        "  Traffic panel: "
        f"{len(panel)} section-year rows, {panel['node_id'].nunique()} sections, "
        f"years {panel['YEAR'].min()}-{panel['YEAR'].max()}"
    )
    return panel, chosen_metric, chosen_kind


def safe_mean(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.mean())


def safe_median(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.median())


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    projects = pd.read_csv(GRAPH_DIR / "projects.csv", low_memory=False)
    projects = projects[projects["event_year"].notna()].copy()
    projects["event_year"] = pd.to_numeric(projects["event_year"], errors="coerce")
    projects = projects.dropna(subset=["event_year"]).copy()
    projects["event_year"] = projects["event_year"].astype(int)

    # Collapse to one section-year event so repeated projects in the same year do not over-weight the signal.
    grouped = (
        projects.groupby(["node_id", "event_year"], as_index=False)
        .agg(
            n_projects=("project_id", "count"),
            broad_treatment_group=("broad_treatment_group", lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"),
            treatment_label=("treatment_label", lambda s: " | ".join(sorted({str(v) for v in s.dropna() if str(v).strip()}))[:500]),
            project_ids=("project_id", lambda s: " | ".join(sorted(s.astype(str).tolist())[:10])),
        )
    )
    log(f"Loaded {len(grouped)} section-year maintenance events across {grouped['node_id'].nunique()} sections")
    log(f"Treatment groups: {grouped['broad_treatment_group'].value_counts().to_dict()}")

    traffic_panel, metric_col, metric_kind = load_traffic_panel()
    traffic_panel = traffic_panel.set_index(["node_id", "YEAR"])
    traffic_dict = traffic_panel["traffic_metric"].to_dict()

    rows = []
    for ev in grouped.itertuples(index=False):
        node = str(ev.node_id)
        t = int(ev.event_year)
        pre = traffic_dict.get((node, t - 1))
        at = traffic_dict.get((node, t))
        post = traffic_dict.get((node, t + 1))
        post2 = traffic_dict.get((node, t + 2))
        rows.append(
            {
                "node_id": node,
                "event_year": t,
                "n_projects": int(ev.n_projects),
                "treatment_group": ev.broad_treatment_group,
                "treatment_label": ev.treatment_label,
                "project_ids": ev.project_ids,
                "traffic_pre": pre,
                "traffic_at": at,
                "traffic_post": post,
                "traffic_post2": post2,
                "delta_pre_to_event": (at - pre) if pre is not None and at is not None else None,
                "delta_pre_to_post": (post - pre) if pre is not None and post is not None else None,
                "delta_pre_to_post2": (post2 - pre) if pre is not None and post2 is not None else None,
                "rel_delta_pre_to_post": ((post - pre) / pre) if pre is not None and post is not None and pre > 0 else None,
            }
        )
    events_df = pd.DataFrame(rows)
    events_df = events_df.dropna(subset=["traffic_pre", "traffic_post"]).copy()
    log(f"Events with valid pre/post traffic metric: {len(events_df)}")

    summary_rows = []
    for group, sub in events_df.groupby("treatment_group"):
        deltas = sub["delta_pre_to_post"].dropna().to_numpy()
        rel = sub["rel_delta_pre_to_post"].dropna().to_numpy()
        if len(deltas) < 5:
            continue
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
        summary_rows.append(
            {
                "treatment_group": group,
                "n_events": int(len(deltas)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "mean_rel_delta": float(np.mean(rel)) if len(rel) else None,
                "median_rel_delta": float(np.median(rel)) if len(rel) else None,
                "t_stat_vs_zero": float(t_stat),
                "p_value_vs_zero": float(p_val),
            }
        )

    deltas_all = events_df["delta_pre_to_post"].dropna().to_numpy()
    rel_all = events_df["rel_delta_pre_to_post"].dropna().to_numpy()
    t_stat_all, p_val_all = stats.ttest_1samp(deltas_all, 0.0)
    summary_rows.append(
        {
            "treatment_group": "ALL",
            "n_events": int(len(deltas_all)),
            "mean_delta": float(np.mean(deltas_all)),
            "median_delta": float(np.median(deltas_all)),
            "mean_rel_delta": float(np.mean(rel_all)) if len(rel_all) else None,
            "median_rel_delta": float(np.median(rel_all)) if len(rel_all) else None,
            "t_stat_vs_zero": float(t_stat_all),
            "p_value_vs_zero": float(p_val_all),
        }
    )

    log("Building control sample ...")
    event_keys = {(str(row.node_id), int(row.event_year)) for row in grouped.itertuples(index=False)}
    traffic_reset = traffic_panel.reset_index()
    controls = []
    for node_id, sub in traffic_reset.groupby("node_id"):
        for year in sub["YEAR"].astype(int).tolist():
            if any((str(node_id), yy) in event_keys for yy in [year - 1, year, year + 1]):
                continue
            pre = traffic_dict.get((str(node_id), year - 1))
            post = traffic_dict.get((str(node_id), year + 1))
            if pre is None or post is None:
                continue
            controls.append(
                {
                    "node_id": str(node_id),
                    "year": int(year),
                    "traffic_pre": pre,
                    "traffic_post": post,
                    "delta_pre_to_post": post - pre,
                    "rel_delta_pre_to_post": ((post - pre) / pre) if pre > 0 else None,
                }
            )
    controls_df = pd.DataFrame(controls)
    log(f"Control sample size: {len(controls_df)}")

    event_deltas = events_df["delta_pre_to_post"].dropna()
    control_deltas = controls_df["delta_pre_to_post"].dropna()
    welch_t, welch_p = stats.ttest_ind(event_deltas, control_deltas, equal_var=False)

    summary = {
        "traffic_metric_column": metric_col,
        "traffic_metric_kind": metric_kind,
        "n_events_with_valid_pre_post": int(len(events_df)),
        "n_controls": int(len(controls_df)),
        "event_mean_delta": safe_mean(events_df["delta_pre_to_post"]),
        "control_mean_delta": safe_mean(controls_df["delta_pre_to_post"]),
        "event_mean_rel_delta": safe_mean(events_df["rel_delta_pre_to_post"]),
        "control_mean_rel_delta": safe_mean(controls_df["rel_delta_pre_to_post"]),
        "event_mean_delta_pre_to_event": safe_mean(events_df["delta_pre_to_event"]),
        "event_mean_delta_pre_to_post2": safe_mean(events_df["delta_pre_to_post2"]),
        "welch_t_stat_events_vs_controls": float(welch_t),
        "welch_p_value_events_vs_controls": float(welch_p),
        "per_group_summary": summary_rows,
    }

    (REPORT_DIR / "traffic_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    events_df.to_csv(REPORT_DIR / "traffic_sensitivity_events.csv", index=False)
    controls_df.to_csv(REPORT_DIR / "traffic_sensitivity_controls.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(REPORT_DIR / "traffic_sensitivity_per_group.csv", index=False)

    metric_label = "AADT" if metric_kind == "aadt" else f"traffic proxy ({metric_col})"
    print("\n## Traffic sensitivity to maintenance events")
    print(f"Metric used: {metric_label}")
    print(f"Events with valid pre/post metric: {len(events_df)}")
    print(f"Controls (no event in t-1..t+1):  {len(controls_df)}")
    print(
        f"\nEvent mean Δ (t+1 vs t-1):    {events_df['delta_pre_to_post'].mean():+.1f}  "
        f"({events_df['rel_delta_pre_to_post'].dropna().mean()*100:+.2f}%)"
    )
    print(
        f"Control mean Δ (t+1 vs t-1):  {controls_df['delta_pre_to_post'].mean():+.1f}  "
        f"({controls_df['rel_delta_pre_to_post'].dropna().mean()*100:+.2f}%)"
    )
    print(f"Mean Δ at event year (t vs t-1): {events_df['delta_pre_to_event'].mean():+.1f}")
    print(f"Mean Δ by t+2 (t+2 vs t-1):     {events_df['delta_pre_to_post2'].dropna().mean():+.1f}")
    print(f"Welch's t-test (events vs controls): t={welch_t:.2f}, p={welch_p:.4f}")
    if welch_p < 0.05:
        print("VERDICT: Statistically significant difference between maintenance events and controls.")
    else:
        print("VERDICT: No statistically significant difference between maintenance events and controls.")
    if metric_kind != "aadt":
        print(f"NOTE: No direct AADT column was available; this check used {metric_col} as the closest annual traffic proxy.")

    print("\n## Per-treatment-group breakdown:")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
