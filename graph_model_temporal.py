"""Temporal graph model for section-level degradation forecasting.

Builds a (node_id, YEAR) panel aligned with the thesis RQ1 formulation:
- target: cracking(s, t+1)
- local features at t: traffic, climate, cracking(s, t), maintenance state
- graph propagation: neighbors influence through GCN convolutions

Outputs:
- models/gcn_temporal.pt
- reports/gcn_temporal_metrics.json
- reports/figures/temporal_predictions.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Research Data"
GRAPH_DIR = ROOT / "graph_data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

SEED = 42
TRAIN_END = 2015
VAL_START = 2016
VAL_END = 2018
TEST_START = 2019
TEST_END = 2021
TARGET_COL = "HPMS16_CRACKING_PERCENT_AC"
TREATMENT_GROUPS = {
    "crack_sealing": ["crack sealing", "joint sealing", "saw and seal"],
    "asphalt_overlay": ["overlay", "mill off ac and overlay", "mill existing pavement and overlay", "warm mix ac overlay"],
    "seal_coat": ["seal coat", "slurry seal", "fog seal", "surface treatment", "prime coat", "tack coat", "sand seal"],
    "patching": ["patch", "pothole", "spot patch", "skin patch", "strip patch"],
    "grinding": ["grinding", "grooving"],
    "shoulder_restoration": ["shoulder restoration", "shoulder replacement"],
    "reconstruction_or_major_rehab": ["reconstruction", "slab replacement", "fracture treatment", "load transfer restoration", "subdrain", "subdrainage", "drainage", "jacking", "subsealing"],
}


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(message: str) -> None:
    print(f"[gcn_temporal] {message}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def normalize_string_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": None})
    return df


def normalize_shrp_id_series(series: pd.Series) -> pd.Series:
    def _normalize(value: object) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        try:
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
        except Exception:
            pass
        stripped = text.lstrip("0")
        return stripped or "0"

    return series.map(_normalize)


def build_node_id(state_code: pd.Series, shrp_id: pd.Series) -> pd.Series:
    return state_code.astype(str).str.strip() + "_" + shrp_id.astype(str).str.strip()


def build_node_id_join(state_code: pd.Series, shrp_id: pd.Series) -> pd.Series:
    return state_code.astype(str).str.strip() + "_" + normalize_shrp_id_series(shrp_id).fillna("")


def build_node_id_join_scalar(node_id: object) -> str | None:
    text = str(node_id).strip()
    if not text or "_" not in text:
        return None
    state_code, shrp_id = text.split("_", 1)
    normalized = normalize_shrp_id_series(pd.Series([shrp_id])).iloc[0]
    if normalized is None:
        return None
    return f"{state_code.strip()}_{normalized}"


def load_base_nodes() -> pd.DataFrame:
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    nodes["node_id"] = nodes["node_id"].astype(str)
    if "node_id_join" not in nodes.columns:
        nodes["node_id_join"] = nodes["node_id"].map(build_node_id_join_scalar)
    nodes["NO_OF_LANES"] = pd.to_numeric(nodes.get("NO_OF_LANES"), errors="coerce")
    nodes["SECTION_LENGTH"] = pd.to_numeric(nodes.get("SECTION_LENGTH"), errors="coerce")
    nodes["SPEED_LIMIT"] = pd.to_numeric(nodes.get("SPEED_LIMIT"), errors="coerce")
    nodes["functional_class"] = pd.to_numeric(nodes.get("functional_class"), errors="coerce")
    climate_cols = [
        col
        for col in nodes.columns
        if col.startswith(("temp_bind_", "humid_", "precip_", "wind_", "solar_", "temp_year_"))
    ]
    keep = [
        "node_id",
        "node_id_join",
        "route_key",
        "functional_class",
        "NO_OF_LANES",
        "SECTION_LENGTH",
        "SPEED_LIMIT",
        "merra_id",
        *climate_cols,
    ]
    return nodes[keep].copy()


def filter_edges_for_variant(edges: pd.DataFrame, graph_variant: str) -> pd.DataFrame:
    variant_to_types = {
        "spatial": {"spatial"},
        "spatial_route": {"spatial", "same_route"},
        "full_refined": {"spatial", "same_route", "same_functional_class"},
    }
    return edges[edges["edge_type"].isin(variant_to_types[graph_variant])].copy()


def load_graph_adjacency(node_ids: list[str], graph_variant: str) -> np.ndarray:
    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    edges = filter_edges_for_variant(edges, graph_variant)
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
    weight_map = {"same_route": 1.0, "spatial": 0.7, "same_functional_class": 0.4}
    edges = edges[edges["source"].isin(index) & edges["target"].isin(index)].copy()
    edges["distance_km"] = pd.to_numeric(edges["distance_km"], errors="coerce").fillna(50.0)
    edges["diversion_potential"] = pd.to_numeric(edges.get("diversion_potential"), errors="coerce").fillna(0.0)
    for row in edges.itertuples(index=False):
        i = index[str(row.source)]
        j = index[str(row.target)]
        learned_weight = getattr(row, "weight_deterioration", np.nan)
        if pd.notna(learned_weight):
            weight = float(learned_weight)
        else:
            base = weight_map.get(str(row.edge_type), 0.2)
            closeness = 1.0 / (1.0 + float(row.distance_km))
            weight = base + 0.5 * float(row.diversion_potential) + 5.0 * closeness
        adjacency[i, j] = max(adjacency[i, j], weight)
        adjacency[j, i] = max(adjacency[j, i], weight)
    adjacency += np.eye(len(node_ids), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    normalized = adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
    return normalized.astype(np.float32)


def load_distress_ac() -> pd.DataFrame:
    path = DATA_DIR / "Analysis Ready Distress.xlsx"
    df = pd.read_excel(
        path,
        sheet_name="ANALYSIS_DIS_AC",
        usecols=["STATE_CODE", "SHRP_ID", "SURVEY_DATE", TARGET_COL],
    )
    df = df.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id"})
    normalize_string_columns(df, ["state_code", "shrp_id"])
    df["node_id"] = build_node_id(df["state_code"], df["shrp_id"])
    df["SURVEY_DATE"] = pd.to_datetime(df["SURVEY_DATE"], errors="coerce")
    df["YEAR"] = df["SURVEY_DATE"].dt.year.astype("Int64")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=["node_id", "YEAR"]).copy()
    agg = df.groupby(["node_id", "YEAR"], as_index=False)[TARGET_COL].mean()
    return agg


def load_traffic_panel() -> pd.DataFrame:
    path = DATA_DIR / "Annual Traffic Inputs Over Time.xlsx"
    specs = {
        "TRF_TREND": ["ANNUAL_ESAL_TREND", "ANNUAL_GESAL_TREND"],
        "TRF_TREND_1": ["AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND"],
        "TRF_TREND_2": [
            "AADTT_VEH_CLASS_4_TREND",
            "AADTT_VEH_CLASS_5_TREND",
            "AADTT_VEH_CLASS_6_TREND",
            "AADTT_VEH_CLASS_7_TREND",
            "AADTT_VEH_CLASS_8_TREND",
            "AADTT_VEH_CLASS_9_TREND",
            "AADTT_VEH_CLASS_10_TREND",
            "AADTT_VEH_CLASS_11_TREND",
            "AADTT_VEH_CLASS_12_TREND",
            "AADTT_VEH_CLASS_13_TREND",
            "CMLTV_VOL_VEH_CLASS_9_TREND",
        ],
    }
    parts: list[pd.DataFrame] = []
    for sheet, value_cols in specs.items():
        keep_cols = ["STATE_CODE", "SHRP_ID", "YEAR", *value_cols]
        df = pd.read_excel(path, sheet_name=sheet, usecols=keep_cols)
        df = df.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id"})
        normalize_string_columns(df, ["state_code", "shrp_id"])
        df["node_id"] = build_node_id(df["state_code"], df["shrp_id"])
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        rename = {col: f"traffic_{sheet.lower()}_{col.lower()}" for col in value_cols}
        df = df[["node_id", "YEAR", *value_cols]].rename(columns=rename)
        parts.append(df.groupby(["node_id", "YEAR"], as_index=False).mean())

    traffic = parts[0]
    for part in parts[1:]:
        traffic = traffic.merge(part, on=["node_id", "YEAR"], how="outer")
    return traffic


def load_annual_climate_panel() -> pd.DataFrame:
    climate_root = DATA_DIR / "MERRA - Temperature, Humidity, Precipitation, Wind, Solar"
    grid = pd.read_excel(climate_root / "GENERAL" / "MERRA_GRID_SECTION.xlsx", sheet_name="MERRA_GRID_SECTION")
    grid = grid.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id", "MERRA_ID": "merra_id"})
    normalize_string_columns(grid, ["state_code", "shrp_id", "merra_id"])
    grid["node_id_join"] = build_node_id_join(grid["state_code"], grid["shrp_id"])
    grid = grid[["node_id_join", "merra_id"]].drop_duplicates(subset=["node_id_join"])

    specs = [
        ("HUMIDITY/MERRA_HUMID_YEAR.xlsx", "MERRA_HUMID_YEAR", ["REL_HUM_AVG_AVG"], "humid_"),
        ("PRECIPITATION/MERRA_PRECIP_YEAR.xlsx", "MERRA_PRECIP_YEAR", ["PRECIPITATION", "EVAPORATION", "PRECIP_DAYS"], "precip_"),
        ("WIND/MERRA_WIND_YEAR.xlsx", "MERRA_WIND_YEAR", ["WIND_VELOCITY_AVG"], "wind_"),
        ("SOLAR/MERRA_SOLAR_YEAR.xlsx", "MERRA_SOLAR_YEAR", ["CLOUD_COVER_AVG", "SHORTWAVE_SURFACE_AVG"], "solar_"),
        ("TEMPERATURE/MERRA_TEMP_YEAR.xlsx", "MERRA_TEMP_YEAR", ["TEMP_AVG", "TEMP_MEAN_AVG", "FREEZE_INDEX", "FREEZE_THAW"], "temp_year_"),
    ]

    merged: pd.DataFrame | None = None
    for rel_path, sheet, value_cols, prefix in specs:
        df = pd.read_excel(climate_root / rel_path, sheet_name=sheet)
        df = df.rename(columns={"MERRA_ID": "merra_id"})
        normalize_string_columns(df, ["merra_id"])
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["merra_id", "YEAR", *value_cols]].copy()
        df = df.groupby(["merra_id", "YEAR"], as_index=False).mean()
        df = df.rename(columns={col: f"{prefix}{col.lower()}" for col in value_cols})
        merged = df if merged is None else merged.merge(df, on=["merra_id", "YEAR"], how="outer")

    assert merged is not None
    return grid.merge(merged, on="merra_id", how="left")


def load_monthly_climate_annual_features() -> pd.DataFrame:
    climate_root = DATA_DIR / "MERRA - Temperature, Humidity, Precipitation, Wind, Solar"
    grid = pd.read_excel(climate_root / "GENERAL" / "MERRA_GRID_SECTION.xlsx", sheet_name="MERRA_GRID_SECTION")
    grid = grid.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id", "MERRA_ID": "merra_id"})
    normalize_string_columns(grid, ["state_code", "shrp_id", "merra_id"])
    grid["node_id_join"] = build_node_id_join(grid["state_code"], grid["shrp_id"])
    grid = grid[["node_id_join", "merra_id"]].drop_duplicates(subset=["node_id_join"])

    def load_monthly(rel_path: str, sheet: str, cols: list[str]) -> pd.DataFrame:
        df = pd.read_excel(climate_root / rel_path, sheet_name=sheet)
        df = df.rename(columns={"MERRA_ID": "merra_id"})
        normalize_string_columns(df, ["merra_id"])
        keep = ["merra_id", "YEAR", "MONTH", *cols]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").astype("Int64")
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["merra_id", "YEAR", "MONTH"])

    def seasonal_mask(df: pd.DataFrame, months: set[int]) -> pd.Series:
        return df["MONTH"].isin(months)

    def summarize(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
        rows = []
        wet_threshold = None
        if value_col == "PRECIPITATION":
            wet_threshold = float(df[value_col].quantile(0.75)) if df[value_col].notna().any() else None
        for (merra_id, year), group in df.groupby(["merra_id", "YEAR"], dropna=True):
            values = group[value_col].dropna()
            if values.empty:
                continue
            row: dict[str, object] = {"merra_id": merra_id, "YEAR": int(year)}
            if prefix == "temp_avg":
                winter = group.loc[seasonal_mask(group, {12, 1, 2}), value_col].dropna()
                summer = group.loc[seasonal_mask(group, {6, 7, 8}), value_col].dropna()
                freeze_count = int((group.get("FREEZE_THAW", pd.Series(index=group.index, dtype=float)).fillna(0) > 0).sum())
                row.update(
                    {
                        "monthlyagg_temp_avg_min": float(values.min()),
                        "monthlyagg_temp_avg_max": float(values.max()),
                        "monthlyagg_temp_avg_range": float(values.max() - values.min()),
                        "monthlyagg_temp_avg_std": float(values.std(ddof=0)),
                        "monthlyagg_temp_avg_winter_mean": float(winter.mean()) if not winter.empty else np.nan,
                        "monthlyagg_temp_avg_summer_mean": float(summer.mean()) if not summer.empty else np.nan,
                        "monthlyagg_freeze_thaw_active_months": freeze_count,
                    }
                )
            elif prefix == "precipitation":
                winter = group.loc[seasonal_mask(group, {12, 1, 2}), value_col].dropna()
                summer = group.loc[seasonal_mask(group, {6, 7, 8}), value_col].dropna()
                wet_count = int((group[value_col].fillna(0) > (wet_threshold or 0.0)).sum())
                row.update(
                    {
                        "monthlyagg_precipitation_max": float(values.max()),
                        "monthlyagg_precipitation_std": float(values.std(ddof=0)),
                        "monthlyagg_precipitation_wet_month_count": wet_count,
                        "monthlyagg_precipitation_winter_sum": float(winter.sum()) if not winter.empty else np.nan,
                        "monthlyagg_precipitation_summer_sum": float(summer.sum()) if not summer.empty else np.nan,
                    }
                )
            elif prefix == "humidity":
                summer = group.loc[seasonal_mask(group, {6, 7, 8}), value_col].dropna()
                row.update(
                    {
                        "monthlyagg_humidity_max": float(values.max()),
                        "monthlyagg_humidity_std": float(values.std(ddof=0)),
                        "monthlyagg_humidity_summer_mean": float(summer.mean()) if not summer.empty else np.nan,
                    }
                )
            elif prefix == "wind":
                row.update(
                    {
                        "monthlyagg_wind_max": float(values.max()),
                        "monthlyagg_wind_std": float(values.std(ddof=0)),
                        "monthlyagg_wind_range": float(values.max() - values.min()),
                    }
                )
            elif prefix == "solar_shortwave":
                summer_sw = group.loc[seasonal_mask(group, {6, 7, 8}), "SHORTWAVE_SURFACE_AVG"].dropna()
                cloud_vals = group.get("CLOUD_COVER_AVG", pd.Series(index=group.index, dtype=float)).dropna()
                summer_cc = group.loc[seasonal_mask(group, {6, 7, 8}), "CLOUD_COVER_AVG"].dropna()
                row.update(
                    {
                        "monthlyagg_shortwave_surface_max": float(values.max()),
                        "monthlyagg_shortwave_surface_std": float(values.std(ddof=0)),
                        "monthlyagg_shortwave_surface_summer_mean": float(summer_sw.mean()) if not summer_sw.empty else np.nan,
                        "monthlyagg_cloud_cover_std": float(cloud_vals.std(ddof=0)) if not cloud_vals.empty else np.nan,
                        "monthlyagg_cloud_cover_summer_mean": float(summer_cc.mean()) if not summer_cc.empty else np.nan,
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows)

    temp_month = load_monthly("TEMPERATURE/MERRA_TEMP_MONTH.xlsx", "MERRA_TEMP_MONTH", ["TEMP_AVG", "FREEZE_THAW"])
    precip_month = load_monthly("PRECIPITATION/MERRA_PRECIP_MONTH.xlsx", "MERRA_PRECIP_MONTH", ["PRECIPITATION"])
    humid_month = load_monthly("HUMIDITY/MERRA_HUMID_MONTH.xlsx", "MERRA_HUMID_MONTH", ["REL_HUM_AVG_AVG"])
    wind_month = load_monthly("WIND/MERRA_WIND_MONTH.xlsx", "MERRA_WIND_MONTH", ["WIND_VELOCITY_AVG"])
    solar_month = load_monthly("SOLAR/MERRA_SOLAR_MONTH.xlsx", "MERRA_SOLAR_MONTH", ["SHORTWAVE_SURFACE_AVG", "CLOUD_COVER_AVG"])

    pieces = [
        summarize(temp_month, "TEMP_AVG", "temp_avg"),
        summarize(precip_month, "PRECIPITATION", "precipitation"),
        summarize(humid_month, "REL_HUM_AVG_AVG", "humidity"),
        summarize(wind_month, "WIND_VELOCITY_AVG", "wind"),
        summarize(solar_month, "SHORTWAVE_SURFACE_AVG", "solar_shortwave"),
    ]
    merged: pd.DataFrame | None = None
    for frame in pieces:
        merged = frame if merged is None else merged.merge(frame, on=["merra_id", "YEAR"], how="outer")
    assert merged is not None
    return grid.merge(merged, on="merra_id", how="left")


def classify_treatment_group(label: object) -> str:
    if pd.isna(label):
        return "unknown"
    text = str(label).strip().lower()
    if not text:
        return "unknown"
    for group, keywords in TREATMENT_GROUPS.items():
        if any(keyword in text for keyword in keywords):
            return group
    if text == "missing / not recorded":
        return "unknown"
    return "other_maintenance"


def load_experiment_treatment_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = DATA_DIR / "General Section Info.xlsx"
    df = pd.read_excel(path, sheet_name="EXPERIMENT_SECTION")
    df = df.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id"})
    normalize_string_columns(df, ["state_code", "shrp_id", "CN_CHANGE_REASON_EXP", "CN_CHANGE_REASON"])
    df["node_id"] = build_node_id(df["state_code"], df["shrp_id"])
    df["construction_no"] = pd.to_numeric(df.get("CONSTRUCTION_NO"), errors="coerce")
    df["event_start_date"] = pd.to_datetime(df.get("CN_ASSIGN_DATE"), errors="coerce")
    fallback_start = pd.to_datetime(df.get("ASSIGN_DATE"), errors="coerce")
    df["event_start_date"] = df["event_start_date"].fillna(fallback_start)
    df["event_end_date"] = pd.to_datetime(df.get("DEASSIGN_DATE"), errors="coerce")
    df["event_year"] = df["event_start_date"].dt.year.astype("Int64")
    df["treatment_code"] = df.get("CN_CHANGE_REASON")
    df["treatment_label"] = df.get("CN_CHANGE_REASON_EXP")
    df["broad_treatment_group"] = df["treatment_label"].map(classify_treatment_group)
    event_cols = [
        "node_id",
        "state_code",
        "shrp_id",
        "construction_no",
        "event_start_date",
        "event_end_date",
        "event_year",
        "treatment_code",
        "treatment_label",
        "broad_treatment_group",
    ]
    events = df[event_cols].copy()
    events = events.dropna(subset=["node_id", "event_year"]).copy()
    counts = (
        events["treatment_label"]
        .fillna("Missing / not recorded")
        .value_counts()
        .rename_axis("treatment_label")
        .reset_index(name="count")
    )
    return events, counts


def load_experiment_treatment_panel(node_ids: list[str], years: list[int], graph_variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events, category_counts = load_experiment_treatment_events()
    events = events[events["node_id"].isin(set(node_ids))].copy()
    idx = pd.MultiIndex.from_product([node_ids, years], names=["node_id", "YEAR"])
    out = pd.DataFrame(index=idx).reset_index()

    feature_defaults = {
        "had_treatment_event_t": 0,
        "years_since_last_treatment_event": np.nan,
        "treatment_count_last_3yr": 0,
        "treatment_count_last_5yr": 0,
        "had_crack_sealing_t": 0,
        "had_overlay_t": 0,
        "had_patching_t": 0,
        "had_surface_treatment_t": 0,
        "had_major_rehab_t": 0,
    }
    for col, default in feature_defaults.items():
        out[col] = default

    group_map = {
        "crack_sealing": "had_crack_sealing_t",
        "asphalt_overlay": "had_overlay_t",
        "patching": "had_patching_t",
        "seal_coat": "had_surface_treatment_t",
        "reconstruction_or_major_rehab": "had_major_rehab_t",
    }

    by_node = events.groupby("node_id")
    for node_id, group in by_node:
        years_for_node = sorted(group["event_year"].dropna().astype(int).tolist())
        counts3 = {year: int(sum(max(year - 2, -10_000) <= y <= year for y in years_for_node)) for year in years}
        counts5 = {year: int(sum(max(year - 4, -10_000) <= y <= year for y in years_for_node)) for year in years}
        group_year = group.groupby("event_year")
        for year in years:
            mask = (out["node_id"] == node_id) & (out["YEAR"] == year)
            year_events = group_year.get_group(year) if year in group_year.groups else None
            out.loc[mask, "had_treatment_event_t"] = int(year_events is not None and len(year_events) > 0)
            out.loc[mask, "treatment_count_last_3yr"] = counts3[year]
            out.loc[mask, "treatment_count_last_5yr"] = counts5[year]
            prior_years = [y for y in years_for_node if y <= year]
            if prior_years:
                out.loc[mask, "years_since_last_treatment_event"] = year - max(prior_years)
            if year_events is not None and len(year_events) > 0:
                groups_present = set(year_events["broad_treatment_group"].astype(str))
                for group_name, col in group_map.items():
                    out.loc[mask, col] = int(group_name in groups_present)

    adjacency = load_graph_adjacency(node_ids, graph_variant).copy()
    np.fill_diagonal(adjacency, 0.0)
    treatment_cols = [col for col in out.columns if col not in {"node_id", "YEAR"}]
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    year_index = {year: idx for idx, year in enumerate(years)}
    base = np.zeros((len(years), len(node_ids), len(treatment_cols)), dtype=np.float32)
    for row in out.itertuples(index=False):
        yi = year_index[int(row.YEAR)]
        ni = node_index[str(row.node_id)]
        base[yi, ni, :] = np.asarray([getattr(row, col) for col in treatment_cols], dtype=np.float32)

    neighbour_specs = {
        "neighbour_had_treatment_event_t": "had_treatment_event_t",
        "neighbour_treatment_count_last_3yr": "treatment_count_last_3yr",
        "neighbour_overlay_count_last_5yr": "had_overlay_t",
        "neighbour_crack_sealing_count_last_5yr": "had_crack_sealing_t",
    }
    for out_col, source_col in neighbour_specs.items():
        source_idx = treatment_cols.index(source_col)
        neighbour_values = np.einsum("ij,tj->ti", adjacency, base[:, :, source_idx], optimize=True)
        if out_col.endswith("_count_last_5yr") and source_col in {"had_overlay_t", "had_crack_sealing_t"}:
            rolling = np.zeros_like(neighbour_values)
            for yi, year in enumerate(years):
                start = max(0, yi - 4)
                rolling[yi] = neighbour_values[start : yi + 1].sum(axis=0)
            neighbour_values = rolling
        long = pd.DataFrame(
            {
                "node_id": np.repeat(node_ids, len(years)),
                "YEAR": np.tile(years, len(node_ids)),
                out_col: neighbour_values.T.reshape(-1),
            }
        )
        out = out.merge(long, on=["node_id", "YEAR"], how="left")

    semantics_rows = [
        {
            "feature_name": "had_treatment_event_t",
            "source_table": "EXPERIMENT_SECTION",
            "source_fields": "CN_ASSIGN_DATE, ASSIGN_DATE, DEASSIGN_DATE, CN_CHANGE_REASON, CN_CHANGE_REASON_EXP",
            "meaning": "At least one dated project/treatment event is recorded for this section in year t.",
        },
        {
            "feature_name": "years_since_last_treatment_event",
            "source_table": "EXPERIMENT_SECTION",
            "source_fields": "CN_ASSIGN_DATE / ASSIGN_DATE",
            "meaning": "Years since the last recorded project/treatment event for the section.",
        },
        {
            "feature_name": "treatment_count_last_3yr / treatment_count_last_5yr",
            "source_table": "EXPERIMENT_SECTION",
            "source_fields": "CN_ASSIGN_DATE / ASSIGN_DATE",
            "meaning": "Rolling count of dated section treatment events over the past 3 or 5 years.",
        },
        {
            "feature_name": "had_crack_sealing_t / had_overlay_t / had_patching_t / had_surface_treatment_t / had_major_rehab_t",
            "source_table": "EXPERIMENT_SECTION",
            "source_fields": "CN_CHANGE_REASON_EXP",
            "meaning": "Broad treatment categories derived from change-reason labels.",
        },
        {
            "feature_name": "neighbour_*",
            "source_table": "EXPERIMENT_SECTION + graph",
            "source_fields": "section treatment features aggregated through graph adjacency",
            "meaning": "Neighbourhood exposure to recent section project/treatment events.",
        },
    ]
    semantics = pd.DataFrame(semantics_rows)
    return out, category_counts, semantics


def build_temporal_panel(
    use_monthly_climate_aggregates: bool = False,
    treatment_mode: str = "experiment",
    graph_variant: str = "full_refined",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    nodes = load_base_nodes()
    distress = load_distress_ac().rename(columns={TARGET_COL: "cracking_t"})
    traffic = load_traffic_panel()
    climate = load_annual_climate_panel()
    monthly_climate = load_monthly_climate_annual_features() if use_monthly_climate_aggregates else None

    years = sorted(int(y) for y in distress["YEAR"].dropna().unique())
    treatment_semantics = None
    treatment_counts = None
    treatment_panel = None
    if treatment_mode == "experiment":
        treatment_panel, treatment_counts, treatment_semantics = load_experiment_treatment_panel(nodes["node_id"].tolist(), years, graph_variant)

    idx = pd.MultiIndex.from_product([nodes["node_id"].tolist(), years], names=["node_id", "YEAR"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(nodes, on="node_id", how="left")
    panel = panel.merge(distress, on=["node_id", "YEAR"], how="left")
    panel = panel.merge(traffic, on=["node_id", "YEAR"], how="left")
    panel = panel.merge(climate, on=["node_id_join", "YEAR"], how="left", suffixes=("", "_annual"))
    if monthly_climate is not None:
        panel = panel.merge(monthly_climate, on=["node_id_join", "YEAR"], how="left", suffixes=("", "_monthly"))
    if treatment_panel is not None:
        panel = panel.merge(treatment_panel, on=["node_id", "YEAR"], how="left")

    panel = panel.sort_values(["node_id", "YEAR"]).reset_index(drop=True)
    panel["target_t1"] = panel.groupby("node_id")["cracking_t"].shift(-1)
    panel["target_year"] = panel["YEAR"] + 1

    transition_mask = panel["target_t1"].notna()
    transitions = panel.loc[transition_mask].copy()
    return panel, transitions, treatment_counts, treatment_semantics


def build_feature_sets(panel: pd.DataFrame, treatment_mode: str) -> tuple[list[str], list[str], list[str]]:
    traffic_cols = [col for col in panel.columns if col.startswith("traffic_")]
    climate_cols = [col for col in panel.columns if col.startswith(("humid_", "precip_", "wind_", "solar_", "temp_year_"))]
    monthlyagg_cols = [col for col in panel.columns if col.startswith("monthlyagg_")]
    static_cols = ["NO_OF_LANES", "SECTION_LENGTH", "SPEED_LIMIT", "functional_class"]
    experiment_cols = [
        "had_treatment_event_t",
        "years_since_last_treatment_event",
        "treatment_count_last_3yr",
        "treatment_count_last_5yr",
        "had_crack_sealing_t",
        "had_overlay_t",
        "had_patching_t",
        "had_surface_treatment_t",
        "had_major_rehab_t",
        "neighbour_had_treatment_event_t",
        "neighbour_treatment_count_last_3yr",
        "neighbour_overlay_count_last_5yr",
        "neighbour_crack_sealing_count_last_5yr",
    ]
    treatment_cols: list[str] = []
    if treatment_mode == "experiment":
        treatment_cols = experiment_cols
    local_cols = [
        "cracking_t",
        *static_cols,
        *traffic_cols,
        *climate_cols,
        *monthlyagg_cols,
        *treatment_cols,
    ]
    local_cols = [col for col in local_cols if col in panel.columns]
    no_project_cols = [col for col in local_cols if col not in set(treatment_cols)]
    return local_cols, no_project_cols, static_cols


@dataclass
class TemporalData:
    node_ids: list[str]
    years: list[int]
    panel: pd.DataFrame
    local_feature_cols: list[str]
    no_maint_feature_cols: list[str]
    x_with_maint: np.ndarray
    x_without_maint: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    train_count: int
    val_count: int
    test_count: int
    treatment_mode: str
    treatment_category_counts: pd.DataFrame | None
    treatment_semantics: pd.DataFrame | None


def fit_imputer_scaler(train_2d: np.ndarray) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(train_2d)
    scaler.fit(x_imp)
    return imputer, scaler


def transform_3d(x: np.ndarray, imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    flat = imputer.transform(flat)
    flat = scaler.transform(flat)
    return flat.reshape(x.shape).astype(np.float32)


def prepare_temporal_data(
    graph_variant: str,
    use_monthly_climate_aggregates: bool = False,
    treatment_mode: str = "experiment",
) -> tuple[TemporalData, np.ndarray]:
    panel, transitions, treatment_counts, treatment_semantics = build_temporal_panel(
        use_monthly_climate_aggregates=use_monthly_climate_aggregates,
        treatment_mode=treatment_mode,
        graph_variant=graph_variant,
    )
    node_ids = sorted(panel["node_id"].unique().tolist())
    years = sorted(panel["YEAR"].unique().tolist())
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    year_index = {year: idx for idx, year in enumerate(years)}

    local_cols, no_maint_cols, _ = build_feature_sets(panel, treatment_mode=treatment_mode)
    y = np.full((len(years), len(node_ids)), np.nan, dtype=np.float32)
    mask = np.zeros((len(years), len(node_ids)), dtype=bool)
    x_with = np.full((len(years), len(node_ids), len(local_cols)), np.nan, dtype=np.float32)
    x_without = np.full((len(years), len(node_ids), len(no_maint_cols)), np.nan, dtype=np.float32)

    panel = panel.copy()
    for row in panel.itertuples(index=False):
        yi = year_index[int(row.YEAR)]
        ni = node_index[str(row.node_id)]
        x_with[yi, ni, :] = np.asarray([getattr(row, col) for col in local_cols], dtype=np.float32)
        x_without[yi, ni, :] = np.asarray([getattr(row, col) for col in no_maint_cols], dtype=np.float32)
        if not pd.isna(row.target_t1):
            y[yi, ni] = float(row.target_t1)
            mask[yi, ni] = True

    train_years = [year for year in years if year <= TRAIN_END]
    val_years = [year for year in years if VAL_START <= year <= VAL_END]
    test_years = [year for year in years if TEST_START <= year <= TEST_END]
    train_idx = [year_index[y] for y in train_years]
    val_idx = [year_index[y] for y in val_years]
    test_idx = [year_index[y] for y in test_years]

    train_with = x_with[train_idx].reshape(-1, x_with.shape[-1])
    train_without = x_without[train_idx].reshape(-1, x_without.shape[-1])
    imp_with, scl_with = fit_imputer_scaler(train_with)
    imp_without, scl_without = fit_imputer_scaler(train_without)
    x_with = transform_3d(x_with, imp_with, scl_with)
    x_without = transform_3d(x_without, imp_without, scl_without)

    adjacency = load_graph_adjacency(node_ids, graph_variant)
    data = TemporalData(
        node_ids=node_ids,
        years=years,
        panel=panel,
        local_feature_cols=local_cols,
        no_maint_feature_cols=no_maint_cols,
        x_with_maint=x_with,
        x_without_maint=x_without,
        y=y,
        mask=mask,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        train_count=int(mask[train_idx].sum()),
        val_count=int(mask[val_idx].sum()),
        test_count=int(mask[test_idx].sum()),
        treatment_mode=treatment_mode,
        treatment_category_counts=treatment_counts,
        treatment_semantics=treatment_semantics,
    )
    return data, adjacency


class SnapshotGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(a_hat, x)
        h = F.relu(self.lin1(h))
        h = self.dropout(h)
        h = torch.matmul(a_hat, h)
        h = F.relu(self.lin2(h))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


class GCNLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def encode_snapshot(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(a_hat, x)
        h = F.relu(self.lin1(h))
        h = self.dropout(h)
        h = torch.matmul(a_hat, h)
        h = F.relu(self.lin2(h))
        return self.dropout(h)

    def forward(self, x_seq: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        encoded = []
        for step in range(x_seq.shape[0]):
            encoded.append(self.encode_snapshot(x_seq[step], a_hat))
        seq = torch.stack(encoded, dim=1)  # nodes x seq x hidden
        out, _ = self.lstm(seq)
        return self.out(out[:, -1, :]).squeeze(-1)


def evaluate_years(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    years: list[int],
    year_index: dict[int, int],
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    a_t = torch.tensor(adjacency, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for year in years:
            yi = year_index[year]
            x_t = torch.tensor(x[yi], dtype=torch.float32)
            pred = model(x_t, a_t).cpu().numpy()
            year_mask = mask[yi]
            preds.append(pred[year_mask])
            trues.append(y[yi][year_mask])
    return np.concatenate(trues), np.concatenate(preds)


def train_snapshot_gcn(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    years: list[int],
    adjacency: np.ndarray,
) -> tuple[SnapshotGCN, dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    year_index = {year: idx for idx, year in enumerate(years)}
    train_years = [year for year in years if year <= TRAIN_END]
    val_years = [year for year in years if VAL_START <= year <= VAL_END]
    test_years = [year for year in years if TEST_START <= year <= TEST_END]

    model = SnapshotGCN(input_dim=x.shape[-1], hidden_dim=64, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    a_t = torch.tensor(adjacency, dtype=torch.float32)
    best_state = None
    best_val = float("inf")
    patience = 20
    no_improve = 0

    for epoch in range(1, 181):
        model.train()
        opt.zero_grad()
        losses = []
        for year in train_years:
            yi = year_index[year]
            if not mask[yi].any():
                continue
            pred = model(torch.tensor(x[yi], dtype=torch.float32), a_t)
            target = torch.tensor(y[yi], dtype=torch.float32)
            year_mask = torch.tensor(mask[yi], dtype=torch.bool)
            losses.append(F.mse_loss(pred[year_mask], target[year_mask]))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        val_true, val_pred = evaluate_years(model, x, y, mask, val_years, year_index, adjacency)
        val_loss = mean_squared_error(val_true, val_pred)
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 20 == 0:
            log(f"snapshot GCN epoch={epoch:03d} train_loss={loss.item():.4f} val_mse={val_loss:.4f}")
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {}
    for split_name, split_years in [("train", train_years), ("val", val_years), ("test", test_years)]:
        true, pred = evaluate_years(model, x, y, mask, split_years, year_index, adjacency)
        metrics[split_name] = metric_block(true, pred)
        if split_name == "test":
            test_true, test_pred = true, pred
    return model, metrics, test_true, test_pred


def build_sequence_end_years(years: list[int], mask: np.ndarray, year_index: dict[int, int]) -> list[int]:
    valid = []
    year_set = set(years)
    for year in years:
        if year - 1 not in year_set or year - 2 not in year_set:
            continue
        yi = year_index[year]
        if mask[yi].any():
            valid.append(year)
    return valid


def evaluate_sequence_years(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    years: list[int],
    year_index: dict[int, int],
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    end_years = build_sequence_end_years(years, mask, year_index)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    a_t = torch.tensor(adjacency, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for year in end_years:
            seq = np.stack([x[year_index[year - 2]], x[year_index[year - 1]], x[year_index[year]]], axis=0)
            pred = model(torch.tensor(seq, dtype=torch.float32), a_t).cpu().numpy()
            year_mask = mask[year_index[year]]
            preds.append(pred[year_mask])
            trues.append(y[year_index[year]][year_mask])
    if not preds:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(trues), np.concatenate(preds)


def train_gcn_lstm(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    years: list[int],
    adjacency: np.ndarray,
) -> tuple[GCNLSTM, dict[str, dict[str, float]]]:
    year_index = {year: idx for idx, year in enumerate(years)}
    train_years = [year for year in years if year <= TRAIN_END]
    val_years = [year for year in years if VAL_START <= year <= VAL_END]
    test_years = [year for year in years if TEST_START <= year <= TEST_END]
    train_seq = build_sequence_end_years(train_years, mask, year_index)

    model = GCNLSTM(input_dim=x.shape[-1], hidden_dim=64, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    a_t = torch.tensor(adjacency, dtype=torch.float32)
    best_state = None
    best_val = float("inf")
    patience = 20
    no_improve = 0

    for epoch in range(1, 161):
        model.train()
        opt.zero_grad()
        losses = []
        for year in train_seq:
            seq = np.stack([x[year_index[year - 2]], x[year_index[year - 1]], x[year_index[year]]], axis=0)
            pred = model(torch.tensor(seq, dtype=torch.float32), a_t)
            target = torch.tensor(y[year_index[year]], dtype=torch.float32)
            year_mask = torch.tensor(mask[year_index[year]], dtype=torch.bool)
            losses.append(F.mse_loss(pred[year_mask], target[year_mask]))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        val_true, val_pred = evaluate_sequence_years(model, x, y, mask, val_years, year_index, adjacency)
        val_loss = mean_squared_error(val_true, val_pred) if len(val_true) else float("inf")
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 20 == 0:
            log(f"GCN+LSTM epoch={epoch:03d} train_loss={loss.item():.4f} val_mse={val_loss:.4f}")
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {}
    for split_name, split_years in [("train", train_years), ("val", val_years), ("test", test_years)]:
        true, pred = evaluate_sequence_years(model, x, y, mask, split_years, year_index, adjacency)
        metrics[split_name] = metric_block(true, pred) if len(true) else {"rmse": None, "mae": None, "r2": None}
    return model, metrics


def build_tabular_rows(data: TemporalData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = data.panel.loc[data.panel["target_t1"].notna()].copy()
    rows["split"] = "train"
    rows.loc[(rows["YEAR"] >= VAL_START) & (rows["YEAR"] <= VAL_END), "split"] = "val"
    rows.loc[(rows["YEAR"] >= TEST_START) & (rows["YEAR"] <= TEST_END), "split"] = "test"
    train = rows[rows["split"] == "train"].copy()
    val = rows[rows["split"] == "val"].copy()
    test = rows[rows["split"] == "test"].copy()
    return train, val, test


def train_tabular_baselines(data: TemporalData) -> dict[str, dict[str, float]]:
    train, val, test = build_tabular_rows(data)
    feature_cols = data.local_feature_cols
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
    ridge.fit(x_train, y_train)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)

    out = {}
    for name, model in [("ridge_local", ridge), ("rf_local", rf)]:
        pred_train = model.predict(x_train)
        pred_val = model.predict(x_val)
        pred_test = model.predict(x_test)
        out[name] = {
            "train": metric_block(y_train, pred_train),
            "val": metric_block(y_val, pred_val),
            "test": metric_block(y_test, pred_test),
        }
    return out


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 900
    margin = 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    lim_min = float(min(np.min(y_true), np.min(y_pred)))
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    if lim_max <= lim_min:
        lim_max = lim_min + 1.0

    def project(value_x: float, value_y: float) -> tuple[float, float]:
        x_norm = (value_x - lim_min) / (lim_max - lim_min)
        y_norm = (value_y - lim_min) / (lim_max - lim_min)
        px = margin + x_norm * (width - 2 * margin)
        py = height - margin - y_norm * (height - 2 * margin)
        return px, py

    draw.rectangle((margin, margin, width - margin, height - margin), outline="black", width=2)
    line_start = project(lim_min, lim_min)
    line_end = project(lim_max, lim_max)
    draw.line((line_start[0], line_start[1], line_end[0], line_end[1]), fill="black", width=2)

    for true_value, pred_value in zip(y_true, y_pred):
        px, py = project(float(true_value), float(pred_value))
        r = 3
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(31, 119, 180), outline=None)

    draw.text((margin, 25), "Temporal GCN test predictions", fill="black")
    draw.text((margin, height - margin + 20), "True cracking(t+1)", fill="black")
    draw.text((20, margin - 25), "Predicted cracking(t+1)", fill="black")
    draw.text((margin, height - margin + 40), f"min={lim_min:.2f}  max={lim_max:.2f}", fill="black")
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the temporal section-level GCN.")
    parser.add_argument(
        "--graph-variant",
        choices=["spatial", "spatial_route", "full_refined"],
        default="full_refined",
    )
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--use-monthly-climate-aggregates", action="store_true")
    parser.add_argument(
        "--treatment-mode",
        choices=["none", "experiment"],
        default="experiment",
    )
    return parser.parse_args()


def tag_path(base_dir: Path, stem: str, output_tag: str, suffix: str) -> Path:
    name = f"{stem}_{output_tag}" if output_tag else stem
    return base_dir / f"{name}{suffix}"


def main() -> None:
    args = parse_args()
    set_seed()
    MODEL_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    ablation_rows: list[dict[str, object]] = []
    ablation_json: dict[str, object] = {
        "target": TARGET_COL,
        "graph_variant": args.graph_variant,
        "use_monthly_climate_aggregates": bool(args.use_monthly_climate_aggregates),
        "variants": {},
    }
    treatment_semantics: pd.DataFrame | None = None
    treatment_category_counts: pd.DataFrame | None = None

    selected_summary = None
    selected_bundle = None

    for treatment_mode, output_tag, label in [
        ("none", "no_project_features", "No project/treatment features"),
        ("experiment", "experiment_section_treatment_features", "EXPERIMENT_SECTION treatment features"),
    ]:
        log(f"Running temporal treatment ablation: {treatment_mode}")
        data, adjacency = prepare_temporal_data(
            args.graph_variant,
            use_monthly_climate_aggregates=args.use_monthly_climate_aggregates,
            treatment_mode=treatment_mode,
        )
        baseline_metrics = train_tabular_baselines(data)
        gcn_with_project, gcn_with_metrics, y_test_with, pred_test_with = train_snapshot_gcn(
            data.x_with_maint, data.y, data.mask, data.years, adjacency
        )
        gcn_without_project, gcn_without_metrics, _, _ = train_snapshot_gcn(
            data.x_without_maint, data.y, data.mask, data.years, adjacency
        )
        gcn_lstm, gcn_lstm_metrics = train_gcn_lstm(
            data.x_with_maint, data.y, data.mask, data.years, adjacency
        )

        summary = {
            "target": TARGET_COL,
            "graph_variant": args.graph_variant,
            "treatment_mode": treatment_mode,
            "use_monthly_climate_aggregates": bool(args.use_monthly_climate_aggregates),
            "split": {"train_end": TRAIN_END, "val_start": VAL_START, "val_end": VAL_END, "test_start": TEST_START, "test_end": TEST_END},
            "transition_counts": {
                "train": data.train_count,
                "val": data.val_count,
                "test": data.test_count,
            },
            "feature_counts": {
                "gcn_with_project_treatment": len(data.local_feature_cols),
                "gcn_without_project_treatment": len(data.no_maint_feature_cols),
            },
            "models": {
                "rf_local": baseline_metrics["rf_local"],
                "ridge_local": baseline_metrics["ridge_local"],
                "gcn_without_project_treatment": gcn_without_metrics,
                "gcn_with_project_treatment": gcn_with_metrics,
                "gcn_without_maint": gcn_without_metrics,
                "gcn_with_maint": gcn_with_metrics,
                "gcn_lstm_with_project_treatment": gcn_lstm_metrics,
                "gcn_lstm_with_maint": gcn_lstm_metrics,
            },
            "r2_gain_gcn_with_vs_without_project_treatment": float(
                gcn_with_metrics["test"]["r2"] - gcn_without_metrics["test"]["r2"]
            ),
        }
        summary["r2_gain_gcn_with_vs_without_maint"] = summary["r2_gain_gcn_with_vs_without_project_treatment"]
        metrics_path = tag_path(REPORT_DIR, "gcn_temporal_metrics", output_tag if not args.output_tag else f"{output_tag}_{args.output_tag}", ".json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        if treatment_mode == args.treatment_mode:
            figure_path = tag_path(FIGURE_DIR, "temporal_predictions", args.output_tag, ".png")
            plot_predictions(y_test_with, pred_test_with, figure_path)
            torch.save(
                {
                    "gcn_with_project_treatment_state_dict": gcn_with_project.state_dict(),
                    "gcn_without_project_treatment_state_dict": gcn_without_project.state_dict(),
                    "gcn_lstm_state_dict": gcn_lstm.state_dict(),
                    "feature_cols_with_project_treatment": data.local_feature_cols,
                    "feature_cols_without_project_treatment": data.no_maint_feature_cols,
                    "node_ids": data.node_ids,
                    "years": data.years,
                    "treatment_mode": treatment_mode,
                },
                tag_path(MODEL_DIR, "gcn_temporal", args.output_tag, ".pt"),
            )
            selected_summary = summary
            selected_bundle = (baseline_metrics, gcn_without_metrics, gcn_with_metrics)

        ablation_rows.append(
            {
                "variant": label,
                "treatment_mode": treatment_mode,
                "rf_test_rmse": baseline_metrics["rf_local"]["test"]["rmse"],
                "rf_test_mae": baseline_metrics["rf_local"]["test"]["mae"],
                "rf_test_r2": baseline_metrics["rf_local"]["test"]["r2"],
                "ridge_test_r2": baseline_metrics["ridge_local"]["test"]["r2"],
                "gcn_without_project_treatment_test_rmse": gcn_without_metrics["test"]["rmse"],
                "gcn_without_project_treatment_test_mae": gcn_without_metrics["test"]["mae"],
                "gcn_without_project_treatment_test_r2": gcn_without_metrics["test"]["r2"],
                "gcn_with_project_treatment_test_rmse": gcn_with_metrics["test"]["rmse"],
                "gcn_with_project_treatment_test_mae": gcn_with_metrics["test"]["mae"],
                "gcn_with_project_treatment_test_r2": gcn_with_metrics["test"]["r2"],
                "gcn_project_treatment_r2_gain": summary["r2_gain_gcn_with_vs_without_project_treatment"],
                "train_transitions": data.train_count,
                "val_transitions": data.val_count,
                "test_transitions": data.test_count,
            }
        )
        ablation_json["variants"][treatment_mode] = summary
        if data.treatment_semantics is not None:
            treatment_semantics = data.treatment_semantics
        if data.treatment_category_counts is not None:
            treatment_category_counts = data.treatment_category_counts

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(REPORT_DIR / "treatment_feature_ablation.csv", index=False)
    with open(REPORT_DIR / "treatment_feature_ablation.json", "w", encoding="utf-8") as fh:
        json.dump(ablation_json, fh, indent=2)
    if treatment_semantics is not None:
        treatment_semantics.to_csv(REPORT_DIR / "treatment_feature_semantics.csv", index=False)
    if treatment_category_counts is not None:
        treatment_category_counts.to_csv(REPORT_DIR / "treatment_feature_category_counts.csv", index=False)

    assert selected_summary is not None and selected_bundle is not None
    baseline_metrics, gcn_no_metrics, gcn_with_metrics = selected_bundle
    rows = [
        ("RF local", baseline_metrics["rf_local"]["test"]),
        ("Ridge local", baseline_metrics["ridge_local"]["test"]),
        ("GCN without project/treatment features", gcn_no_metrics["test"]),
        ("GCN with project/treatment features", gcn_with_metrics["test"]),
    ]
    print("Model,RMSE,MAE,R2")
    for name, metrics in rows:
        print(f"{name},{metrics['rmse']:.6f},{metrics['mae']:.6f},{metrics['r2']:.6f}")
    print(
        "R2 gain GCN with project/treatment - without project/treatment:",
        f"{selected_summary['r2_gain_gcn_with_vs_without_project_treatment']:.6f}",
    )
    print(
        "Transitions train/val/test:",
        selected_summary["transition_counts"]["train"],
        selected_summary["transition_counts"]["val"],
        selected_summary["transition_counts"]["test"],
    )


if __name__ == "__main__":
    main()
