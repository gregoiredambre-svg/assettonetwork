"""Exploratory event-study analysis around EXPERIMENT_SECTION treatment/change events.

Purpose:
- understand how EXPERIMENT_SECTION treatment groups differ in pre/post distress, traffic, and climate
- compare annual vs monthly-derived climate exposure before events
- study neighbour changes around treatment events on the graph

Inputs:
- Research Data/General Section Info.xlsx (EXPERIMENT_SECTION, Codes Reference)
- Research Data/GENERAL/Fields.xlsx
- Research Data/GENERAL/Tables.xlsx
- Research Data/Analysis Ready Distress.xlsx
- Research Data/Annual Traffic Inputs Over Time.xlsx
- Research Data/MERRA - Temperature, Humidity, Precipitation, Wind, Solar/*
- graph_data/nodes.csv
- graph_data/edges.csv

Outputs:
- reports/experiment_event_study_by_event.csv
- reports/experiment_event_study_by_group.csv
- reports/experiment_event_study_climate_by_group.csv
- reports/experiment_event_study_climate_features.csv
- reports/event_study_monthly_vs_annual_climate_redundancy.csv
- reports/experiment_event_study_treatment_classifier.csv
- reports/experiment_event_study_feature_importance.csv
- reports/experiment_event_study_neighbour_climate_summary.csv
- reports/experiment_event_study_neighbour_vs_control.csv
- reports/experiment_event_study_interpretation.json

Run:
- ./.venv/bin/python experiment_event_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Research Data"
REPORT_DIR = ROOT / "reports"
GRAPH_DIR = ROOT / "graph_data"

STATE_NAMES = {
    "1": "Alabama",
    "2": "Alaska",
    "4": "Arizona",
    "5": "Arkansas",
    "6": "California",
    "8": "Colorado",
    "9": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
}

TREATMENT_GROUPS = {
    "crack_sealing": ["crack sealing", "joint sealing", "saw and seal"],
    "asphalt_overlay": [
        "asphalt concrete overlay",
        "overlay",
        "mill off ac and overlay",
        "mill existing pavement and overlay",
        "warm mix ac overlay",
        "recycled asphalt concrete overlay",
    ],
    "seal_coat": [
        "seal coat",
        "slurry seal",
        "fog seal",
        "surface treatment",
        "prime coat",
        "tack coat",
        "sand seal",
    ],
    "patching": ["patch", "pothole", "spot patch", "skin patch", "strip patch"],
    "grinding": ["grinding", "grooving"],
    "shoulder_restoration": ["shoulder restoration", "shoulder replacement"],
    "reconstruction_or_major_rehab": [
        "reconstruction",
        "slab replacement",
        "fracture treatment",
        "load transfer restoration",
        "subsealing",
        "jacking",
    ],
    "longitudinal_subdrains_or_drainage": ["subdrain", "subdrainage", "drainage blanket", "well system"],
}

ANNUAL_MEAN_COLS = {
    "humid_rel_hum_avg_avg",
    "wind_wind_velocity_avg",
    "solar_cloud_cover_avg",
    "solar_shortwave_surface_avg",
    "temp_year_temp_avg",
    "temp_year_temp_mean_avg",
}
ANNUAL_SUM_COLS = {
    "precip_precipitation",
    "precip_evaporation",
    "precip_precip_days",
    "temp_year_freeze_index",
    "temp_year_freeze_thaw",
}


def normalize_shrp_id(value: object) -> str | None:
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


def build_node_id(state_code: object, shrp_id: object) -> str:
    return f"{str(state_code).strip()}_{str(shrp_id).strip()}"


def build_node_id_join(state_code: object, shrp_id: object) -> str:
    return f"{str(state_code).strip()}_{normalize_shrp_id(shrp_id) or ''}"


def classify_treatment_group(label: object) -> str:
    if pd.isna(label):
        return "unknown"
    text = str(label).strip().lower()
    if not text or text == "missing / not recorded":
        return "unknown"
    for group, keywords in TREATMENT_GROUPS.items():
        if any(keyword in text for keyword in keywords):
            return group
    if "lane" in text or "widen" in text:
        return "widening_or_lane_change"
    if "experiment" in text or "out-of-study" in text or "status" in text:
        return "experiment_status_change"
    return "other_maintenance"


def month_distance(event_year: int, event_month: int, year: int, month: int) -> int:
    return (event_year - year) * 12 + (event_month - month)


def safe_iqr(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def load_experiment_events() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "General Section Info.xlsx", sheet_name="EXPERIMENT_SECTION")
    df["state_code"] = df["STATE_CODE"].astype(str).str.strip()
    df["shrp_id"] = df["SHRP_ID"].astype(str).str.strip()
    df["node_id"] = df.apply(lambda row: build_node_id(row["state_code"], row["shrp_id"]), axis=1)
    df["node_id_join"] = df.apply(lambda row: build_node_id_join(row["state_code"], row["shrp_id"]), axis=1)
    df["state_name"] = df["state_code"].map(lambda code: STATE_NAMES.get(code, f"State {code}"))
    for col in ["CN_CHANGE_REASON", "CN_CHANGE_REASON_EXP", "STATUS", "STATUS_EXP", "GPS_SPS", "GPS_SPS_EXP", "EXPERIMENT_NO", "EXPERIMENT_NO_EXP"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": None})
    df["event_start_date"] = pd.to_datetime(df["CN_ASSIGN_DATE"], errors="coerce").fillna(pd.to_datetime(df["ASSIGN_DATE"], errors="coerce"))
    df["event_end_date"] = pd.to_datetime(df["DEASSIGN_DATE"], errors="coerce")
    df["event_year"] = df["event_start_date"].dt.year.astype("Int64")
    df["event_month"] = df["event_start_date"].dt.month.astype("Int64")
    df["treatment_label"] = df["CN_CHANGE_REASON_EXP"]
    df["broad_treatment_group"] = df["treatment_label"].map(classify_treatment_group)
    keep_cols = [
        "node_id",
        "node_id_join",
        "state_code",
        "state_name",
        "shrp_id",
        "CONSTRUCTION_NO",
        "event_start_date",
        "event_end_date",
        "event_year",
        "event_month",
        "CN_CHANGE_REASON",
        "CN_CHANGE_REASON_EXP",
        "treatment_label",
        "broad_treatment_group",
        "EXPERIMENT_NO",
        "EXPERIMENT_NO_EXP",
        "STATUS",
        "STATUS_EXP",
        "GPS_SPS",
        "GPS_SPS_EXP",
    ]
    return df[keep_cols].rename(
        columns={
            "CONSTRUCTION_NO": "construction_no",
            "CN_CHANGE_REASON": "cn_change_reason",
            "CN_CHANGE_REASON_EXP": "cn_change_reason_exp",
            "EXPERIMENT_NO": "experiment_no",
            "EXPERIMENT_NO_EXP": "experiment_label",
            "STATUS": "status",
            "STATUS_EXP": "status_label",
            "GPS_SPS": "gps_sps",
            "GPS_SPS_EXP": "gps_sps_label",
        }
    )


def load_nodes() -> pd.DataFrame:
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["state_code"] = nodes["state_code"].astype(str).str.strip()
    if "node_id_join" not in nodes.columns:
        nodes["node_id_join"] = nodes["node_id"]
    return nodes


def load_neighbour_map() -> dict[str, set[str]]:
    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges = edges[edges["edge_type"].isin(["spatial", "same_route"])].copy()
    neighbours: dict[str, set[str]] = {}
    for row in edges.itertuples(index=False):
        neighbours.setdefault(row.source, set()).add(row.target)
        neighbours.setdefault(row.target, set()).add(row.source)
    return neighbours


def load_distress() -> pd.DataFrame:
    specs = [
        ("ANALYSIS_DIS_AC", "HPMS16_CRACKING_PERCENT_AC", "AC", []),
        ("ANALYSIS_DIS_CRCP", "HPMS16_CRACKING_PERCENT_CRCP", "CRCP", []),
        ("ANALYSIS_DIS_JPCC", "HPMS16_CRACKING_PERCENT_JPCC", "JPCC", ["AVG_EDGE_FAULT", "AVG_WHEELPATH_FAULT"]),
    ]
    parts: list[pd.DataFrame] = []
    for sheet, cracking_col, pavement_type, extra_cols in specs:
        usecols = ["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "SURVEY_DATE", cracking_col, *extra_cols]
        df = pd.read_excel(DATA_DIR / "Analysis Ready Distress.xlsx", sheet_name=sheet, usecols=usecols)
        df["state_code"] = df["STATE_CODE"].astype(str).str.strip()
        df["shrp_id"] = df["SHRP_ID"].astype(str).str.strip()
        df["node_id"] = df.apply(lambda row: build_node_id(row["state_code"], row["shrp_id"]), axis=1)
        df["survey_date"] = pd.to_datetime(df["SURVEY_DATE"], errors="coerce")
        df["survey_year"] = df["survey_date"].dt.year.astype("Int64")
        df["cracking_value"] = pd.to_numeric(df[cracking_col], errors="coerce")
        df["construction_no"] = pd.to_numeric(df["CONSTRUCTION_NO"], errors="coerce")
        out = df[["node_id", "survey_date", "survey_year", "construction_no", "cracking_value"]].copy()
        out["pavement_type"] = pavement_type
        out["cracking_metric"] = cracking_col
        for col in extra_cols:
            out[col.lower()] = pd.to_numeric(df[col], errors="coerce")
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def load_traffic() -> pd.DataFrame:
    trend = pd.read_excel(
        DATA_DIR / "Annual Traffic Inputs Over Time.xlsx",
        sheet_name="TRF_TREND",
        usecols=["STATE_CODE", "SHRP_ID", "YEAR", "ANNUAL_ESAL_TREND", "ANNUAL_GESAL_TREND"],
    )
    trend1 = pd.read_excel(
        DATA_DIR / "Annual Traffic Inputs Over Time.xlsx",
        sheet_name="TRF_TREND_1",
        usecols=["STATE_CODE", "SHRP_ID", "YEAR", "AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND"],
    )
    for frame in (trend, trend1):
        frame["state_code"] = frame["STATE_CODE"].astype(str).str.strip()
        frame["shrp_id"] = frame["SHRP_ID"].astype(str).str.strip()
        frame["node_id"] = frame.apply(lambda row: build_node_id(row["state_code"], row["shrp_id"]), axis=1)
        frame["YEAR"] = pd.to_numeric(frame["YEAR"], errors="coerce").astype("Int64")
    merged = trend[["node_id", "YEAR", "ANNUAL_ESAL_TREND", "ANNUAL_GESAL_TREND"]].merge(
        trend1[["node_id", "YEAR", "AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND"]],
        on=["node_id", "YEAR"],
        how="outer",
    )
    for col in ["ANNUAL_ESAL_TREND", "ANNUAL_GESAL_TREND", "AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def load_annual_climate() -> pd.DataFrame:
    climate_root = DATA_DIR / "MERRA - Temperature, Humidity, Precipitation, Wind, Solar"
    grid = pd.read_excel(climate_root / "GENERAL" / "MERRA_GRID_SECTION.xlsx", sheet_name="MERRA_GRID_SECTION")
    grid["state_code"] = grid["STATE_CODE"].astype(str).str.strip()
    grid["node_id_join"] = grid.apply(lambda row: build_node_id_join(row["state_code"], row["SHRP_ID"]), axis=1)
    grid["merra_id"] = grid["MERRA_ID"].astype(str).str.strip()
    grid = grid[["node_id_join", "merra_id"]].drop_duplicates()

    specs = [
        ("HUMIDITY/MERRA_HUMID_YEAR.xlsx", "MERRA_HUMID_YEAR", ["REL_HUM_AVG_AVG"], "humid_"),
        ("PRECIPITATION/MERRA_PRECIP_YEAR.xlsx", "MERRA_PRECIP_YEAR", ["PRECIPITATION", "EVAPORATION", "PRECIP_DAYS"], "precip_"),
        ("WIND/MERRA_WIND_YEAR.xlsx", "MERRA_WIND_YEAR", ["WIND_VELOCITY_AVG"], "wind_"),
        ("SOLAR/MERRA_SOLAR_YEAR.xlsx", "MERRA_SOLAR_YEAR", ["CLOUD_COVER_AVG", "SHORTWAVE_SURFACE_AVG"], "solar_"),
        ("TEMPERATURE/MERRA_TEMP_YEAR.xlsx", "MERRA_TEMP_YEAR", ["TEMP_AVG", "TEMP_MEAN_AVG", "FREEZE_INDEX", "FREEZE_THAW"], "temp_year_"),
    ]
    merged: pd.DataFrame | None = None
    for rel_path, sheet, value_cols, prefix in specs:
        df = pd.read_excel(climate_root / rel_path, sheet_name=sheet).rename(columns={"MERRA_ID": "merra_id"})
        df["merra_id"] = df["merra_id"].astype(str).str.strip()
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["merra_id", "YEAR", *value_cols]].groupby(["merra_id", "YEAR"], as_index=False).mean()
        df = df.rename(columns={col: f"{prefix}{col.lower()}" for col in value_cols})
        merged = df if merged is None else merged.merge(df, on=["merra_id", "YEAR"], how="outer")
    assert merged is not None
    return grid.merge(merged, on="merra_id", how="left")


def load_monthly_climate() -> dict[str, pd.DataFrame]:
    climate_root = DATA_DIR / "MERRA - Temperature, Humidity, Precipitation, Wind, Solar"

    def load_one(rel_path: str, sheet: str, cols: list[str]) -> pd.DataFrame:
        df = pd.read_excel(climate_root / rel_path, sheet_name=sheet).rename(columns={"MERRA_ID": "merra_id"})
        df["merra_id"] = df["merra_id"].astype(str).str.strip()
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").astype("Int64")
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["merra_id", "YEAR", "MONTH", *cols]].dropna(subset=["merra_id", "YEAR", "MONTH"])

    return {
        "temp": load_one("TEMPERATURE/MERRA_TEMP_MONTH.xlsx", "MERRA_TEMP_MONTH", ["TEMP_AVG", "FREEZE_THAW", "FREEZE_INDEX"]),
        "humid": load_one("HUMIDITY/MERRA_HUMID_MONTH.xlsx", "MERRA_HUMID_MONTH", ["REL_HUM_AVG_AVG"]),
        "precip": load_one("PRECIPITATION/MERRA_PRECIP_MONTH.xlsx", "MERRA_PRECIP_MONTH", ["PRECIPITATION", "PRECIP_DAYS", "EVAPORATION"]),
        "wind": load_one("WIND/MERRA_WIND_MONTH.xlsx", "MERRA_WIND_MONTH", ["WIND_VELOCITY_AVG"]),
        "solar": load_one("SOLAR/MERRA_SOLAR_MONTH.xlsx", "MERRA_SOLAR_MONTH", ["CLOUD_COVER_AVG", "SHORTWAVE_SURFACE_AVG"]),
    }


def annual_window_features(node_id_join: str, event_year: int, annual_climate: pd.DataFrame) -> dict[str, float]:
    rows = annual_climate[(annual_climate["node_id_join"] == node_id_join) & (annual_climate["YEAR"].notna())].copy()
    rows["YEAR"] = rows["YEAR"].astype(int)
    out: dict[str, float] = {}
    for window_name, years_back in [("t1", [event_year - 1]), ("t1_t3", [event_year - 1, event_year - 2, event_year - 3]), ("t1_t5", [event_year - 1, event_year - 2, event_year - 3, event_year - 4, event_year - 5])]:
        sub = rows[rows["YEAR"].isin(years_back)]
        if sub.empty:
            continue
        for col in [c for c in rows.columns if c not in {"node_id_join", "merra_id", "YEAR"}]:
            clean = pd.to_numeric(sub[col], errors="coerce").dropna()
            if clean.empty:
                continue
            if col in ANNUAL_SUM_COLS:
                out[f"annual_{window_name}_{col}_sum"] = float(clean.sum())
                out[f"annual_{window_name}_{col}_mean"] = float(clean.mean())
            else:
                out[f"annual_{window_name}_{col}_mean"] = float(clean.mean())
                out[f"annual_{window_name}_{col}_max"] = float(clean.max())
                out[f"annual_{window_name}_{col}_min"] = float(clean.min())
    return out


def monthly_window_features(merra_id: str, event_year: int, event_month: int | None, monthly_climate: dict[str, pd.DataFrame]) -> dict[str, float]:
    merged: pd.DataFrame | None = None
    for df in monthly_climate.values():
        sub = df[df["merra_id"] == merra_id].copy()
        if sub.empty:
            continue
        merged = sub if merged is None else merged.merge(sub, on=["merra_id", "YEAR", "MONTH"], how="outer")
    if merged is None or merged.empty:
        return {}
    # Use event month when available; otherwise approximate with December of event-1 window logic.
    ev_month = int(event_month) if event_month and not pd.isna(event_month) else 12
    merged["months_before"] = merged.apply(lambda row: month_distance(event_year, ev_month, int(row["YEAR"]), int(row["MONTH"])), axis=1)
    merged = merged[(merged["months_before"] >= 1) & (merged["months_before"] <= 36)].copy()
    if merged.empty:
        return {}

    # distribution thresholds within node history
    thresholds = {}
    for col in ["PRECIPITATION", "REL_HUM_AVG_AVG", "WIND_VELOCITY_AVG", "SHORTWAVE_SURFACE_AVG", "TEMP_AVG"]:
        vals = pd.to_numeric(merged.get(col), errors="coerce").dropna()
        if not vals.empty:
            thresholds[col] = {
                "p90": float(vals.quantile(0.9)),
                "p10": float(vals.quantile(0.1)),
            }

    def season_mask(month_series: pd.Series, months: set[int]) -> pd.Series:
        return month_series.isin(months)

    out: dict[str, float] = {}
    for label, max_months in [("m3", 3), ("m6", 6), ("m12", 12), ("m24", 24), ("m36", 36)]:
        sub = merged[merged["months_before"] <= max_months]
        if sub.empty:
            continue
        temp = pd.to_numeric(sub.get("TEMP_AVG"), errors="coerce").dropna()
        if not temp.empty:
            out[f"monthly_{label}_temp_avg_mean"] = float(temp.mean())
            out[f"monthly_{label}_temp_avg_min"] = float(temp.min())
            out[f"monthly_{label}_temp_avg_max"] = float(temp.max())
            out[f"monthly_{label}_temp_avg_range"] = float(temp.max() - temp.min())
            out[f"monthly_{label}_temp_avg_std"] = float(temp.std(ddof=0))
            if "TEMP_AVG" in thresholds:
                out[f"monthly_{label}_temp_extreme_cold_months"] = int((pd.to_numeric(sub["TEMP_AVG"], errors="coerce") <= thresholds["TEMP_AVG"]["p10"]).sum())
                out[f"monthly_{label}_temp_extreme_hot_months"] = int((pd.to_numeric(sub["TEMP_AVG"], errors="coerce") >= thresholds["TEMP_AVG"]["p90"]).sum())
        freeze_thaw = pd.to_numeric(sub.get("FREEZE_THAW"), errors="coerce").dropna()
        if not freeze_thaw.empty:
            out[f"monthly_{label}_freeze_thaw_active_months"] = int((freeze_thaw > 0).sum())
            out[f"monthly_{label}_freeze_thaw_total"] = float(freeze_thaw.sum())
        freeze_index = pd.to_numeric(sub.get("FREEZE_INDEX"), errors="coerce").dropna()
        if not freeze_index.empty:
            out[f"monthly_{label}_freeze_index_total"] = float(freeze_index.sum())
            out[f"monthly_{label}_freeze_index_max"] = float(freeze_index.max())

        precip = pd.to_numeric(sub.get("PRECIPITATION"), errors="coerce").dropna()
        if not precip.empty:
            out[f"monthly_{label}_precip_total"] = float(precip.sum())
            out[f"monthly_{label}_precip_mean"] = float(precip.mean())
            out[f"monthly_{label}_precip_max"] = float(precip.max())
            out[f"monthly_{label}_precip_std"] = float(precip.std(ddof=0))
            if "PRECIPITATION" in thresholds:
                out[f"monthly_{label}_wet_month_count"] = int((pd.to_numeric(sub["PRECIPITATION"], errors="coerce") > thresholds["PRECIPITATION"]["p10"]).sum())
                out[f"monthly_{label}_extreme_precip_months"] = int((pd.to_numeric(sub["PRECIPITATION"], errors="coerce") >= thresholds["PRECIPITATION"]["p90"]).sum())
        precip_days = pd.to_numeric(sub.get("PRECIP_DAYS"), errors="coerce").dropna()
        if not precip_days.empty:
            out[f"monthly_{label}_precip_days_total"] = float(precip_days.sum())
            out[f"monthly_{label}_precip_days_max"] = float(precip_days.max())
        evap = pd.to_numeric(sub.get("EVAPORATION"), errors="coerce").dropna()
        if not evap.empty:
            out[f"monthly_{label}_evap_total"] = float(evap.sum())

        humid = pd.to_numeric(sub.get("REL_HUM_AVG_AVG"), errors="coerce").dropna()
        if not humid.empty:
            out[f"monthly_{label}_humidity_mean"] = float(humid.mean())
            out[f"monthly_{label}_humidity_max"] = float(humid.max())
            out[f"monthly_{label}_humidity_min"] = float(humid.min())
            out[f"monthly_{label}_humidity_std"] = float(humid.std(ddof=0))
            if "REL_HUM_AVG_AVG" in thresholds:
                out[f"monthly_{label}_high_humidity_months"] = int((pd.to_numeric(sub["REL_HUM_AVG_AVG"], errors="coerce") >= thresholds["REL_HUM_AVG_AVG"]["p90"]).sum())

        wind = pd.to_numeric(sub.get("WIND_VELOCITY_AVG"), errors="coerce").dropna()
        if not wind.empty:
            out[f"monthly_{label}_wind_mean"] = float(wind.mean())
            out[f"monthly_{label}_wind_max"] = float(wind.max())
            out[f"monthly_{label}_wind_std"] = float(wind.std(ddof=0))
            out[f"monthly_{label}_wind_range"] = float(wind.max() - wind.min())
            if "WIND_VELOCITY_AVG" in thresholds:
                out[f"monthly_{label}_high_wind_months"] = int((pd.to_numeric(sub["WIND_VELOCITY_AVG"], errors="coerce") >= thresholds["WIND_VELOCITY_AVG"]["p90"]).sum())

        cloud = pd.to_numeric(sub.get("CLOUD_COVER_AVG"), errors="coerce").dropna()
        if not cloud.empty:
            out[f"monthly_{label}_cloud_mean"] = float(cloud.mean())
            out[f"monthly_{label}_cloud_std"] = float(cloud.std(ddof=0))
        solar = pd.to_numeric(sub.get("SHORTWAVE_SURFACE_AVG"), errors="coerce").dropna()
        if not solar.empty:
            out[f"monthly_{label}_shortwave_mean"] = float(solar.mean())
            out[f"monthly_{label}_shortwave_max"] = float(solar.max())
            out[f"monthly_{label}_shortwave_std"] = float(solar.std(ddof=0))
            if "SHORTWAVE_SURFACE_AVG" in thresholds:
                out[f"monthly_{label}_high_shortwave_months"] = int((pd.to_numeric(sub["SHORTWAVE_SURFACE_AVG"], errors="coerce") >= thresholds["SHORTWAVE_SURFACE_AVG"]["p90"]).sum())

        if max_months >= 12:
            winter = sub[season_mask(sub["MONTH"], {12, 1, 2})]
            summer = sub[season_mask(sub["MONTH"], {6, 7, 8})]
            winter_temp = pd.to_numeric(winter.get("TEMP_AVG"), errors="coerce").dropna()
            summer_temp = pd.to_numeric(summer.get("TEMP_AVG"), errors="coerce").dropna()
            if not winter_temp.empty:
                out[f"monthly_{label}_winter_temp_mean"] = float(winter_temp.mean())
            if not summer_temp.empty:
                out[f"monthly_{label}_summer_temp_mean"] = float(summer_temp.mean())
            if not winter_temp.empty and not summer_temp.empty:
                out[f"monthly_{label}_seasonal_temp_contrast"] = float(summer_temp.mean() - winter_temp.mean())
            winter_precip = pd.to_numeric(winter.get("PRECIPITATION"), errors="coerce").dropna()
            summer_precip = pd.to_numeric(summer.get("PRECIPITATION"), errors="coerce").dropna()
            if not winter_precip.empty:
                out[f"monthly_{label}_winter_precip_total"] = float(winter_precip.sum())
            if not summer_precip.empty:
                out[f"monthly_{label}_summer_precip_total"] = float(summer_precip.sum())
            if not winter_precip.empty and not summer_precip.empty:
                out[f"monthly_{label}_seasonal_precip_contrast"] = float(max(winter_precip.sum(), summer_precip.sum()) - min(winter_precip.sum(), summer_precip.sum()))
            winter_freeze = pd.to_numeric(winter.get("FREEZE_THAW"), errors="coerce").dropna()
            if not winter_freeze.empty:
                out[f"monthly_{label}_winter_freeze_thaw_total"] = float(winter_freeze.sum())
            summer_shortwave = pd.to_numeric(summer.get("SHORTWAVE_SURFACE_AVG"), errors="coerce").dropna()
            if not summer_shortwave.empty:
                out[f"monthly_{label}_summer_shortwave_mean"] = float(summer_shortwave.mean())
    return out


def nearest_pre_post(distress: pd.DataFrame, node_id: str, event_date: pd.Timestamp) -> dict[str, object]:
    node = distress[distress["node_id"] == node_id].copy()
    if node.empty:
        return {}
    node["pre_gap_days"] = (event_date - node["survey_date"]).dt.days
    node["post_gap_days"] = (node["survey_date"] - event_date).dt.days
    out: dict[str, object] = {}
    for years, days in [(1, 365), (2, 730), (3, 1095)]:
        pre = node[(node["pre_gap_days"] >= 0) & (node["pre_gap_days"] <= days)].sort_values("pre_gap_days").head(1)
        post = node[(node["post_gap_days"] >= 0) & (node["post_gap_days"] <= days)].sort_values("post_gap_days").head(1)
        if not pre.empty:
            out[f"pre_cracking_{years}yr"] = float(pre.iloc[0]["cracking_value"])
            out[f"pre_gap_days_{years}yr"] = int(pre.iloc[0]["pre_gap_days"])
            out[f"pre_survey_date_{years}yr"] = pre.iloc[0]["survey_date"]
        if not post.empty:
            out[f"post_cracking_{years}yr"] = float(post.iloc[0]["cracking_value"])
            out[f"post_gap_days_{years}yr"] = int(post.iloc[0]["post_gap_days"])
            out[f"post_survey_date_{years}yr"] = post.iloc[0]["survey_date"]
    return out


def traffic_before_after(traffic: pd.DataFrame, node_id: str, event_year: int) -> dict[str, object]:
    node = traffic[traffic["node_id"] == node_id].copy()
    if node.empty:
        return {}
    out: dict[str, object] = {}
    for offset, label in [(-1, "pre"), (0, "event"), (1, "post1"), (2, "post2"), (3, "post3")]:
        sub = node[node["YEAR"] == event_year + offset]
        if sub.empty:
            continue
        row = sub.iloc[0]
        out[f"{label}_aadt_trucks"] = row.get("AADTT_ALL_TRUCKS_TREND")
        out[f"{label}_truck_volume"] = row.get("ANNUAL_TRUCK_VOLUME_TREND")
        out[f"{label}_esal"] = row.get("ANNUAL_ESAL_TREND")
        out[f"{label}_gesal"] = row.get("ANNUAL_GESAL_TREND")
    if "pre_aadt_trucks" in out and "post1_aadt_trucks" in out:
        out["traffic_change_pre_to_post"] = float(out["post1_aadt_trucks"] - out["pre_aadt_trucks"])
    if "pre_truck_volume" in out and "post1_truck_volume" in out:
        out["truck_change_pre_to_post"] = float(out["post1_truck_volume"] - out["pre_truck_volume"])
    if "pre_esal" in out and "post1_esal" in out:
        out["esal_change_pre_to_post"] = float(out["post1_esal"] - out["pre_esal"])
    return out


def summarise_neighbours(node_ids: list[str], by_event: pd.DataFrame, climate_cols: list[str], prefix: str) -> dict[str, float]:
    sub = by_event[by_event["node_id"].isin(node_ids)].copy()
    out: dict[str, float] = {}
    if sub.empty:
        return out
    for col in ["pre_cracking_3yr", "post_cracking_3yr", "traffic_change_pre_to_post", "esal_change_pre_to_post", "truck_change_pre_to_post"]:
        if col in sub.columns:
            clean = pd.to_numeric(sub[col], errors="coerce").dropna()
            if not clean.empty:
                out[f"{prefix}_{col}_mean"] = float(clean.mean())
    for col in climate_cols:
        clean = pd.to_numeric(sub[col], errors="coerce").dropna()
        if not clean.empty:
            out[f"{prefix}_{col}_mean"] = float(clean.mean())
    out[f"{prefix}_n_sections"] = int(len(sub))
    return out


def run_classifier(df: pd.DataFrame, feature_cols: list[str], label: str, model_name: str, model) -> dict[str, object]:
    usable = df.dropna(subset=["broad_treatment_group"]).copy()
    usable = usable[usable["broad_treatment_group"] != "unknown"].copy()
    counts = usable["broad_treatment_group"].value_counts()
    usable = usable[usable["broad_treatment_group"].isin(counts[counts >= 20].index)].copy()
    if len(usable) < 150 or usable["broad_treatment_group"].nunique() < 3 or not feature_cols:
        return {
            "feature_set": label,
            "model": model_name,
            "n_rows": int(len(usable)),
            "n_classes": int(usable["broad_treatment_group"].nunique()) if not usable.empty else 0,
            "accuracy": None,
            "macro_f1": None,
            "balanced_accuracy": None,
        }
    X = usable[feature_cols].copy()
    y = usable["broad_treatment_group"].copy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, bals = [], [], []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("model", model)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        accs.append(accuracy_score(y_test, pred))
        f1s.append(f1_score(y_test, pred, average="macro"))
        bals.append(balanced_accuracy_score(y_test, pred))
    return {
        "feature_set": label,
        "model": model_name,
        "n_rows": int(len(usable)),
        "n_classes": int(y.nunique()),
        "accuracy": float(np.mean(accs)),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(bals)),
    }


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    events = load_experiment_events()
    nodes = load_nodes()
    neighbours = load_neighbour_map()
    distress = load_distress()
    traffic = load_traffic()
    annual_climate = load_annual_climate()
    monthly_climate = load_monthly_climate()

    events = events.merge(
        nodes[["node_id", "route_key", "functional_class", "NO_OF_LANES", "state_code"]],
        on=["node_id", "state_code"],
        how="left",
    )

    merra_lookup = annual_climate[["node_id_join", "merra_id"]].drop_duplicates()
    events = events.merge(merra_lookup, on="node_id_join", how="left")

    by_event_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        if pd.isna(event.event_start_date) or pd.isna(event.event_year):
            continue
        event_date = pd.Timestamp(event.event_start_date)
        row = {
            "node_id": event.node_id,
            "state_code": event.state_code,
            "state_name": event.state_name,
            "route_key": event.route_key,
            "functional_class": event.functional_class,
            "construction_no": event.construction_no,
            "event_date": event_date,
            "event_year": int(event.event_year),
            "event_month": int(event.event_month) if not pd.isna(event.event_month) else np.nan,
            "treatment_label": event.treatment_label,
            "broad_treatment_group": event.broad_treatment_group,
            "experiment_label": event.experiment_label,
            "status_label": event.status_label,
            "gps_sps_label": event.gps_sps_label,
            "merra_id": event.merra_id,
        }
        row.update(nearest_pre_post(distress, event.node_id, event_date))
        row.update(traffic_before_after(traffic, event.node_id, int(event.event_year)))
        if event.node_id_join and event.event_year:
            row.update(annual_window_features(str(event.node_id_join), int(event.event_year), annual_climate))
        if event.merra_id:
            row.update(monthly_window_features(str(event.merra_id), int(event.event_year), int(event.event_month) if not pd.isna(event.event_month) else None, monthly_climate))
        by_event_rows.append(row)

    by_event = pd.DataFrame(by_event_rows)

    # Neighbour summaries
    climate_cols = [c for c in by_event.columns if c.startswith("annual_") or c.startswith("monthly_")]
    neighbour_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for row in by_event.itertuples(index=False):
        node_neigh = sorted(neighbours.get(row.node_id, set()))
        one_hop = summarise_neighbours(node_neigh, by_event, climate_cols, "neighbour")
        base = row._asdict()
        base.update(one_hop)
        if "pre_cracking_3yr" in base and "neighbour_pre_cracking_3yr_mean" in base and "post_cracking_3yr" in base and "neighbour_post_cracking_3yr_mean" in base:
            base["treated_cracking_change"] = float(base["post_cracking_3yr"] - base["pre_cracking_3yr"])
            base["neighbour_cracking_change"] = float(base["neighbour_post_cracking_3yr_mean"] - base["neighbour_pre_cracking_3yr_mean"])
            base["treated_minus_neighbour_cracking_change"] = float(base["treated_cracking_change"] - base["neighbour_cracking_change"])
        if "traffic_change_pre_to_post" in base and "neighbour_traffic_change_pre_to_post_mean" in base:
            base["treated_minus_neighbour_traffic_change"] = float(base["traffic_change_pre_to_post"] - base["neighbour_traffic_change_pre_to_post_mean"])
        if "esal_change_pre_to_post" in base and "neighbour_esal_change_pre_to_post_mean" in base:
            base["treated_minus_neighbour_esal_change"] = float(base["esal_change_pre_to_post"] - base["neighbour_esal_change_pre_to_post_mean"])
        # climate differences
        for col in climate_cols:
            neigh_col = f"neighbour_{col}_mean"
            if col in base and neigh_col in base and pd.notna(base[col]) and pd.notna(base[neigh_col]):
                base[f"treated_minus_neighbour_{col}"] = float(base[col] - base[neigh_col])
        neighbour_rows.append(base)

        # simple control match: same state + same functional class + not neighbour
        candidates = by_event[
            (by_event["state_code"] == row.state_code)
            & (by_event["functional_class"] == row.functional_class)
            & (by_event["node_id"] != row.node_id)
            & (~by_event["node_id"].isin(node_neigh))
        ].copy()
        if not candidates.empty and "pre_cracking_3yr" in candidates.columns and pd.notna(getattr(row, "pre_cracking_3yr", np.nan)):
            candidates["match_score"] = (pd.to_numeric(candidates["pre_cracking_3yr"], errors="coerce") - float(getattr(row, "pre_cracking_3yr", np.nan))).abs()
            if "pre_aadt_trucks" in candidates.columns and pd.notna(getattr(row, "pre_aadt_trucks", np.nan)):
                candidates["match_score"] += (pd.to_numeric(candidates["pre_aadt_trucks"], errors="coerce") - float(getattr(row, "pre_aadt_trucks", np.nan))).abs() / 100.0
            control = candidates.sort_values("match_score").head(1)
            if not control.empty:
                ctrl = control.iloc[0]
                control_rows.append(
                    {
                        "treated_node_id": row.node_id,
                        "control_node_id": ctrl["node_id"],
                        "state_code": row.state_code,
                        "functional_class": row.functional_class,
                        "treated_cracking_change": (getattr(row, "post_cracking_3yr", np.nan) - getattr(row, "pre_cracking_3yr", np.nan)) if pd.notna(getattr(row, "post_cracking_3yr", np.nan)) and pd.notna(getattr(row, "pre_cracking_3yr", np.nan)) else np.nan,
                        "control_cracking_change": (ctrl.get("post_cracking_3yr", np.nan) - ctrl.get("pre_cracking_3yr", np.nan)) if pd.notna(ctrl.get("post_cracking_3yr", np.nan)) and pd.notna(ctrl.get("pre_cracking_3yr", np.nan)) else np.nan,
                        "treated_traffic_change": getattr(row, "traffic_change_pre_to_post", np.nan),
                        "control_traffic_change": ctrl.get("traffic_change_pre_to_post", np.nan),
                        "treated_esal_change": getattr(row, "esal_change_pre_to_post", np.nan),
                        "control_esal_change": ctrl.get("esal_change_pre_to_post", np.nan),
                    }
                )

    by_event = pd.DataFrame(neighbour_rows)
    control_df = pd.DataFrame(control_rows)

    # Group summaries
    group_rows = []
    climate_group_rows = []
    for group, sub in by_event.groupby("broad_treatment_group", dropna=False):
        group_row = {"broad_treatment_group": group, "n_events": int(len(sub))}
        for col in ["pre_cracking_3yr", "post_cracking_3yr", "treated_cracking_change", "pre_aadt_trucks", "post1_aadt_trucks", "traffic_change_pre_to_post", "esal_change_pre_to_post", "neighbour_cracking_change", "treated_minus_neighbour_cracking_change"]:
            if col in sub.columns:
                clean = pd.to_numeric(sub[col], errors="coerce")
                group_row[f"{col}_median"] = float(clean.median()) if clean.notna().any() else np.nan
                group_row[f"{col}_iqr"] = safe_iqr(clean)
        group_rows.append(group_row)

        climate_row = {"broad_treatment_group": group, "n_events": int(len(sub))}
        for col in climate_cols:
            clean = pd.to_numeric(sub[col], errors="coerce")
            if clean.notna().any():
                climate_row[f"{col}_median"] = float(clean.median())
                climate_row[f"{col}_iqr"] = safe_iqr(clean)
                climate_row[f"{col}_missing_pct"] = float(clean.isna().mean() * 100)
        climate_group_rows.append(climate_row)

    by_group = pd.DataFrame(group_rows)
    climate_by_group = pd.DataFrame(climate_group_rows)

    # Climate feature diagnostics / redundancy
    climate_feature_rows = []
    redundancy_rows = []
    annual_cols = [c for c in by_event.columns if c.startswith("annual_")]
    monthly_cols = [c for c in by_event.columns if c.startswith("monthly_")]
    for col in monthly_cols:
        clean = pd.to_numeric(by_event[col], errors="coerce")
        variance = float(clean.var()) if clean.notna().any() else np.nan
        missingness = float(clean.isna().mean() * 100)
        closest_annual = None
        best_corr = -np.inf
        for annual_col in annual_cols:
            pair = by_event[[col, annual_col]].dropna()
            if len(pair) < 20:
                continue
            corr = float(pair[col].corr(pair[annual_col]))
            if np.isfinite(corr) and abs(corr) > best_corr:
                best_corr = abs(corr)
                closest_annual = annual_col
        recommendation = "keep"
        if missingness > 40 or (np.isfinite(variance) and variance < 1e-8):
            recommendation = "drop"
        elif np.isfinite(best_corr) and best_corr > 0.95:
            recommendation = "redundant"
        climate_feature_rows.append(
            {
                "feature_name": col,
                "feature_family": col.split("_")[2] if len(col.split("_")) > 2 else "other",
                "variance": variance,
                "missingness_pct": missingness,
                "closest_annual_feature": closest_annual,
                "closest_abs_correlation": best_corr if np.isfinite(best_corr) else np.nan,
                "recommendation": recommendation,
            }
        )
        redundancy_rows.append(
            {
                "monthly_feature": col,
                "closest_annual_feature": closest_annual,
                "correlation": best_corr if np.isfinite(best_corr) else np.nan,
                "missingness_pct": missingness,
                "variance": variance,
                "keep_redundant_recommendation": recommendation,
            }
        )

    climate_features = pd.DataFrame(climate_feature_rows)
    redundancy = pd.DataFrame(redundancy_rows)

    # Classifiers
    feature_sets = {
        "distress_only": [c for c in ["pre_cracking_1yr", "pre_cracking_2yr", "pre_cracking_3yr"] if c in by_event.columns],
        "traffic_only": [c for c in ["pre_aadt_trucks", "pre_truck_volume", "pre_esal", "pre_gesal"] if c in by_event.columns],
        "annual_climate_only": annual_cols,
        "monthly_climate_only": monthly_cols,
    }
    feature_sets["distress_plus_traffic"] = sorted(set(feature_sets["distress_only"] + feature_sets["traffic_only"]))
    feature_sets["distress_plus_traffic_plus_annual"] = sorted(set(feature_sets["distress_plus_traffic"] + feature_sets["annual_climate_only"]))
    feature_sets["distress_plus_traffic_plus_monthly"] = sorted(set(feature_sets["distress_plus_traffic"] + feature_sets["monthly_climate_only"]))
    feature_sets["distress_plus_traffic_plus_annual_plus_monthly"] = sorted(
        set(feature_sets["distress_plus_traffic"] + feature_sets["annual_climate_only"] + feature_sets["monthly_climate_only"])
    )

    classifier_rows = []
    importance_rows = []
    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=2000),
    }
    for feature_set_name, cols in feature_sets.items():
        for model_name, model in models.items():
            classifier_rows.append(run_classifier(by_event, cols, feature_set_name, model_name, model))
        # Feature importances on RF fitted on full usable sample
        usable = by_event.dropna(subset=["broad_treatment_group"]).copy()
        usable = usable[usable["broad_treatment_group"] != "unknown"].copy()
        counts = usable["broad_treatment_group"].value_counts()
        usable = usable[usable["broad_treatment_group"].isin(counts[counts >= 20].index)].copy()
        if len(usable) >= 150 and usable["broad_treatment_group"].nunique() >= 3 and cols:
            X = usable[cols].copy()
            y = usable["broad_treatment_group"].copy()
            pipe = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42)),
                ]
            )
            pipe.fit(X, y)
            for feature, importance in sorted(zip(cols, pipe.named_steps["rf"].feature_importances_), key=lambda pair: pair[1], reverse=True):
                importance_rows.append({"feature_set": feature_set_name, "feature": feature, "importance": float(importance)})

    classifier_df = pd.DataFrame(classifier_rows)
    importance_df = pd.DataFrame(importance_rows)

    # Neighbour climate summary
    neighbour_climate_cols = [c for c in by_event.columns if c.startswith("treated_minus_neighbour_annual_") or c.startswith("treated_minus_neighbour_monthly_")]
    neighbour_climate_summary = []
    for group, sub in by_event.groupby("broad_treatment_group", dropna=False):
        row = {"broad_treatment_group": group, "n_events": int(len(sub))}
        for col in neighbour_climate_cols:
            clean = pd.to_numeric(sub[col], errors="coerce")
            if clean.notna().any():
                row[f"{col}_median"] = float(clean.median())
        neighbour_climate_summary.append(row)
    neighbour_climate_df = pd.DataFrame(neighbour_climate_summary)

    interpretation = {
        "top_groups": by_event["broad_treatment_group"].value_counts(dropna=False).head(10).to_dict(),
        "n_events": int(len(by_event)),
        "n_controls": int(len(control_df)),
        "notes": {
            "monthly_modeling": "Traffic is annual and distress is irregular, so this remains an annual event-study with exact event dates used only for monthly climate windows.",
            "causality": "This is exploratory and descriptive, not a causal proof.",
        },
    }

    by_event.to_csv(REPORT_DIR / "experiment_event_study_by_event.csv", index=False)
    by_group.to_csv(REPORT_DIR / "experiment_event_study_by_group.csv", index=False)
    climate_by_group.to_csv(REPORT_DIR / "experiment_event_study_climate_by_group.csv", index=False)
    climate_features.to_csv(REPORT_DIR / "experiment_event_study_climate_features.csv", index=False)
    redundancy.to_csv(REPORT_DIR / "event_study_monthly_vs_annual_climate_redundancy.csv", index=False)
    classifier_df.to_csv(REPORT_DIR / "experiment_event_study_treatment_classifier.csv", index=False)
    importance_df.to_csv(REPORT_DIR / "experiment_event_study_feature_importance.csv", index=False)
    neighbour_climate_df.to_csv(REPORT_DIR / "experiment_event_study_neighbour_climate_summary.csv", index=False)
    control_df.to_csv(REPORT_DIR / "experiment_event_study_neighbour_vs_control.csv", index=False)
    with open(REPORT_DIR / "experiment_event_study_interpretation.json", "w", encoding="utf-8") as fh:
        json.dump(interpretation, fh, indent=2)

    print("Saved event-study outputs to reports/.")
    print("Events:", len(by_event))
    print("Classifier rows:", len(classifier_df))
    print("Controls:", len(control_df))


if __name__ == "__main__":
    main()
