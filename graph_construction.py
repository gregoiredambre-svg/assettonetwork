"""Graph construction and interdependency modelling for the thesis project."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
CLIMATE_REPORT: dict[str, object] = {}


def log(message: str):
    print(f"[graph_construction] {message}")
DATA_DIR = ROOT / "Research Data"
GRAPH_DIR = ROOT / "graph_data"
SPATIAL_K = 8
SPATIAL_MAX_DISTANCE_KM = 80.0
ROUTE_MAX_DISTANCE_KM = 100.0
SIMILARITY_K = 5
SIMILARITY_MAX_DISTANCE_KM = 80.0

CONTIGUOUS_US_STATES = {
    'Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Delaware', 'Florida', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
    'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
    'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia'
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometers."""
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def similarity_from_series(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if not mask.any():
        return 0.5
    left_vals = left[mask].astype(float)
    right_vals = right[mask].astype(float)
    scale = np.maximum(np.maximum(left_vals.abs(), right_vals.abs()), 1.0)
    similarity = 1.0 - (left_vals.sub(right_vals).abs() / scale)
    return float(np.clip(similarity.mean(), 0.0, 1.0))


def valid_route_key(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & text.ne("_")


TREATMENT_GROUPS = {
    "crack_sealing": ["crack sealing", "joint sealing", "saw and seal"],
    "asphalt_overlay": ["overlay", "mill off ac and overlay", "mill existing pavement and overlay", "warm mix ac overlay"],
    "seal_coat": ["seal coat", "slurry seal", "fog seal", "surface treatment", "prime coat", "tack coat", "sand seal"],
    "patching": ["patch", "pothole", "spot patch", "skin patch", "strip patch"],
    "grinding": ["grinding", "grooving"],
    "shoulder_restoration": ["shoulder restoration", "shoulder replacement"],
    "reconstruction_or_major_rehab": ["reconstruction", "slab replacement", "fracture treatment", "load transfer restoration", "subdrain", "subdrainage", "drainage", "jacking", "subsealing"],
}


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


def load_excel_table(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


def normalize_string_columns(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'nan': None})
    return df


def clean_numeric_frame(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    work = df.copy()
    for col in work.columns:
        if col in keep_cols:
            continue
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def prepare_node_table() -> pd.DataFrame:
    """Build the section node table with spatial and asset metadata."""
    log("Loading section metadata from Excel...")
    file = DATA_DIR / "General Section Info.xlsx"
    xls = pd.ExcelFile(file)

    coords = load_excel_table(file, "SECTION_COORDINATES")
    route = load_excel_table(file, "PROJECT_ID_EXP")
    section = load_excel_table(file, "SECTION_GENERAL_EXP")

    for df in (coords, route, section):
        normalize_string_columns(df, ["STATE_CODE_EXP", "SHRP_ID", "ROUTE_SIGNING", "ROUTE_SIGNING_EXP", "FUNCTIONAL_CLASS", "FUNCTIONAL_CLASS_EXP"])

    coords = coords[coords["STATE_CODE_EXP"].isin(CONTIGUOUS_US_STATES)].copy()
    route = route[route["STATE_CODE_EXP"].isin(CONTIGUOUS_US_STATES)].copy()
    section = section[section["STATE_CODE_EXP"].isin(CONTIGUOUS_US_STATES)].copy()
    log(f"Sections after US filter: coords={len(coords)}, route={len(route)}, section={len(section)}")

    coords = coords.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id"})
    route = route.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id"})
    section = section.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id"})

    coords = coords.dropna(subset=["LATITUDE", "LONGITUDE"])
    coords = coords[(coords["LONGITUDE"] >= -125) & (coords["LONGITUDE"] <= -66) & (coords["LATITUDE"] >= 24) & (coords["LATITUDE"] <= 50)]
    log(f"Coordinates after bounding box filter: {len(coords)} rows")

    nodes = coords.drop_duplicates(subset=["state_code", "shrp_id"]).copy()
    nodes["node_id"] = nodes["state_code"].astype(str) + "_" + nodes["shrp_id"].astype(str)
    nodes = nodes.set_index("node_id")

    route_cols = [c for c in route.columns if c not in ["state_code", "state_code_exp", "shrp_id"]]
    route = route.drop_duplicates(subset=["state_code", "shrp_id"])[["state_code", "shrp_id"] + route_cols]
    route["node_id"] = route["state_code"].astype(str) + "_" + route["shrp_id"].astype(str)
    route = route.set_index("node_id")

    section_cols = [c for c in section.columns if c not in ["state_code", "state_code_exp", "shrp_id"]]
    section = section.drop_duplicates(subset=["state_code", "shrp_id"])[["state_code", "shrp_id"] + section_cols]
    section["node_id"] = section["state_code"].astype(str) + "_" + section["shrp_id"].astype(str)
    section = section.set_index("node_id")

    node_table = nodes.join(route.drop(columns=["state_code", "STATE_CODE_EXP", "shrp_id"], errors="ignore"), how="left")
    node_table = node_table.join(section.drop(columns=["state_code", "STATE_CODE_EXP", "shrp_id"], errors="ignore"), how="left")
    node_table = node_table.reset_index()
    node_table = node_table.rename(columns={"LATITUDE": "latitude", "LONGITUDE": "longitude", "ELEVATION": "elevation"})

    # Simple feature engineering for route/corridor grouping
    node_table["route_key"] = (
        node_table["ROUTE_SIGNING"].fillna("") + "_" + node_table["ROUTE_NO"].astype(str).fillna("")
    )
    node_table["functional_class"] = node_table["FUNCTIONAL_CLASS"].fillna(node_table["FUNCTIONAL_CLASS_EXP"])
    log(f"Node table prepared with {len(node_table)} nodes")
    return node_table


def load_climate_features() -> pd.DataFrame:
    """Load bind + extended yearly MERRA climate features aggregated at section level."""
    log("Loading climate features...")
    climate_root = DATA_DIR / "MERRA - Temperature, Humidity, Precipitation, Wind, Solar"

    grid_path = climate_root / "GENERAL" / "MERRA_GRID_SECTION.xlsx"
    grid = load_excel_table(grid_path, "MERRA_GRID_SECTION")
    grid = grid.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id", "MERRA_ID": "merra_id"})
    normalize_string_columns(grid, ["state_code", "shrp_id", "merra_id"])
    grid["node_id"] = grid["state_code"].astype(str) + "_" + grid["shrp_id"].astype(str)
    grid["merra_grid_elevation"] = pd.to_numeric(grid["ELEVATION"], errors="coerce")
    grid = grid[["node_id", "merra_id", "merra_grid_elevation"]].drop_duplicates(subset=["node_id"])

    bind_path = climate_root / "TEMPERATURE" / "VW_MERRA_BIND_CLIMATE_DATA.xlsx"
    bind = load_excel_table(bind_path, "VW_MERRA_BIND_CLIMATE_DATA")
    bind = bind.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id", "MERRA_ID": "merra_id"})
    normalize_string_columns(bind, ["state_code", "shrp_id", "merra_id"])
    bind["node_id"] = bind["state_code"].astype(str) + "_" + bind["shrp_id"].astype(str)
    bind = clean_numeric_frame(bind, ["state_code", "shrp_id", "merra_id", "node_id"])
    bind_cols = [
        col for col in bind.columns
        if col not in {"state_code", "shrp_id", "merra_id", "node_id"}
    ]
    bind_agg = bind.groupby("node_id")[bind_cols].mean().reset_index()
    bind_agg = bind_agg.rename(columns={col: f"temp_bind_{col.lower()}" for col in bind_cols})

    year_specs = [
        (
            climate_root / "HUMIDITY" / "MERRA_HUMID_YEAR.xlsx",
            "MERRA_HUMID_YEAR",
            ["REL_HUM_AVG_AVG"],
            "humid_",
        ),
        (
            climate_root / "PRECIPITATION" / "MERRA_PRECIP_YEAR.xlsx",
            "MERRA_PRECIP_YEAR",
            ["PRECIPITATION", "EVAPORATION", "PRECIP_DAYS"],
            "precip_",
        ),
        (
            climate_root / "WIND" / "MERRA_WIND_YEAR.xlsx",
            "MERRA_WIND_YEAR",
            ["WIND_VELOCITY_AVG"],
            "wind_",
        ),
        (
            climate_root / "SOLAR" / "MERRA_SOLAR_YEAR.xlsx",
            "MERRA_SOLAR_YEAR",
            ["CLOUD_COVER_AVG", "SHORTWAVE_SURFACE_AVG"],
            "solar_",
        ),
        (
            climate_root / "TEMPERATURE" / "MERRA_TEMP_YEAR.xlsx",
            "MERRA_TEMP_YEAR",
            ["TEMP_AVG", "TEMP_MEAN_AVG", "FREEZE_INDEX", "FREEZE_THAW"],
            "temp_year_",
        ),
    ]

    year_frames: list[pd.DataFrame] = []
    for path, sheet, value_cols, prefix in year_specs:
        frame = load_excel_table(path, sheet)
        frame = frame.rename(columns={"MERRA_ID": "merra_id"})
        normalize_string_columns(frame, ["merra_id"])
        keep_cols = ["merra_id", "YEAR", *value_cols]
        frame = frame[[col for col in keep_cols if col in frame.columns]].copy()
        frame = clean_numeric_frame(frame, ["merra_id"])
        agg = frame.groupby("merra_id")[value_cols].mean().reset_index()
        agg = agg.rename(columns={col: f"{prefix}{col.lower()}" for col in value_cols})
        year_frames.append(agg)

    climate = grid.merge(bind_agg, on="node_id", how="left")
    for frame in year_frames:
        climate = climate.merge(frame, on="merra_id", how="left")

    global CLIMATE_REPORT
    feature_cols = [col for col in climate.columns if col not in {"node_id", "merra_id"}]
    prefixed_climate_cols = [col for col in feature_cols if col.startswith(("temp_bind_", "humid_", "precip_", "wind_", "solar_", "temp_year_"))]
    new_annual_climate_cols = [col for col in prefixed_climate_cols if not col.startswith("temp_bind_")]
    CLIMATE_REPORT = {
        "climate_cols_before": int(len(bind_cols)),
        "climate_cols_after": int(len(prefixed_climate_cols)),
        "new_climate_cols_added": int(len(new_annual_climate_cols)),
        "nodes_with_merra_id": None,
        "coverage_pct": None,
    }
    log(f"Climate features loaded for {len(climate)} node_ids with {len(feature_cols)} columns")
    return climate


def load_distress_features() -> pd.DataFrame:
    """Aggregate distress features from AC/CRCP/JPCC datasets by section."""
    log("Loading distress features...")
    file = DATA_DIR / "Analysis Ready Distress.xlsx"
    xls = pd.ExcelFile(file)
    distress_dfs = []
    for sheet in ["ANALYSIS_DIS_AC", "ANALYSIS_DIS_CRCP", "ANALYSIS_DIS_JPCC"]:
        df = load_excel_table(file, sheet)
        df = df.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id", "CONSTRUCTION_NO": "construction_no"})
        df["node_id"] = df["state_code"].astype(str) + "_" + df["shrp_id"].astype(str)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            continue
        cols = [c for c in numeric.columns if c not in ["state_code", "shrp_id", "construction_no"]]
        agg = df[["node_id"] + cols].groupby("node_id").mean()
        agg.columns = [f"distress_{sheet.lower()}_{c}" for c in agg.columns]
        distress_dfs.append(agg)

    if not distress_dfs:
        return pd.DataFrame(columns=["node_id"])
    features = pd.concat(distress_dfs, axis=1)
    features = features.loc[:, ~features.columns.duplicated()].reset_index()
    log(f"Distress features assembled for {len(features)} node_ids")
    return features


def load_traffic_features() -> pd.DataFrame:
    """Load traffic trend data and aggregate the latest annual values by section."""
    file = DATA_DIR / "Annual Traffic Inputs Over Time.xlsx"
    xls = pd.ExcelFile(file)
    traffic_dfs = []
    for sheet in ["TRF_TREND", "TRF_TREND_1", "TRF_TREND_2"]:
        df = load_excel_table(file, sheet)
        df = df.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id", "CONSTRUCTION_NO": "construction_no"})
        df["node_id"] = df["state_code"].astype(str) + "_" + df["shrp_id"].astype(str)
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            continue
        # keep the latest year for each section
        df = df.sort_values(["node_id", "YEAR"]).groupby("node_id").tail(1)
        prefix = f"traffic_{sheet.lower()}_"
        agg = df[["node_id"] + numeric].set_index("node_id")
        agg.columns = [prefix + c for c in agg.columns]
        traffic_dfs.append(agg)
    if not traffic_dfs:
        log("No traffic feature sheets found")
        return pd.DataFrame(columns=["node_id"])
    features = pd.concat(traffic_dfs, axis=1)
    features = features.loc[:, ~features.columns.duplicated()].reset_index()
    log(f"Traffic features assembled for {len(features)} node_ids")
    return features


def load_project_events() -> pd.DataFrame:
    """Load dated project/treatment events from EXPERIMENT_SECTION."""
    file = DATA_DIR / "General Section Info.xlsx"
    df = load_excel_table(file, "EXPERIMENT_SECTION")
    df = df.rename(columns={"STATE_CODE": "state_code", "SHRP_ID": "shrp_id"})
    df = normalize_string_columns(
        df,
        [
            "state_code",
            "shrp_id",
            "CN_CHANGE_REASON",
            "CN_CHANGE_REASON_EXP",
            "EXPERIMENT_NO_EXP",
            "STATUS_EXP",
            "GPS_SPS_EXP",
        ],
    )
    df["node_id"] = df["state_code"].astype(str) + "_" + df["shrp_id"].astype(str)
    df["construction_no"] = pd.to_numeric(df.get("CONSTRUCTION_NO"), errors="coerce")
    df["construction_date"] = pd.to_datetime(df.get("CN_ASSIGN_DATE"), errors="coerce")
    fallback_start = pd.to_datetime(df.get("ASSIGN_DATE"), errors="coerce")
    df["construction_date"] = df["construction_date"].fillna(fallback_start)
    df["traffic_open_date"] = pd.to_datetime(df.get("DEASSIGN_DATE"), errors="coerce")
    df["event_year"] = df["construction_date"].dt.year.astype("Int64")
    df["treatment_code"] = df.get("CN_CHANGE_REASON")
    df["treatment_label"] = df.get("CN_CHANGE_REASON_EXP").fillna("Missing / not recorded")
    df["broad_treatment_group"] = df["treatment_label"].map(classify_treatment_group)
    df["experiment_label"] = df.get("EXPERIMENT_NO_EXP")
    df["status_label"] = df.get("STATUS_EXP")
    df["gps_sps_label"] = df.get("GPS_SPS_EXP")
    df = df.dropna(subset=["construction_date"]).copy()
    df = df.sort_values(["node_id", "construction_date", "construction_no", "treatment_label"]).reset_index(drop=True)
    seq = df.groupby(["node_id", "construction_date"]).cumcount() + 1
    date_text = df["construction_date"].dt.strftime("%Y%m%d")
    construction_text = df["construction_no"].fillna(-1).astype(int).astype(str)
    df["project_id"] = df["node_id"] + "_" + date_text + "_" + construction_text + "_" + seq.astype(str)
    result = df[
        [
            "project_id",
            "node_id",
            "construction_date",
            "traffic_open_date",
            "construction_no",
            "event_year",
            "treatment_code",
            "treatment_label",
            "broad_treatment_group",
            "experiment_label",
            "status_label",
            "gps_sps_label",
        ]
    ].copy()
    log(f"Loaded {len(result)} project/treatment events from EXPERIMENT_SECTION")
    return result


def build_spatial_edges(
    nodes: pd.DataFrame,
    n_neighbors: int = SPATIAL_K,
    max_distance_km: float = SPATIAL_MAX_DISTANCE_KM,
) -> pd.DataFrame:
    coords = nodes[["node_id", "latitude", "longitude"]].dropna()
    log(f"Building spatial edges for {len(coords)} nodes")
    if coords.empty:
        return pd.DataFrame(columns=["source", "target", "distance_km", "edge_type"])

    X = np.vstack([coords["latitude"].astype(float), coords["longitude"].astype(float)]).T
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(X)), algorithm="ball_tree", metric="haversine")
    rad = np.radians(X)
    nbrs.fit(rad)
    distances, indices = nbrs.kneighbors(rad)
    edges = []
    for i, node_id in enumerate(coords["node_id"]):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):
            km = dist * 6371.0
            if km <= max_distance_km:
                target_id = coords.iloc[j]["node_id"]
                if node_id == target_id:
                    continue
                source, target = sorted([node_id, target_id])
                edges.append((source, target, km))

    edge_df = pd.DataFrame(edges, columns=["source", "target", "distance_km"]).drop_duplicates()
    edge_df["edge_type"] = "spatial"
    return edge_df


def build_route_edges(
    nodes: pd.DataFrame,
    max_distance_km: float = ROUTE_MAX_DISTANCE_KM,
) -> pd.DataFrame:
    """Build corridor edges as a local chain along the same route.

    The goal is to represent corridor continuity as A-B-C rather than
    a denser local clique A-B, A-C, B-C. Information can still travel
    across multiple hops in the graph model.
    """
    route_groups = nodes.dropna(subset=["route_key", "latitude", "longitude"]).copy()
    route_groups = route_groups[valid_route_key(route_groups["route_key"])]
    edge_rows = []
    for route_key, group in route_groups.groupby("route_key"):
        node_ids = group["node_id"].tolist()
        if len(node_ids) < 2:
            continue
        lat = group["latitude"].astype(float).to_numpy()
        lon = group["longitude"].astype(float).to_numpy()
        pair_dist = haversine_distance(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        if group["MILEPOINT"].notna().sum() >= 2:
            order = np.argsort(group["MILEPOINT"].fillna(np.inf).to_numpy())
        else:
            # Fallback when milepoint is missing: use a simple geographic
            # ordering so we still form a corridor-like chain rather than a clique.
            anchor_lon = group["longitude"].astype(float).mean()
            anchor_lat = group["latitude"].astype(float).mean()
            anchor_dist = haversine_distance(
                group["latitude"].astype(float).to_numpy(),
                group["longitude"].astype(float).to_numpy(),
                np.full(len(group), anchor_lat),
                np.full(len(group), anchor_lon),
            )
            order = np.argsort(anchor_dist)

        ordered_ids = [node_ids[i] for i in order]
        ordered_pair_dist = pair_dist[np.ix_(order, order)]
        for pos in range(len(ordered_ids) - 1):
            source = ordered_ids[pos]
            target = ordered_ids[pos + 1]
            distance_km = float(ordered_pair_dist[pos, pos + 1])
            if not np.isfinite(distance_km) or distance_km > max_distance_km:
                continue
            edge_rows.append((*sorted([source, target]), route_key, distance_km))
    edge_df = pd.DataFrame(edge_rows, columns=["source", "target", "route_key", "distance_km"]).drop_duplicates()
    edge_df["edge_type"] = "same_route"
    log(f"Built {len(edge_df)} same_route edges")
    return edge_df


def build_functional_edges(
    nodes: pd.DataFrame,
    max_neighbors: int = SIMILARITY_K,
    max_distance_km: float = SIMILARITY_MAX_DISTANCE_KM,
) -> pd.DataFrame:
    """Build sparse similarity edges instead of a full functional-class clique."""
    groups = nodes.dropna(subset=["functional_class", "latitude", "longitude", "state_code"]).copy()
    traffic_cols = [c for c in nodes.columns if c.startswith("traffic_") and np.issubdtype(nodes[c].dtype, np.number)]
    climate_cols = [
        c
        for c in nodes.columns
        if c.startswith(("temp_bind_", "humid_", "precip_", "wind_", "solar_", "temp_year_"))
        and np.issubdtype(nodes[c].dtype, np.number)
    ]
    structural_cols = [c for c in ["NO_OF_LANES", "LANE_WIDTH", "SECTION_LENGTH"] if c in nodes.columns]
    edge_rows = []
    for (_, func), group in groups.groupby(["state_code", "functional_class"]):
        node_ids = group["node_id"].tolist()
        if len(node_ids) < 2:
            continue
        lat = group["latitude"].astype(float).to_numpy()
        lon = group["longitude"].astype(float).to_numpy()
        pair_dist = haversine_distance(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        np.fill_diagonal(pair_dist, np.inf)
        group = group.reset_index(drop=True)
        for i, source in enumerate(node_ids):
            candidates = np.where(pair_dist[i] <= max_distance_km)[0]
            if len(candidates) == 0:
                continue
            scored: list[tuple[float, str, float, float, float, float]] = []
            for j in candidates:
                target = node_ids[j]
                left = group.iloc[i]
                right = group.iloc[j]
                traffic_similarity = similarity_from_series(left[traffic_cols], right[traffic_cols]) if traffic_cols else 0.5
                climate_similarity = similarity_from_series(left[climate_cols], right[climate_cols]) if climate_cols else 0.5
                pavement_similarity = similarity_from_series(left[structural_cols], right[structural_cols]) if structural_cols else 0.5
                spatial_score = float(np.exp(-pair_dist[i, j] / max_distance_km))
                score = (
                    0.40 * spatial_score
                    + 0.15 * traffic_similarity
                    + 0.20 * climate_similarity
                    + 0.25 * pavement_similarity
                )
                scored.append((score, target, float(pair_dist[i, j]), traffic_similarity, climate_similarity, pavement_similarity))
            scored.sort(key=lambda item: (-item[0], item[2]))
            for score, target, distance_km, traffic_similarity, climate_similarity, pavement_similarity in scored[:max_neighbors]:
                source_id, target_id = sorted([source, target])
                edge_rows.append(
                    (
                        source_id,
                        target_id,
                        func,
                        distance_km,
                        traffic_similarity,
                        climate_similarity,
                        pavement_similarity,
                        score,
                    )
                )
    edge_df = pd.DataFrame(
        edge_rows,
        columns=[
            "source",
            "target",
            "functional_class",
            "distance_km",
            "traffic_similarity",
            "climate_similarity",
            "pavement_similarity",
            "edge_weight",
        ],
    ).drop_duplicates(subset=["source", "target", "functional_class"])
    edge_df["edge_type"] = "same_functional_class"
    log(f"Built {len(edge_df)} same_functional_class edges")
    return edge_df


def add_edge_weight_views(edges: pd.DataFrame) -> pd.DataFrame:
    edge_df = edges.copy()
    distance = pd.to_numeric(edge_df.get("distance_km"), errors="coerce").fillna(999.0)
    edge_df["spatial_score"] = np.exp(-distance / SPATIAL_MAX_DISTANCE_KM)
    edge_df["route_score"] = np.where(edge_df["edge_type"].eq("same_route"), 1.0, 0.0)
    if "traffic_similarity" not in edge_df.columns:
        edge_df["traffic_similarity"] = 0.5
    if "climate_similarity" not in edge_df.columns:
        edge_df["climate_similarity"] = 0.5
    if "pavement_similarity" not in edge_df.columns:
        edge_df["pavement_similarity"] = 0.5
    if "edge_weight" not in edge_df.columns:
        edge_df["edge_weight"] = (
            0.40 * edge_df["spatial_score"]
            + 0.25 * edge_df["route_score"]
            + 0.15 * edge_df["traffic_similarity"]
            + 0.10 * edge_df["climate_similarity"]
            + 0.10 * edge_df["pavement_similarity"]
        )
    edge_df["weight_deterioration"] = (
        0.35 * edge_df["spatial_score"]
        + 0.15 * edge_df["route_score"]
        + 0.20 * edge_df["traffic_similarity"]
        + 0.20 * edge_df["climate_similarity"]
        + 0.10 * edge_df["pavement_similarity"]
    )
    edge_df["weight_disruption"] = (
        0.30 * edge_df["spatial_score"]
        + 0.40 * edge_df["route_score"]
        + 0.20 * pd.to_numeric(edge_df.get("diversion_potential"), errors="coerce").fillna(0.0)
        + 0.10 * edge_df["traffic_similarity"]
    )
    return edge_df


def augment_edges(edge_df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    log("Augmenting edges with node metadata")
    node_meta = nodes.set_index("node_id")[["state_code", "shrp_id", "route_key"]]
    edge_df = edge_df.copy()
    edge_df["source_state"] = edge_df["source"].map(node_meta["state_code"])
    edge_df["target_state"] = edge_df["target"].map(node_meta["state_code"])
    edge_df["same_state"] = edge_df["source_state"] == edge_df["target_state"]
    edge_df["same_route_key"] = edge_df["source"].map(node_meta["route_key"]) == edge_df["target"].map(node_meta["route_key"])
    log("Edge augmentation complete")
    return edge_df


def augment_diversion(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    log("Computing diversion potential for edges")
    edge_df = edges.copy()
    node_meta = nodes.set_index("node_id")
    traffic_cols = [
        c for c in node_meta.columns
        if c.startswith("traffic_") and np.issubdtype(node_meta[c].dtype, np.number)
    ]
    node_meta = node_meta.assign(
        traffic_mean=node_meta[traffic_cols].mean(axis=1, skipna=True) if traffic_cols else np.nan
    )

    source_node = node_meta[["route_key", "functional_class", "traffic_mean"]].rename(
        columns={
            "route_key": "source_route_key",
            "functional_class": "source_functional_class",
            "traffic_mean": "source_traffic",
        }
    )
    target_node = node_meta[["route_key", "functional_class", "traffic_mean"]].rename(
        columns={
            "route_key": "target_route_key",
            "functional_class": "target_functional_class",
            "traffic_mean": "target_traffic",
        }
    )

    edge_df = edge_df.merge(source_node, left_on="source", right_index=True, how="left")
    edge_df = edge_df.merge(target_node, left_on="target", right_index=True, how="left")

    edge_df["same_route"] = edge_df["source_route_key"] == edge_df["target_route_key"]
    edge_df["same_functional"] = edge_df["source_functional_class"] == edge_df["target_functional_class"]

    source_traffic = edge_df["source_traffic"]
    target_traffic = edge_df["target_traffic"]
    similarity = pd.Series(0.5, index=edge_df.index)
    both_traffic = source_traffic.notna() & target_traffic.notna()
    if both_traffic.any():
        similarity.loc[both_traffic] = 1.0 - (
            source_traffic.loc[both_traffic].sub(target_traffic.loc[both_traffic]).abs()
            / np.maximum(np.maximum(source_traffic.loc[both_traffic].abs(), target_traffic.loc[both_traffic].abs()), 1.0)
        )
    similarity = similarity.clip(lower=0.25, upper=1.0)

    base = np.where(edge_df["same_route"], 0.85, 0.55)
    score = base * similarity
    if "distance_km" in edge_df.columns:
        factor = np.clip(1.0 - edge_df["distance_km"].astype(float).fillna(80.0) / 80.0, 0.35, 1.0)
        score = score * factor
    score = np.where(edge_df["same_route"] | edge_df["same_functional"], score, 0.0)
    edge_df["diversion_potential"] = np.minimum(score, 1.0).astype(float)

    edge_df = edge_df.drop(columns=[
        "source_route_key",
        "target_route_key",
        "source_functional_class",
        "target_functional_class",
        "source_traffic",
        "target_traffic",
        "same_route",
        "same_functional",
    ], errors="ignore")
    log("Diversion potential computed")
    return edge_df


def build_project_conflicts(projects: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
    """Find simultaneous projects on adjacent sections."""
    log("Building project conflict edges")
    projects = projects.copy()
    projects["end_date"] = projects["traffic_open_date"].fillna(projects["construction_date"])
    projects = projects.dropna(subset=["construction_date"])
    projects = projects.sort_values(["node_id", "construction_date"])

    adjacency = edge_df[edge_df["edge_type"].isin(["spatial", "same_route"])][["source", "target"]].drop_duplicates()
    node_to_projects = projects.groupby("node_id").apply(lambda g: g.to_dict("records")).to_dict()
    conflict_rows = []
    for _, row in adjacency.iterrows():
        source_projects = node_to_projects.get(row["source"], [])
        target_projects = node_to_projects.get(row["target"], [])
        for ps in source_projects:
            for pt in target_projects:
                start = max(ps["construction_date"], pt["construction_date"])
                end = min(ps["end_date"], pt["end_date"])
                if pd.notna(start) and pd.notna(end) and end >= start:
                    conflict_rows.append(
                        {
                            "source_project_id": ps["project_id"],
                            "target_project_id": pt["project_id"],
                            "source_node": row["source"],
                            "target_node": row["target"],
                            "overlap_start": start,
                            "overlap_end": end,
                            "edge_type": "project_conflict",
                        }
                    )
    if not conflict_rows:
        return pd.DataFrame(columns=["source_project_id", "target_project_id", "source_node", "target_node", "overlap_start", "overlap_end", "edge_type"])
    return pd.DataFrame(conflict_rows)


def build_node_conflict_edges(projects: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    log("Building node-level project conflict summary")
    projects = projects.copy()
    projects["end_date"] = projects["traffic_open_date"].fillna(projects["construction_date"])
    projects = projects.dropna(subset=["construction_date"])

    adjacency = edges[edges["edge_type"].isin(["spatial", "same_route"])][["source", "target"]].drop_duplicates()
    node_to_projects = projects.groupby("node_id").apply(lambda g: g.to_dict("records")).to_dict()

    conflict_rows = []
    for _, row in adjacency.iterrows():
        source_projects = node_to_projects.get(row["source"], [])
        target_projects = node_to_projects.get(row["target"], [])
        for ps in source_projects:
            for pt in target_projects:
                start = max(ps["construction_date"], pt["construction_date"])
                end = min(ps["end_date"], pt["end_date"])
                if pd.notna(start) and pd.notna(end) and end >= start:
                    conflict_rows.append(
                        {
                            "source_node": row["source"],
                            "target_node": row["target"],
                            "conflict_count": 1,
                            "overlap_days": (end - start).days,
                            "edge_type": "node_project_conflict",
                        }
                    )

    if not conflict_rows:
        return pd.DataFrame(columns=["source_node", "target_node", "conflict_count", "overlap_days", "edge_type"])

    conflicts = pd.DataFrame(conflict_rows)
    summary = (
        conflicts.groupby(["source_node", "target_node"], as_index=False)
        .agg(
            conflict_count=("conflict_count", "sum"),
            total_overlap_days=("overlap_days", "sum"),
            max_overlap_days=("overlap_days", "max"),
        )
    )
    summary["edge_type"] = "node_project_conflict"
    return summary


def augment_diversion(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    log("Computing diversion potential for edges")
    edge_df = edges.copy()
    node_meta = nodes.set_index("node_id")
    traffic_cols = [
        c for c in node_meta.columns
        if c.startswith("traffic_") and np.issubdtype(node_meta[c].dtype, np.number)
    ]

    node_meta = node_meta.assign(
        traffic_mean=node_meta[traffic_cols].mean(axis=1, skipna=True) if traffic_cols else np.nan
    )
    source_node = node_meta[["route_key", "functional_class", "traffic_mean"]].rename(
        columns={
            "route_key": "source_route_key",
            "functional_class": "source_functional_class",
            "traffic_mean": "source_traffic",
        }
    )
    target_node = node_meta[["route_key", "functional_class", "traffic_mean"]].rename(
        columns={
            "route_key": "target_route_key",
            "functional_class": "target_functional_class",
            "traffic_mean": "target_traffic",
        }
    )

    edge_df = edge_df.merge(source_node, left_on="source", right_index=True, how="left")
    edge_df = edge_df.merge(target_node, left_on="target", right_index=True, how="left")

    edge_df["same_route"] = edge_df["source_route_key"] == edge_df["target_route_key"]
    edge_df["same_functional"] = edge_df["source_functional_class"] == edge_df["target_functional_class"]

    source_traffic = edge_df["source_traffic"]
    target_traffic = edge_df["target_traffic"]
    similarity = pd.Series(0.5, index=edge_df.index)
    both_traffic = source_traffic.notna() & target_traffic.notna()
    if both_traffic.any():
        similarity.loc[both_traffic] = 1.0 - (
            source_traffic.loc[both_traffic].sub(target_traffic.loc[both_traffic]).abs()
            / np.maximum(np.maximum(source_traffic.loc[both_traffic].abs(), target_traffic.loc[both_traffic].abs()), 1.0)
        )
    similarity = similarity.clip(lower=0.25, upper=1.0)

    base = np.where(edge_df["same_route"], 0.85, 0.55)
    score = base * similarity
    if "distance_km" in edge_df.columns:
        factor = np.clip(1.0 - edge_df["distance_km"].astype(float).fillna(80.0) / 80.0, 0.35, 1.0)
        score = score * factor
    score = np.where(edge_df["same_route"] | edge_df["same_functional"], score, 0.0)
    edge_df["diversion_potential"] = np.minimum(score, 1.0).astype(float)

    edge_df = edge_df.drop(columns=[
        "source_route_key",
        "target_route_key",
        "source_functional_class",
        "target_functional_class",
        "source_traffic",
        "target_traffic",
        "same_route",
        "same_functional",
    ], errors="ignore")
    log("Diversion potential computed")
    return edge_df


def assemble_graph() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = prepare_node_table()
    climate = load_climate_features()
    distress = load_distress_features()
    traffic = load_traffic_features()

    nodes = nodes.merge(climate, on="node_id", how="left")
    if "merra_grid_elevation" in nodes.columns and "elevation" in nodes.columns:
        nodes["elevation"] = pd.to_numeric(nodes["elevation"], errors="coerce").fillna(nodes["merra_grid_elevation"])
    nodes = nodes.merge(distress, on="node_id", how="left")
    nodes = nodes.merge(traffic, on="node_id", how="left")

    global CLIMATE_REPORT
    if CLIMATE_REPORT:
        CLIMATE_REPORT["nodes_with_merra_id"] = int(nodes["merra_id"].notna().sum()) if "merra_id" in nodes.columns else 0
        CLIMATE_REPORT["coverage_pct"] = float(100.0 * nodes["merra_id"].notna().mean()) if "merra_id" in nodes.columns else 0.0

    spatial_edges = build_spatial_edges(nodes)
    route_edges = build_route_edges(nodes)
    functional_edges = build_functional_edges(nodes)

    edges = pd.concat([spatial_edges, route_edges, functional_edges], ignore_index=True, sort=False)
    log(f"Concatenated {len(edges)} raw edges")
    edges = edges.drop_duplicates(subset=["source", "target", "edge_type"])
    log(f"Deduplicated edges: {len(edges)} remain")
    edges = augment_edges(edges, nodes)
    edges = augment_diversion(edges, nodes)
    edges = add_edge_weight_views(edges)

    projects = load_project_events()
    projects = projects[projects["node_id"].isin(set(nodes["node_id"].astype(str)))].copy()
    conflicts = build_project_conflicts(projects, edges)
    node_conflicts = build_node_conflict_edges(projects, edges)
    return nodes, edges, projects, conflicts, node_conflicts


def save_graph_data(nodes: pd.DataFrame, edges: pd.DataFrame, projects: pd.DataFrame, conflicts: pd.DataFrame, node_conflicts: pd.DataFrame):
    GRAPH_DIR.mkdir(exist_ok=True)

    def save_table(df: pd.DataFrame, path: Path):
        df_to_save = df.copy()
        for col in df_to_save.select_dtypes(include=["object", "string"]).columns:
            df_to_save[col] = df_to_save[col].astype("string")
        df_to_save.to_parquet(path, index=False)

    save_table(nodes, GRAPH_DIR / "nodes.parquet")
    save_table(edges, GRAPH_DIR / "edges.parquet")
    save_table(projects, GRAPH_DIR / "projects.parquet")
    save_table(conflicts, GRAPH_DIR / "project_conflicts.parquet")
    save_table(node_conflicts, GRAPH_DIR / "node_project_conflicts.parquet")
    nodes.to_csv(GRAPH_DIR / "nodes.csv", index=False)
    edges.to_csv(GRAPH_DIR / "edges.csv", index=False)
    projects.to_csv(GRAPH_DIR / "projects.csv", index=False)
    conflicts.to_csv(GRAPH_DIR / "project_conflicts.csv", index=False)
    node_conflicts.to_csv(GRAPH_DIR / "node_project_conflicts.csv", index=False)

    from pandas.api.types import is_numeric_dtype

    feature_cols = [
        c for c in nodes.columns
        if c == "node_id" or is_numeric_dtype(nodes[c])
    ]
    model_nodes = nodes[feature_cols].copy()
    model_nodes.to_parquet(GRAPH_DIR / "node_features.parquet", index=False)

    edge_index_cols = [
        c
        for c in [
            "source",
            "target",
            "edge_type",
            "distance_km",
            "diversion_potential",
            "same_state",
            "same_route_key",
            "edge_weight",
            "weight_deterioration",
            "weight_disruption",
        ]
        if c in edges.columns
    ]
    edges[edge_index_cols].to_csv(GRAPH_DIR / "edge_index.csv", index=False)
    deterioration_edges = edges[edges["edge_type"].isin(["spatial", "same_route", "same_functional_class"])].copy()
    disruption_edges = edges[edges["edge_type"].isin(["spatial", "same_route"])].copy()
    deterioration_edges.to_csv(GRAPH_DIR / "edges_deterioration.csv", index=False)
    disruption_edges.to_csv(GRAPH_DIR / "edges_disruption.csv", index=False)


def build_networkx_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in nodes.iterrows():
        attrs = row.drop(labels=["node_id"]).to_dict()
        G.add_node(row["node_id"], **attrs)
    for _, row in edges.iterrows():
        source = row["source"]
        target = row["target"]
        attrs = row.drop(labels=["source", "target"]).to_dict()
        G.add_edge(source, target, **attrs)
    return G


def main():
    log("Starting graph assembly")
    nodes, edges, projects, conflicts, node_conflicts = assemble_graph()
    log("Saving graph data")
    save_graph_data(nodes, edges, projects, conflicts, node_conflicts)
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Projects: {len(projects)}")
    print(f"Project conflicts: {len(conflicts)}")
    print(f"Node conflict edges: {len(node_conflicts)}")
    if CLIMATE_REPORT:
        print(f"Climate columns before enrichment: {CLIMATE_REPORT['climate_cols_before']}")
        print(f"Climate columns after enrichment: {CLIMATE_REPORT['climate_cols_after']}")
        print(f"New climate columns added: {CLIMATE_REPORT['new_climate_cols_added']}")
        print(f"MERRA-covered nodes: {CLIMATE_REPORT['nodes_with_merra_id']}")
        print(f"MERRA coverage: {CLIMATE_REPORT['coverage_pct']:.2f}%")
    G = build_networkx_graph(edges, nodes)
    print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


if __name__ == "__main__":
    main()
