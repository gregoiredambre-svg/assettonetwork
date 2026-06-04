from __future__ import annotations

import json
from pathlib import Path
import textwrap

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"
RAW_DATA_DIR = ROOT / "Research Data"

st.set_page_config(
    page_title="From Asset to Network: Graph-Based Road Maintenance Explorer",
    layout="wide",
)

APP_TITLE = "From Asset to Network: Graph-Based Road Maintenance Explorer"

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: #f7f7f7;
        border: 1px solid #e6e6e6;
        border-radius: 14px;
        padding: 0.85rem 1rem;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.95rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        line-height: 1.1 !important;
    }
    .story-lead {
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

GRAPH_VARIANTS = {
    "spatial": "Spatial only",
    "spatial_route": "Spatial + Route",
    "full_refined": "Full refined",
}

GRAPH_VARIANT_STORY = {
    "spatial": "Sections are linked when they are geographically close. This is the simplest graph and the easiest one to interpret as local disruption neighbourhoods.",
    "spatial_route": "Sections are linked when they are geographically close, and extra corridor links are added along the same route so that A-B-C corridor continuity is visible in the graph.",
    "full_refined": "This graph keeps the spatial and route links, then adds sparse filtered similarity links between sections that look alike in road role, traffic, climate, and structure.",
}

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

FRIENDLY_TARGETS = {
    "delta_vht_proxy": "Extra travel-time proxy",
    "connectivity_loss_pct": "Connectivity loss share",
    "disconnected_od_pct": "Disconnected OD share",
    "disruption_score": "Overall disruption score",
    "pred_delta_vht_proxy": "Predicted extra travel-time proxy",
    "pred_connectivity_loss_pct": "Predicted connectivity loss share",
    "pred_disconnected_od_pct": "Predicted disconnected OD share",
    "pred_disruption_score": "Predicted overall disruption score",
}

TREATMENT_LABELS = {
    "crack_sealing": "Crack sealing",
    "asphalt_overlay": "Asphalt overlay",
    "seal_coat": "Seal coat",
    "patching": "Patching",
    "grinding": "Grinding",
    "shoulder_restoration": "Shoulder restoration",
    "reconstruction_or_major_rehab": "Reconstruction / major rehab",
    "longitudinal_subdrains_or_drainage": "Drainage / longitudinal subdrains",
    "other_maintenance": "Other maintenance",
    "unknown": "Unknown / mixed",
}

CORE_TERMS_MARKDOWN = """
**Useful vocabulary**

- **LTPP**: Long-Term Pavement Performance programme. This is the source of the monitored road sections.
- **MERRA**: a gridded climate dataset used to attach temperature, precipitation, humidity, wind, and solar exposure to each section.
- **RF**: Random Forest, a strong non-linear baseline model that predicts from section-level features only.
- **GCN**: Graph Convolutional Network, a graph model that lets neighbouring sections share information.
- **R²**: coefficient of determination. Closer to `1` means better predictive fit; around `0` means weak explanatory power.
- **OD pair**: an **origin-destination pair**, meaning a trip from one important road section to another.
- **ESAL / GESAL**: traffic loading indicators summarising how much heavy-vehicle damage the pavement is expected to carry.
- **AADTT**: Average Annual Daily Truck Traffic.
"""

YEARLY_SERIES_LABELS = {
    "temp_year_temp_avg": "Temperature average",
    "temp_year_temp_mean_avg": "Mean air temperature",
    "humid_rel_hum_avg_avg": "Relative humidity",
    "precip_precipitation": "Precipitation",
    "precip_evaporation": "Evaporation",
    "precip_precip_days": "Precipitation days",
    "wind_wind_velocity_avg": "Wind velocity",
    "solar_cloud_cover_avg": "Cloud cover",
    "solar_shortwave_surface_avg": "Shortwave solar exposure",
    "ANNUAL_ESAL_TREND": "Annual ESAL",
    "ANNUAL_GESAL_TREND": "Annual GESAL",
    "AADTT_ALL_TRUCKS_TREND": "AADTT all trucks",
    "ANNUAL_TRUCK_VOLUME_TREND": "Annual truck volume",
    "CMLTV_VOL_VEH_CLASS_9_TREND": "Class 9 cumulative volume",
}

DISTRESS_LABELS = {
    "HPMS16_CRACKING_PERCENT_AC": "HPMS cracking (%)",
    "MEPDG_CRACKING_PERCENT_AC": "MEPDG cracking (%)",
    "MEPDG_TRANS_CRACK_LENGTH_AC": "MEPDG transverse cracking length",
    "GATOR_CRACK_A": "Alligator cracking area",
    "LONG_CRACK_WP_L": "Longitudinal cracking in wheel path",
    "LONG_CRACK_NWP_L": "Longitudinal cracking outside wheel path",
    "LONG_CRACK_WP_SEAL_L": "Sealed longitudinal cracking in wheel path",
    "LONG_CRACK_NWP_SEAL_L": "Sealed longitudinal cracking outside wheel path",
    "TRANS_CRACK_L": "Transverse cracking length",
    "TRANS_CRACK_SEAL_L": "Sealed transverse cracking length",
    "PATCH_A": "Patched area",
    "POTHOLES_A": "Pothole area",
}


def tagged_graph_path(stem: str, variant: str, suffix: str) -> Path:
    if variant == "full_refined":
        tagged = DATA_DIR / f"{stem}_full_refined{suffix}"
        default = DATA_DIR / f"{stem}{suffix}"
        return tagged if tagged.exists() else default
    return DATA_DIR / f"{stem}_{variant}{suffix}"


def state_code_to_label(code: object) -> str:
    code_str = str(code)
    return f"{STATE_NAMES.get(code_str, f'State {code_str}')} ({code_str})"


def compact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.2f}" if value % 1 else f"{int(value)}"


def safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


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


def build_node_id_join_from_parts(state_code: object, shrp_id: object) -> str | None:
    state = str(state_code).strip() if not pd.isna(state_code) else ""
    shrp = normalize_shrp_id(shrp_id)
    if not state or shrp is None:
        return None
    return f"{state}_{shrp}"


def build_node_id_join_from_node_id(node_id: object) -> str | None:
    if pd.isna(node_id):
        return None
    text = str(node_id).strip()
    if "_" not in text:
        return None
    state_code, shrp_id = text.split("_", 1)
    return build_node_id_join_from_parts(state_code, shrp_id)


def node_label(row: pd.Series) -> str:
    node_id = str(row.get("node_id", ""))
    state_name = STATE_NAMES.get(str(row.get("state_code", "")), f"State {row.get('state_code', '')}")
    route_key = row.get("route_key")
    route_txt = str(route_key) if pd.notna(route_key) and str(route_key).strip() else "no route"
    func = row.get("functional_class")
    func_txt = str(func) if pd.notna(func) and str(func).strip() else "no class"
    return f"{node_id} — {state_name} — {route_txt} — class {func_txt}"


@st.cache_data(show_spinner=False)
def load_nodes() -> pd.DataFrame:
    nodes = pd.read_csv(DATA_DIR / "nodes.csv", low_memory=False)
    nodes = nodes.assign(
        node_id=nodes["node_id"].astype(str),
        state_code=nodes["state_code"].astype(str),
    ).copy()
    nodes = nodes.assign(
        state_name=nodes["state_code"].map(lambda code: STATE_NAMES.get(str(code), f"State {code}")),
        node_id_join=nodes["node_id"].map(build_node_id_join_from_node_id),
    ).copy()
    return nodes.dropna(subset=["latitude", "longitude"]).copy()


@st.cache_data(show_spinner=False)
def load_projects() -> pd.DataFrame:
    projects = pd.read_csv(DATA_DIR / "projects.csv", low_memory=False)
    projects["node_id"] = projects["node_id"].astype(str)
    return projects


@st.cache_data(show_spinner=False)
def load_core_edges() -> pd.DataFrame:
    edges = pd.read_csv(DATA_DIR / "edges.csv", low_memory=False)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["edge_type"] = edges["edge_type"].astype(str)
    return edges


def filter_core_edges_for_variant(edges: pd.DataFrame, variant: str) -> pd.DataFrame:
    if edges.empty or "edge_type" not in edges.columns:
        return edges.copy()
    if variant == "spatial":
        keep = {"spatial"}
    elif variant == "spatial_route":
        keep = {"spatial", "same_route"}
    else:
        keep = {"spatial", "same_route", "same_functional_class"}
    return edges[edges["edge_type"].isin(keep)].copy()


@st.cache_data(show_spinner=False)
def load_variant_bundle(variant: str) -> dict[str, object]:
    bundle: dict[str, object] = {}
    bundle["network_edges"] = pd.read_csv(tagged_graph_path("network_edges_research", variant, ".csv"), low_memory=False)
    bundle["od_pairs"] = pd.read_csv(tagged_graph_path("network_od_pairs", variant, ".csv"), low_memory=False)
    bundle["scenarios"] = pd.read_csv(tagged_graph_path("network_scenarios", variant, ".csv"), low_memory=False)
    bundle["predictions"] = pd.read_csv(tagged_graph_path("network_scenario_predictions", variant, ".csv"), low_memory=False)
    with open(tagged_graph_path("network_model_metrics", variant, ".json"), "r", encoding="utf-8") as fh:
        bundle["static_metrics"] = json.load(fh)
    return bundle


@st.cache_data(show_spinner=False)
def load_report_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / name, low_memory=False)


@st.cache_data(show_spinner=False)
def load_report_json(name: str) -> dict:
    with open(REPORT_DIR / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_optional_report_csv(name: str) -> pd.DataFrame:
    path = REPORT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_optional_report_json(name: str) -> dict:
    path = REPORT_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_optional_graph_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_inspector_yearly_panel() -> pd.DataFrame:
    nodes = load_nodes()[["node_id", "node_id_join", "state_code", "state_name", "route_key", "functional_class", "merra_id"]].copy()

    traffic_specs = {
        "TRF_TREND": ["ANNUAL_ESAL_TREND", "ANNUAL_GESAL_TREND"],
        "TRF_TREND_1": ["AADTT_ALL_TRUCKS_TREND", "ANNUAL_TRUCK_VOLUME_TREND"],
        "TRF_TREND_2": ["CMLTV_VOL_VEH_CLASS_9_TREND"],
    }
    traffic_path = RAW_DATA_DIR / "Annual Traffic Inputs Over Time.xlsx"
    traffic_parts: list[pd.DataFrame] = []
    for sheet_name, value_cols in traffic_specs.items():
        raw = pd.read_excel(traffic_path, sheet_name=sheet_name, usecols=["STATE_CODE", "SHRP_ID", "YEAR", *value_cols])
        raw["node_id_join"] = raw.apply(lambda row: build_node_id_join_from_parts(row["STATE_CODE"], row["SHRP_ID"]), axis=1)
        raw["YEAR"] = pd.to_numeric(raw["YEAR"], errors="coerce").astype("Int64")
        for col in value_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        traffic_parts.append(raw[["node_id_join", "YEAR", *value_cols]].groupby(["node_id_join", "YEAR"], as_index=False).mean())

    traffic = traffic_parts[0]
    for part in traffic_parts[1:]:
        traffic = traffic.merge(part, on=["node_id_join", "YEAR"], how="outer")

    climate_root = RAW_DATA_DIR / "MERRA - Temperature, Humidity, Precipitation, Wind, Solar"
    grid = pd.read_excel(climate_root / "GENERAL" / "MERRA_GRID_SECTION.xlsx", sheet_name="MERRA_GRID_SECTION")
    grid["node_id_join"] = grid.apply(lambda row: build_node_id_join_from_parts(row.get("STATE_CODE"), row.get("SHRP_ID")), axis=1)
    grid = grid.rename(columns={"MERRA_ID": "merra_id"})
    grid["merra_id"] = grid["merra_id"].astype(str).str.strip()
    grid = grid[["node_id_join", "merra_id"]].drop_duplicates(subset=["node_id_join"])

    climate_specs = [
        ("HUMIDITY/MERRA_HUMID_YEAR.xlsx", "MERRA_HUMID_YEAR", ["REL_HUM_AVG_AVG"], "humid_"),
        ("PRECIPITATION/MERRA_PRECIP_YEAR.xlsx", "MERRA_PRECIP_YEAR", ["PRECIPITATION", "EVAPORATION", "PRECIP_DAYS"], "precip_"),
        ("WIND/MERRA_WIND_YEAR.xlsx", "MERRA_WIND_YEAR", ["WIND_VELOCITY_AVG"], "wind_"),
        ("SOLAR/MERRA_SOLAR_YEAR.xlsx", "MERRA_SOLAR_YEAR", ["CLOUD_COVER_AVG", "SHORTWAVE_SURFACE_AVG"], "solar_"),
        ("TEMPERATURE/MERRA_TEMP_YEAR.xlsx", "MERRA_TEMP_YEAR", ["TEMP_AVG", "TEMP_MEAN_AVG", "FREEZE_INDEX", "FREEZE_THAW"], "temp_year_"),
    ]
    climate_frames: list[pd.DataFrame] = []
    for rel_path, sheet_name, value_cols, prefix in climate_specs:
        raw = pd.read_excel(climate_root / rel_path, sheet_name=sheet_name)
        raw = raw.rename(columns={"MERRA_ID": "merra_id"})
        raw["merra_id"] = raw["merra_id"].astype(str).str.strip()
        raw["YEAR"] = pd.to_numeric(raw["YEAR"], errors="coerce").astype("Int64")
        for col in value_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        renamed = raw[["merra_id", "YEAR", *value_cols]].groupby(["merra_id", "YEAR"], as_index=False).mean()
        renamed = renamed.rename(columns={col: f"{prefix}{col.lower()}" for col in value_cols})
        climate_frames.append(renamed)

    annual_climate = climate_frames[0]
    for part in climate_frames[1:]:
        annual_climate = annual_climate.merge(part, on=["merra_id", "YEAR"], how="outer")
    annual_climate = grid.merge(annual_climate, on="merra_id", how="left")

    yearly = traffic.merge(annual_climate, on=["node_id_join", "YEAR"], how="outer")
    panel = nodes.merge(yearly, on="node_id_join", how="left")
    panel["YEAR"] = pd.to_numeric(panel["YEAR"], errors="coerce").astype("Int64")
    return panel


@st.cache_data(show_spinner=False)
def load_inspector_distress() -> pd.DataFrame:
    distress_path = RAW_DATA_DIR / "Analysis Ready Distress.xlsx"
    specs = {
        "ANALYSIS_DIS_AC": [
            "HPMS16_CRACKING_PERCENT_AC",
            "GATOR_CRACK_A",
            "LONG_CRACK_WP_L",
            "LONG_CRACK_NWP_L",
            "LONG_CRACK_WP_SEAL_L",
            "LONG_CRACK_NWP_SEAL_L",
            "TRANS_CRACK_L",
            "TRANS_CRACK_SEAL_L",
            "PATCH_A",
            "POTHOLES_A",
        ],
        "ANALYSIS_DIS_JPCC": ["TRANS_CRACK_L", "TRANS_CRACK_SEAL_L"],
        "ANALYSIS_DIS_CRCP": ["TRANS_CRACK_L"],
    }
    pieces: list[pd.DataFrame] = []
    for sheet_name, metric_cols in specs.items():
        raw = pd.read_excel(
            distress_path,
            sheet_name=sheet_name,
            usecols=["STATE_CODE", "SHRP_ID", "SURVEY_DATE", "CONSTRUCTION_NO", *metric_cols],
        )
        raw["node_id_join"] = raw.apply(lambda row: build_node_id_join_from_parts(row["STATE_CODE"], row["SHRP_ID"]), axis=1)
        node_lookup = load_nodes()[["node_id", "node_id_join"]].drop_duplicates(subset=["node_id_join"])
        raw = raw.merge(node_lookup, on="node_id_join", how="left")
        raw["SURVEY_DATE"] = pd.to_datetime(raw["SURVEY_DATE"], errors="coerce")
        raw["YEAR"] = raw["SURVEY_DATE"].dt.year.astype("Int64")
        raw["distress_source"] = sheet_name.replace("ANALYSIS_DIS_", "")
        for col in metric_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        pieces.append(raw)
    return pd.concat(pieces, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_inspector_treatment_events() -> pd.DataFrame:
    events = pd.read_csv(REPORT_DIR / "experiment_section_event_table.csv", low_memory=False)
    events["node_id"] = events["node_id"].astype(str)
    events["event_start_date"] = pd.to_datetime(events["event_start_date"], errors="coerce")
    events["event_end_date"] = pd.to_datetime(events["event_end_date"], errors="coerce")
    return events


@st.cache_data(show_spinner=False)
def add_project_counts(nodes: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    counts = projects.groupby("node_id").size().rename("project_count").reset_index()
    frame = nodes.merge(counts, on="node_id", how="left")
    frame["project_count"] = frame["project_count"].fillna(0).astype(int)
    return frame


def filter_by_states(df: pd.DataFrame, selected_states: list[str], state_col: str = "state_code") -> pd.DataFrame:
    if not selected_states:
        return df.copy()
    return df[df[state_col].astype(str).isin(selected_states)].copy()


def filter_edges_for_visible_nodes(edges: pd.DataFrame, visible_node_ids: set[str]) -> pd.DataFrame:
    if not visible_node_ids:
        return edges.copy()
    return edges[edges["source"].isin(visible_node_ids) & edges["target"].isin(visible_node_ids)].copy()


def scenario_nodes(closed_node_ids: str) -> list[str]:
    return [node for node in str(closed_node_ids).split(";") if node]


def filter_scenarios_to_states(frame: pd.DataFrame, selected_states: list[str], nodes: pd.DataFrame) -> pd.DataFrame:
    if not selected_states:
        return frame.copy()
    node_state = nodes.set_index("node_id")["state_code"].astype(str).to_dict()

    def valid(closed: str) -> bool:
        ids = scenario_nodes(closed)
        return bool(ids) and all(node_state.get(node) in selected_states for node in ids)

    return frame[frame["closed_node_ids"].map(valid)].copy()


def build_section_lookup(nodes: pd.DataFrame) -> dict[str, str]:
    temp = nodes[["node_id", "state_code", "route_key", "functional_class"]].drop_duplicates("node_id").copy()
    return {row["node_id"]: node_label(row) for _, row in temp.iterrows()}


def draw_node_map(
    nodes: pd.DataFrame,
    title: str,
    color_col: str | None = None,
    highlight_nodes: list[str] | None = None,
    height: int = 650,
) -> go.Figure:
    frame = nodes.copy()
    frame["highlight"] = frame["node_id"].isin(highlight_nodes or [])
    if color_col and color_col in frame.columns:
        fig = px.scatter_geo(
            frame,
            lat="latitude",
            lon="longitude",
            color=color_col,
            hover_name="node_id",
            custom_data=["node_id"],
            hover_data={"state_name": True, "route_key": True, "functional_class": True, "project_count": True},
            title=title,
            opacity=0.75,
        )
    else:
        fig = px.scatter_geo(
            frame,
            lat="latitude",
            lon="longitude",
            hover_name="node_id",
            custom_data=["node_id"],
            hover_data={"state_name": True, "route_key": True, "functional_class": True, "project_count": True},
            title=title,
            opacity=0.75,
        )
        fig.update_traces(marker=dict(color="#1f77b4", size=7))
    if highlight_nodes:
        focus = frame[frame["highlight"]]
        if not focus.empty:
            fig.add_trace(
                go.Scattergeo(
                    lat=focus["latitude"],
                    lon=focus["longitude"],
                    mode="markers",
                    marker=dict(size=11, color="#d62728", line=dict(color="white", width=1)),
                    name="Highlighted sections",
                    text=focus["node_id"],
                )
            )
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#f7f4ea",
        showlakes=False,
        showcountries=False,
        showsubunits=True,
        subunitcolor="white",
        coastlinecolor="rgba(0,0,0,0)",
        fitbounds="locations" if len(frame) < len(nodes) else False,
    )
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def assign_component_clusters(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    frame = nodes.copy()
    graph = nx.Graph()
    graph.add_nodes_from(frame["node_id"].astype(str).tolist())
    if not edges.empty:
        graph.add_edges_from(edges[["source", "target"]].astype(str).itertuples(index=False, name=None))

    component_map: dict[str, int] = {}
    component_sizes: dict[int, int] = {}
    ordered_components = sorted(nx.connected_components(graph), key=len, reverse=True)
    for idx, comp in enumerate(ordered_components, start=1):
        component_sizes[idx] = len(comp)
        for node_id in comp:
            component_map[str(node_id)] = idx
    frame["cluster_id"] = frame["node_id"].astype(str).map(component_map).fillna(-1).astype(int)
    frame["cluster_size"] = frame["cluster_id"].map(component_sizes).fillna(1).astype(int)
    frame["cluster_label"] = frame["cluster_id"].map(lambda value: f"Cluster {value}" if value > 0 else "Isolated")
    cluster_order = [f"Cluster {idx}" for idx in sorted(component_sizes, key=lambda idx: component_sizes[idx], reverse=True)]
    if (frame["cluster_id"] <= 0).any():
        cluster_order.append("Isolated")
    frame["cluster_label"] = pd.Categorical(frame["cluster_label"], categories=cluster_order, ordered=True)
    return frame


def draw_network_web(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    title: str,
    focus_node: str | None = None,
    max_edges: int | None = None,
    color_by_cluster: bool = False,
    height: int = 700,
) -> go.Figure:
    frame_nodes = nodes.copy()
    edge_frame = edges.copy()
    if not edge_frame.empty:
        sort_frame = edge_frame.copy()
        sort_frame["distance_km"] = pd.to_numeric(sort_frame.get("distance_km"), errors="coerce")
        sort_frame["role_priority"] = sort_frame.get("edge_role", "").map({"corridor": 0, "spatial": 1, "diversion": 2}).fillna(3)
        if focus_node and focus_node in set(frame_nodes["node_id"].astype(str)):
            temp_graph = nx.Graph()
            temp_graph.add_nodes_from(frame_nodes["node_id"].astype(str).tolist())
            temp_graph.add_edges_from(sort_frame[["source", "target"]].astype(str).itertuples(index=False, name=None))
            try:
                focus_component = nx.node_connected_component(temp_graph, focus_node)
            except Exception:
                focus_component = {focus_node}
            sort_frame["focus_component_edge"] = sort_frame["source"].astype(str).isin(focus_component) & sort_frame["target"].astype(str).isin(focus_component)
        else:
            sort_frame["focus_component_edge"] = False
        sort_frame = sort_frame.sort_values(
            by=["focus_component_edge", "role_priority", "distance_km"],
            ascending=[False, True, True],
            na_position="last",
        )
        if max_edges is not None:
            sort_frame = sort_frame.head(max_edges)
        edge_frame = sort_frame.drop(columns=[c for c in ["focus_component_edge", "role_priority"] if c in sort_frame.columns]).copy()
    if color_by_cluster:
        frame_nodes = assign_component_clusters(frame_nodes, edge_frame)
    fig = go.Figure()
    role_colors = {"spatial": "#1f77b4", "corridor": "#d62728", "diversion": "#2ca02c"}

    for role, group in edge_frame.groupby("edge_role"):
        xs, ys = [], []
        for row in group.itertuples(index=False):
            src = frame_nodes.loc[frame_nodes["node_id"] == row.source]
            dst = frame_nodes.loc[frame_nodes["node_id"] == row.target]
            if src.empty or dst.empty:
                continue
            xs += [float(src.iloc[0]["longitude"]), float(dst.iloc[0]["longitude"]), None]
            ys += [float(src.iloc[0]["latitude"]), float(dst.iloc[0]["latitude"]), None]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=2.6, color=role_colors.get(role, "#888")),
                opacity=0.7,
                name=role.title(),
                hoverinfo="skip",
            )
        )

    if color_by_cluster and "cluster_label" in frame_nodes.columns:
        cluster_order = list(frame_nodes["cluster_label"].cat.categories)
        cluster_fig = px.scatter(
            frame_nodes,
            x="longitude",
            y="latitude",
            color="cluster_label",
            category_orders={"cluster_label": cluster_order},
            hover_name="node_id",
            hover_data={"state_name": True, "route_key": True, "functional_class": True, "cluster_label": False},
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        for trace in cluster_fig.data:
            trace.marker.size = 7
            trace.marker.line = dict(color="white", width=0.4)
            trace.opacity = 0.82
            fig.add_trace(trace)
    else:
        fig.add_trace(
            go.Scatter(
                x=frame_nodes["longitude"],
                y=frame_nodes["latitude"],
                mode="markers",
                marker=dict(size=6, color="rgba(70,70,70,0.35)", line=dict(color="white", width=0.35)),
                text=frame_nodes["node_id"],
                customdata=frame_nodes[["state_name", "route_key", "functional_class"]],
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Route: %{customdata[1]}<br>Class: %{customdata[2]}<extra></extra>",
                name="Road sections",
            )
        )

    if focus_node:
        focus = frame_nodes[frame_nodes["node_id"] == focus_node]
        if not focus.empty:
            fig.add_trace(
                go.Scatter(
                    x=focus["longitude"],
                    y=focus["latitude"],
                    mode="markers",
                    marker=dict(size=14, color="#f4b400", symbol="star", line=dict(color="black", width=1)),
                    text=focus["node_id"],
                    name="Selected road section",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Relationship / cluster",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    if not frame_nodes.empty:
        lon_pad = max((frame_nodes["longitude"].max() - frame_nodes["longitude"].min()) * 0.08, 0.5)
        lat_pad = max((frame_nodes["latitude"].max() - frame_nodes["latitude"].min()) * 0.08, 0.5)
        fig.update_xaxes(range=[frame_nodes["longitude"].min() - lon_pad, frame_nodes["longitude"].max() + lon_pad])
        fig.update_yaxes(range=[frame_nodes["latitude"].min() - lat_pad, frame_nodes["latitude"].max() + lat_pad])
    return fig


def draw_cluster_map(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    title: str,
    height: int = 650,
    min_cluster_size: int = 1,
) -> go.Figure:
    frame = assign_component_clusters(nodes, edges)
    frame = frame[frame["cluster_size"] >= int(min_cluster_size)].copy()
    cluster_order = list(frame["cluster_label"].cat.categories)
    fig = px.scatter_geo(
        frame,
        lat="latitude",
        lon="longitude",
        color="cluster_label",
        category_orders={"cluster_label": cluster_order},
        hover_name="node_id",
        hover_data={"state_name": True, "route_key": True, "functional_class": True},
        title=title,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        opacity=0.82,
    )
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#f7f4ea",
        showlakes=False,
        showcountries=False,
        showsubunits=True,
        subunitcolor="white",
        coastlinecolor="rgba(0,0,0,0)",
    )
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=50, b=0), legend_title_text="Local cluster")
    return fig


def draw_local_network(nodes: pd.DataFrame, edges: pd.DataFrame, focus_node: str) -> go.Figure:
    one_hop = set()
    local_edges = edges[(edges["source"] == focus_node) | (edges["target"] == focus_node)].copy()
    for row in local_edges.itertuples(index=False):
        one_hop.add(str(row.source))
        one_hop.add(str(row.target))
    local_nodes = nodes[nodes["node_id"].isin(one_hop | {focus_node})].copy()
    return draw_network_web(local_nodes, local_edges, "Local section neighbourhood", focus_node=focus_node, max_edges=300)


def metric_card_row(metrics: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, help_text) in zip(cols, metrics):
        col.metric(label, value, help=help_text)


def wrap_multiline_label(text: object, width: int = 24) -> str:
    label = str(text).strip() if text is not None else ""
    if not label:
        return "Missing / not recorded"
    return "<br>".join(textwrap.wrap(label, width=width)) or label


def best_static_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    static_rows = comparison[comparison["model_family"] == "static"].copy()
    idx = static_rows.groupby("target")["gcn_test_r2"].idxmax()
    best = static_rows.loc[idx].copy()
    best["target_friendly"] = best["target"].map(lambda x: FRIENDLY_TARGETS.get(x, x))
    best["graph_variant_friendly"] = best["graph_variant"].map(GRAPH_VARIANTS)
    return best.sort_values("target_friendly")


def build_static_result_table(comparison: pd.DataFrame) -> pd.DataFrame:
    static_rows = comparison[comparison["model_family"] == "static"].copy()
    static_rows["Target"] = static_rows["target"].map(lambda x: FRIENDLY_TARGETS.get(x, x))
    static_rows["Graph variant"] = static_rows["graph_variant"].map(GRAPH_VARIANTS)
    static_rows["GCN test R²"] = static_rows["gcn_test_r2"]
    static_rows["GCN test RMSE"] = static_rows["gcn_test_rmse"]
    static_rows["Ridge test R²"] = static_rows["ridge_test_r2"]
    return static_rows[["Graph variant", "Target", "GCN test R²", "GCN test RMSE", "Ridge test R²"]]


def build_temporal_result_table(treatment_ablation: pd.DataFrame) -> pd.DataFrame:
    table = treatment_ablation.copy()
    friendly_variant = {
        "none": "No project/treatment features",
        "experiment": "EXPERIMENT_SECTION treatment features",
    }
    table["Variant"] = table["treatment_mode"].map(friendly_variant)
    table["RF local R²"] = table["rf_test_r2"]
    table["GCN with project/treatment features R²"] = table["gcn_with_project_treatment_test_r2"]
    return table[
        [
            "Variant",
            "RF local R²",
            "GCN with project/treatment features R²",
        ]
    ]


def available_yearly_series(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "temp_year_temp_avg",
        "temp_year_temp_mean_avg",
        "humid_rel_hum_avg_avg",
        "precip_precipitation",
        "precip_evaporation",
        "precip_precip_days",
        "wind_wind_velocity_avg",
        "solar_cloud_cover_avg",
        "solar_shortwave_surface_avg",
        "ANNUAL_ESAL_TREND",
        "ANNUAL_GESAL_TREND",
        "AADTT_ALL_TRUCKS_TREND",
        "ANNUAL_TRUCK_VOLUME_TREND",
        "CMLTV_VOL_VEH_CLASS_9_TREND",
    ]
    return [col for col in candidates if col in frame.columns]


def normalize_yearly_series(frame: pd.DataFrame, value_cols: list[str], mode: str) -> pd.DataFrame:
    chart_df = frame[["YEAR", *value_cols]].dropna(subset=["YEAR"]).copy()
    chart_df["YEAR"] = chart_df["YEAR"].astype(int)
    out = chart_df.copy()
    for col in value_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().sum() < 2:
            out[col] = np.nan
            continue
        if mode == "Z-score":
            mu = values.mean()
            sd = values.std(ddof=0)
            out[col] = (values - mu) / (sd if sd and sd > 0 else 1.0)
        elif mode == "Min-Max":
            lo, hi = values.min(), values.max()
            out[col] = (values - lo) / ((hi - lo) if hi != lo else 1.0)
        else:
            med = values.median()
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            out[col] = (values - med) / (iqr if iqr and iqr > 0 else 1.0)
    long = out.melt(id_vars="YEAR", value_vars=value_cols, var_name="variable", value_name="normalized_value").dropna()
    long["Series"] = long["variable"].map(lambda col: YEARLY_SERIES_LABELS.get(col, col))
    return long


def render_yearly_inspector(selected_node: str) -> None:
    yearly_panel = load_inspector_yearly_panel()
    point = yearly_panel[yearly_panel["node_id"] == selected_node].copy()
    point = point.sort_values("YEAR")
    series_cols = available_yearly_series(point)
    if point.empty or not series_cols:
        st.info("No yearly climate or traffic values are available for this road section.")
        return

    norm_mode = st.selectbox("Normalization", ["Z-score", "Min-Max", "Robust"], index=0, key=f"norm_{selected_node}")
    st.info(
        """
**How to read the normalization options**

- **Z-score**: centres each variable around its own average and scales it by its standard deviation. This is useful for comparing whether a year is above or below the section's typical level.
- **Min-Max**: rescales each variable between `0` and `1`. This is useful when you want to compare relative peaks and troughs.
- **Robust**: centres each variable on its median and scales it by its interquartile range. This is more resistant to outliers than Z-score.

In all three cases, the values are normalised **within the selected road section only**, so the goal is comparison across variables over time, not comparison of absolute units.
"""
    )
    normalized = normalize_yearly_series(point, series_cols, norm_mode)
    if normalized.empty:
        st.info("Not enough yearly data to normalize any climate or traffic series for this road section.")
    else:
        fig = px.line(
            normalized,
            x="YEAR",
            y="normalized_value",
            color="Series",
            markers=True,
            title=f"Yearly climate and traffic profile for {selected_node}",
            labels={"YEAR": "Year", "normalized_value": f"Normalized value ({norm_mode})"},
        )
        st.plotly_chart(fig, width="stretch")
    with st.expander("Show raw yearly values"):
        raw = point[["YEAR", *series_cols]].copy().rename(columns={col: YEARLY_SERIES_LABELS.get(col, col) for col in series_cols})
        st.dataframe(raw, use_container_width=True, hide_index=True)


def render_distress_inspector(selected_node: str) -> None:
    distress = load_inspector_distress()
    point = distress[distress["node_id"] == selected_node].copy().sort_values("SURVEY_DATE")
    if point.empty:
        st.info("No distress survey history was found for this road section.")
        return

    st.caption(
        f"{len(point)} survey rows from {point['SURVEY_DATE'].min().date()} to {point['SURVEY_DATE'].max().date()} "
        f"across {', '.join(sorted(point['distress_source'].dropna().unique()))}."
    )
    metric_cols = [col for col in DISTRESS_LABELS if col in point.columns and pd.to_numeric(point[col], errors="coerce").notna().any()]
    if not metric_cols:
        st.info("Distress surveys exist, but none of the headline metrics are populated for this section.")
        return

    selected_metrics = st.multiselect(
        "Distress metrics",
        options=metric_cols,
        default=metric_cols[: min(4, len(metric_cols))],
        format_func=lambda col: DISTRESS_LABELS.get(col, col),
        key=f"distress_metrics_{selected_node}",
    )
    if selected_metrics:
        plot_df = point[["SURVEY_DATE", "distress_source", *selected_metrics]].melt(
            id_vars=["SURVEY_DATE", "distress_source"],
            value_vars=selected_metrics,
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])
        plot_df["Metric"] = plot_df["metric"].map(lambda col: DISTRESS_LABELS.get(col, col))
        fig = px.line(
            plot_df,
            x="SURVEY_DATE",
            y="value",
            color="Metric",
            markers=True,
            hover_data={"distress_source": True},
            title=f"Distress survey timeline for {selected_node}",
            labels={"SURVEY_DATE": "Survey date", "value": "Measured value"},
        )
        st.plotly_chart(fig, width="stretch")
    with st.expander("Show raw distress survey rows"):
        cols = ["SURVEY_DATE", "distress_source", "CONSTRUCTION_NO", *metric_cols]
        st.dataframe(point[cols], use_container_width=True, hide_index=True)


def render_treatment_inspector(selected_node: str) -> None:
    events = load_inspector_treatment_events()
    point = events[events["node_id"] == selected_node].copy().sort_values("event_start_date")
    if point.empty:
        st.info("No EXPERIMENT_SECTION treatment or change events were found for this road section.")
        return

    plot_df = point.copy()
    plot_df["plot_end_date"] = plot_df["event_end_date"].fillna(plot_df["event_start_date"] + pd.Timedelta(days=30))
    plot_df["Treatment group"] = plot_df["broad_treatment_group"].map(lambda value: TREATMENT_LABELS.get(value, value))
    plot_df["Treatment label"] = plot_df["treatment_label"].fillna("Missing / not recorded")
    plot_df["Treatment label wrapped"] = plot_df["Treatment label"].map(lambda value: wrap_multiline_label(value, width=26))
    fig = px.timeline(
        plot_df,
        x_start="event_start_date",
        x_end="plot_end_date",
        y="Treatment label wrapped",
        color="Treatment group",
        title=f"Treatment / change history for {selected_node}",
        hover_data={
            "event_year": True,
            "experiment_label": True,
            "status_label": True,
            "route_key": True,
            "functional_class": True,
            "plot_end_date": False,
        },
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(520, 90 + 42 * plot_df["Treatment label wrapped"].nunique()))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Show raw treatment / change events"):
        cols = [
            "event_start_date",
            "event_end_date",
            "event_year",
            "treatment_label",
            "broad_treatment_group",
            "experiment_label",
            "status_label",
            "construction_no",
        ]
        st.dataframe(point[cols], use_container_width=True, hide_index=True)


def page_intro_box(text: str) -> None:
    st.markdown(text)


def file_modified_label(path: Path) -> str:
    if not path.exists():
        return "missing"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def build_data_freshness_table() -> pd.DataFrame:
    rows = []
    for rel in [
        ROOT / "graph_data" / "nodes.csv",
        ROOT / "graph_data" / "edges.csv",
        REPORT_DIR / "treatment_feature_ablation.csv",
        REPORT_DIR / "part1_rgcn_temporal.csv",
        REPORT_DIR / "distress_target_profile.csv",
        REPORT_DIR / "distress_model_comparison.csv",
        REPORT_DIR / "part1_ood_temporal.csv",
        REPORT_DIR / "same_route_real_axis_summary.csv",
        REPORT_DIR / "cracking_correlation_by_edge_type.csv",
    ]:
        rows.append({"File": rel.name, "Last modified": file_modified_label(rel)})
    return pd.DataFrame(rows)


def show_core_terms_box() -> None:
    st.info(CORE_TERMS_MARKDOWN)


nodes = add_project_counts(load_nodes(), load_projects())
projects = load_projects()
core_edges_all = load_core_edges()
graph_diag = load_report_csv("graph_diagnostics.csv")
graph_distance = load_report_csv("graph_distance_summary.csv")
graph_variant_comparison = load_report_csv("graph_variant_model_comparison.csv")
treatment_ablation = load_report_csv("treatment_feature_ablation.csv")
treatment_counts = load_report_csv("experiment_treatment_group_counts.csv")
event_by_group = load_report_csv("experiment_event_study_by_group.csv")
event_classifier = load_report_csv("experiment_event_study_treatment_classifier.csv")
event_interpretation = load_report_json("experiment_event_study_interpretation.json")
case_studies = load_report_csv("dissertation_case_study_summaries.csv")
part1_rgcn = load_report_csv("part1_rgcn_temporal.csv")
osm_validation_findings = load_optional_graph_json("osm_validation_findings.json")
osm_comparison_summary = load_report_csv("osm_comparison_summary.csv")
osm_topology_status = load_report_csv("osm_topology_status_summary.csv")
osm_meta = load_report_json("osm_comparison_summary.json")
same_route_real_axis_summary = load_report_csv("same_route_real_axis_summary.csv")
same_route_edge_suspects_summary = load_report_csv("same_route_edge_suspects_summary.csv")
cracking_corr_by_edge = load_optional_report_csv("cracking_correlation_by_edge_type.csv")
cracking_corr_spatial_bins = load_optional_report_csv("cracking_correlation_spatial_bins.csv")
temporal_feature_audit = load_optional_report_csv("temporal_feature_audit.csv")
temporal_feature_audit_summary = load_optional_report_json("temporal_feature_audit_summary.json")
materials_weight_sweep = load_optional_report_json("materials_weight_sweep.json")
materials_weight_sweep_hpms16 = load_optional_report_json("materials_weight_sweep_hpms16.json")
part1_ood_ensemble_summary = load_optional_report_json("part1_ood_ensemble_summary.json")
ensemble_results = load_optional_graph_json("ensemble_results.json")
mepdg_benchmark = load_optional_report_json("mepdg_benchmark.json")

sidebar = st.sidebar
sidebar.title("Explore the project")
advanced_mode = sidebar.toggle(
    "Advanced mode",
    value=False,
    help="Show extra controls and a light technical appendix.",
)
graph_variant = sidebar.selectbox(
    "Graph variant",
    options=list(GRAPH_VARIANTS.keys()),
    index=1,
    format_func=lambda key: GRAPH_VARIANTS[key],
)

state_options = sorted(nodes["state_code"].astype(str).unique(), key=lambda x: int(x))
state_labels = {code: state_code_to_label(code) for code in state_options}
selected_state_labels = sidebar.multiselect(
    "State filter",
    options=[state_labels[code] for code in state_options],
    default=[],
    help="Leave empty to keep the full national view.",
)
selected_states = [code for code, label in state_labels.items() if label in selected_state_labels]

variant_bundle = load_variant_bundle(graph_variant)
network_edges = variant_bundle["network_edges"].copy()
network_edges["source"] = network_edges["source"].astype(str)
network_edges["target"] = network_edges["target"].astype(str)
variant_predictions = variant_bundle["predictions"].copy()

visible_nodes = filter_by_states(nodes, selected_states)
visible_node_ids = set(visible_nodes["node_id"].astype(str))
visible_edges = filter_edges_for_visible_nodes(network_edges, visible_node_ids)
selected_view_nodes = visible_nodes if not visible_nodes.empty else nodes
selected_view_edges = visible_edges if not visible_edges.empty else network_edges
selected_core_edges = filter_core_edges_for_variant(core_edges_all, graph_variant)
selected_core_edges = filter_edges_for_visible_nodes(selected_core_edges, visible_node_ids)
if selected_core_edges.empty and not selected_view_nodes.empty and not core_edges_all.empty:
    fallback_core = filter_core_edges_for_variant(core_edges_all, graph_variant)
    selected_core_edges = fallback_core[fallback_core["source"].isin(selected_view_nodes["node_id"]) & fallback_core["target"].isin(selected_view_nodes["node_id"])].copy()

section_lookup = build_section_lookup(selected_view_nodes)
section_options = sorted(section_lookup.keys())
default_section = (
    case_studies.iloc[0]["node_id"]
    if not case_studies.empty and case_studies.iloc[0]["node_id"] in section_lookup
    else (section_options[0] if section_options else None)
)
if section_options:
    pending_focus = st.session_state.get("pending_focus_node")
    if pending_focus in section_options:
        st.session_state["focus_node_widget"] = pending_focus
        del st.session_state["pending_focus_node"]
    if "focus_node_widget" not in st.session_state or st.session_state.get("focus_node_widget") not in section_options:
        st.session_state["focus_node_widget"] = default_section
else:
    st.session_state["focus_node_widget"] = None

focus_node = None
if section_options:
    focus_node = sidebar.selectbox(
        "Road section",
        options=section_options,
        index=section_options.index(st.session_state["focus_node_widget"]) if st.session_state.get("focus_node_widget") in section_options else 0,
        format_func=lambda node_id: section_lookup.get(node_id, node_id),
        key="focus_node_widget",
    )

max_edges = None
if advanced_mode:
    limit_edges = sidebar.toggle(
        "Limit edges in wide network view",
        value=False,
        help="Useful only if the national spider-web view becomes heavy to render.",
    )
    if limit_edges:
        max_edges = sidebar.slider("Maximum rendered edges", min_value=300, max_value=6000, value=2200, step=100)

st.title(APP_TITLE)
st.caption("A streamlined dissertation app showing how asset-level pavement records were turned into network-aware maintenance evidence.")

tab_labels = ["Road Section Inspector", "From Asset to Network", "Models and Findings"]
if advanced_mode:
    tab_labels.append("Technical Appendix")
tabs = st.tabs(tab_labels)

with tabs[0]:
    page_intro_box(
        """
This inspector connects the network back to one real monitored road section.

Start here if you want to see **what one node actually is**: where it is, how traffic and climate evolved, what distress surveys recorded, and what treatment or change history was logged for it.
"""
    )
    if not focus_node:
        st.info("No road section is currently available under the selected state filter.")
    else:
        focus_info = nodes[nodes["node_id"] == focus_node].iloc[0]
        inspector_nodes = selected_view_nodes
        inspector_map = draw_node_map(
            inspector_nodes,
            "Click a point to inspect a road section",
            color_col=None,
            highlight_nodes=[focus_node],
            height=720,
        )
        selection = st.plotly_chart(
            inspector_map,
            width="stretch",
            key="inspector_node_map",
            on_select="rerun",
            selection_mode=("points",),
        )
        selected_points = []
        if isinstance(selection, dict):
            selected_points = selection.get("selection", {}).get("points", []) or selection.get("points", [])
        for point in selected_points:
            custom = point.get("customdata") if isinstance(point, dict) else None
            if custom:
                clicked_node = str(custom[0])
                if clicked_node in section_options and clicked_node != st.session_state.get("focus_node_widget"):
                    st.session_state["pending_focus_node"] = clicked_node
                    st.rerun()

        st.caption("The national map is the main entry point for inspection: click a point to make it the selected road section.")
        metric_card_row(
            [
                ("Road section", str(focus_info["node_id"]), "Selected LTPP section identifier."),
                ("State", str(focus_info["state_name"]), "State where the section is located."),
                ("Route", str(focus_info.get("route_key", "n/a")), "Route label used for corridor links in the graph."),
                ("Functional class", str(focus_info.get("functional_class", "n/a")), "Broad road-role category."),
                ("MERRA grid", str(focus_info.get("merra_id", "n/a")), "Climate grid linked to this section."),
                ("Project records", str(int(focus_info.get("project_count", 0))), "Number of linked treatment or project records."),
            ]
        )
        st.info(
            """
**How to read this page**

- **AADT** means *Annual Average Daily Traffic*: the average number of vehicles per day over a given year.
- **AADTT** means *Annual Average Daily Truck Traffic*: the truck-only version of the same idea.
- The traffic series are annual records, even though they are expressed as average daily flow.
- The treatment timeline comes from **EXPERIMENT_SECTION**, so it should be read as broad project/treatment history, not as cracking-only maintenance.
"""
        )
        tabs_inspector = st.tabs(["Yearly climate and traffic", "Distress timeline", "Treatment / change history", "Local neighbourhood"])
        with tabs_inspector[0]:
            render_yearly_inspector(focus_node)
        with tabs_inspector[1]:
            render_distress_inspector(focus_node)
        with tabs_inspector[2]:
            render_treatment_inspector(focus_node)
        with tabs_inspector[3]:
            local_fig = draw_local_network(selected_view_nodes, selected_view_edges, focus_node)
            st.plotly_chart(local_fig, width="stretch")
            st.caption("This small neighbourhood view shows only the selected section and its direct graph neighbours in the currently chosen graph variant.")
        with st.expander("Show curated section metadata"):
            meta_rows = [
                ("Node ID", focus_info.get("node_id")),
                ("State", focus_info.get("state_name")),
                ("Route key", focus_info.get("route_key")),
                ("Functional class", focus_info.get("functional_class")),
                ("Latitude", focus_info.get("latitude")),
                ("Longitude", focus_info.get("longitude")),
                ("Number of lanes", focus_info.get("NO_OF_LANES")),
                ("Section length", focus_info.get("SECTION_LENGTH")),
                ("Speed limit", focus_info.get("SPEED_LIMIT")),
                ("MERRA climate grid", focus_info.get("merra_id")),
            ]
            meta_df = pd.DataFrame(meta_rows, columns=["Field", "Value"])
            st.dataframe(meta_df, use_container_width=True, hide_index=True)
        with st.expander("What the temporal model actually uses"):
            st.markdown(
                """
The temporal model does **not** ingest the full raw `nodes.csv` row.

Instead, for a pair **(section, year)** it builds a compact feature vector containing:
- current cracking,
- yearly traffic,
- yearly climate,
- a few static geometry variables,
- and, in the treatment-aware version, section and neighbour treatment/project features.
"""
            )
            if not temporal_feature_audit.empty:
                retained = temporal_feature_audit[temporal_feature_audit["retained"] == 1].copy()
                retained_view = retained[["feature_name", "feature_family", "train_non_missing_share", "decision_reason"]].copy()
                retained_view["train_non_missing_share"] = retained_view["train_non_missing_share"].map(lambda v: f"{100.0 * float(v):.1f}%")
                st.dataframe(retained_view, use_container_width=True, hide_index=True)
            else:
                st.info("The feature-audit file is not available yet. Re-run the temporal pipeline to populate it.")

with tabs[1]:
    st.markdown("## From asset-level records to a network view")
    st.markdown(
        """
We started with **LTPP data**: a long-running US pavement monitoring programme that follows real road sections over time.

In the raw data, each section is mostly treated as an **asset-level record**. The main idea of this project is to ask whether that is enough. If a section is closed for treatment, traffic, disruption, and scheduling pressure can spill over to nearby sections or to the same corridor. That is why the project moves **from asset to network**.
"""
    )
    st.info(
        """
**Simple definitions**

- A **node** is one LTPP road section.
- An **edge** is one plausible relationship between two sections.
- A **graph** is the collection of those nodes and edges.
- In this project, the graph is not a perfect operational road map of the whole US. It is a **section-level interdependency graph** designed to test whether maintenance relationships matter.
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Why asset-level modelling is incomplete")
        st.markdown(
            """
An asset-level model treats each section independently. That is useful for local condition prediction, but it misses questions like:

- could closing section **A** affect traffic or disruption on **B**?
- are **A** and **B** on the same corridor, so scheduling them together is risky?
- do neighbouring treatments carry information about what will happen next on section **A**?
"""
        )
    with c2:
        st.markdown("### What the network adds")
        st.markdown(
            """
The graph adds three relationship layers:

- **Spatial**: sections that are geographically close.
- **Same route / corridor**: sections on the same named route corridor.
- **Same functional class**: sections with similar road role in the same state.

These edges are hypotheses about interdependence. The rest of the analysis tests whether those hypotheses are useful and plausible.
"""
        )

    current_diag = graph_diag[graph_diag["graph_variant"] == graph_variant].iloc[0]
    total_projects = int(projects.shape[0])
    metric_card_row(
        [
            ("Road sections", compact_number(len(nodes)), "All LTPP sections that became graph nodes."),
            ("Project / treatment rows", compact_number(total_projects), "Prepared treatment or project records linked to nodes."),
            ("Current visible edges", compact_number(len(selected_view_edges)), "Edges visible after the current graph-variant and state filter."),
            ("Largest component share", f"{safe_float(current_diag['largest_component_share_pct']):.1f}%", "How much of the current graph sits in the biggest connected component."),
        ]
    )

    st.markdown("### The three graph variants")
    parameter_rows = [
        {
            "Graph variant": "Spatial only",
            "Edge families used": "spatial",
            "k / sparsity rule": "Each section proposes up to 8 nearest spatial neighbours",
            "Distance rule": "Spatial candidates allowed up to 80 km",
            "Extra rule": "Undirected graph after symmetrising neighbour choices",
        },
        {
            "Graph variant": "Spatial + Route",
            "Edge families used": "spatial + same_route",
            "k / sparsity rule": "Spatial layer as above; route links connect adjacent sections along the same route ordering",
            "Distance rule": "Spatial <= 80 km; same-route links <= 100 km",
            "Extra rule": "Adds corridor continuity on top of local spatial links",
        },
        {
            "Graph variant": "Full refined",
            "Edge families used": "spatial + same_route + same_functional_class",
            "k / sparsity rule": "Same-class layer keeps the top 5 scored candidates per source section",
            "Distance rule": "Spatial <= 80 km; same-route <= 100 km; same-class candidates <= 80 km",
            "Extra rule": "Same-class score = 0.40 spatial + 0.15 traffic + 0.20 climate + 0.25 pavement similarity",
        },
    ]
    st.dataframe(pd.DataFrame(parameter_rows), use_container_width=True, hide_index=True)
    st.caption("These are the graph-construction parameters from graph_construction.py, not display settings from the app.")

    variant_table = graph_diag.copy()
    variant_table["Graph variant"] = variant_table["graph_variant"].map(GRAPH_VARIANTS)
    variant_table = variant_table.rename(
        columns={
            "nodes": "Nodes",
            "edges": "Edges",
            "average_degree": "Average degree",
            "isolated_nodes": "Isolated nodes",
            "connected_components": "Connected components",
            "largest_component_share_pct": "Largest component share (%)",
        }
    )
    st.dataframe(
        variant_table[["Graph variant", "Nodes", "Edges", "Average degree", "Isolated nodes", "Connected components", "Largest component share (%)"]],
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        """
**What these names mean**

- **Spatial only**: local geographic proximity only.
- **Spatial + Route**: spatial links plus route-continuity corridor links.
- **Full refined**: spatial + corridor + a sparse filtered similarity layer.

A richer graph is not automatically a better graph. The point is to test which representation is most useful for each modelling task.
"""
    )

    st.markdown("### How the full refined similarity score is built")
    st.markdown(
        """
For the extra **same functional class** edges in the **full refined** graph, the code does **not** use a hard threshold like `score > 0.60`.

Instead, it works in two steps:

1. **Candidate filter**
- same **state**
- same **functional class**
- within **80 km** great-circle distance

2. **Top-k acceptance rule**
- compute a similarity score for each candidate pair
- keep the **top 5 highest-scoring neighbours** for each source section

So the acceptance rule is:

`candidate is accepted if it is among the 5 best-scoring same-class candidates within 80 km`
"""
    )
    st.code(
        """spatial_score = exp(- distance_km / 80)
score = 0.40 * spatial_score \
      + 0.15 * traffic_similarity \
      + 0.20 * climate_similarity \
      + 0.25 * pavement_similarity""",
        language="text",
    )
    st.info(
        """
**How to read the formula**

- **Spatial score** gets larger when two sections are closer together.
- **Traffic similarity** compares their traffic profiles.
- **Climate similarity** compares their climate exposure.
- **Pavement similarity** compares their structural road attributes such as lanes, width, and section length.
- The weights sum to **1.00**, so the score is a weighted average.

This means the `full refined` graph is still mostly local, because distance carries the biggest single weight and candidates beyond 80 km are excluded before scoring.
"""
    )

    edge_summary = graph_diag[["graph_variant", "edges"]].copy()
    edge_summary["Graph variant"] = edge_summary["graph_variant"].map(GRAPH_VARIANTS)
    fig_edges = px.bar(edge_summary, x="Graph variant", y="edges", title="How many relationships does each graph variant contain?", labels={"edges": "Edges"})
    st.plotly_chart(fig_edges, width="stretch")

    st.markdown("### National graph view")
    corridor_distance_default = 20
    show_long_corridors = st.toggle(
        "Show long-range corridor membership links",
        value=False,
        help="Turn this on only if you want to see every same-route link, including very long corridor links that are not local neighbours.",
        key="show_long_corridors_main",
    )
    corridor_distance_km = corridor_distance_default
    if advanced_mode:
        corridor_distance_km = st.slider(
            "If long corridor links are hidden, keep corridor edges up to this distance (km)",
            min_value=5,
            max_value=80,
            value=20,
            step=5,
        )

    web_edges = selected_view_edges.copy()
    hidden_corridor_count = 0
    if not show_long_corridors and not web_edges.empty and {"edge_role", "distance_km"}.issubset(web_edges.columns):
        web_edges["distance_km"] = pd.to_numeric(web_edges["distance_km"], errors="coerce")
        hide_mask = web_edges["edge_role"].eq("corridor") & web_edges["distance_km"].gt(corridor_distance_km)
        hidden_corridor_count = int(hide_mask.sum())
        web_edges = web_edges[~hide_mask].copy()

    visible_graph = nx.Graph()
    visible_graph.add_nodes_from(selected_view_nodes["node_id"].astype(str).tolist())
    if not web_edges.empty:
        visible_graph.add_edges_from(web_edges[["source", "target"]].astype(str).itertuples(index=False, name=None))
    visible_isolated = sum(1 for _, degree in visible_graph.degree() if degree == 0)
    role_counts = web_edges["edge_role"].value_counts().to_dict() if "edge_role" in web_edges.columns else {}
    spatial_count = int(role_counts.get("spatial", 0))
    corridor_count = int(role_counts.get("corridor", 0))
    diversion_count = int(role_counts.get("diversion", 0))

    metric_card_row(
        [
            ("Rendered edges", str(len(web_edges) if max_edges is None else min(len(web_edges), max_edges)), "Edges actually drawn in the spider-web view."),
            ("Spatial links", str(spatial_count), "Nearby-section links currently visible."),
            ("Corridor links", str(corridor_count), "Same-route links currently visible."),
            ("Hidden long corridor links", str(hidden_corridor_count), "Long same-route links hidden from the main view because they are corridor membership rather than local neighbourhood."),
            ("Isolated visible sections", str(visible_isolated), "Visible sections with no remaining edge in the current view."),
        ]
    )
    st.info(
        f"""
**How to read the spider-web view**

- **Blue** = local spatial links.
- **Red** = corridor links along the same route.
- **Green** = diversion-style links, if present in the selected graph variant.
- **Grey dots** = road sections.
- **Gold star** = the currently selected section.

The default view hides corridor links longer than **{corridor_distance_km} km** because the OSM audit showed that many long same-route links are better interpreted as **corridor membership**, not as **local neighbours**.
"""
    )
    web_fig = draw_network_web(
        selected_view_nodes,
        web_edges,
        title="Spider-web view of the selected graph",
        focus_node=focus_node,
        max_edges=max_edges,
    )
    st.plotly_chart(web_fig, width="stretch")

    min_cluster_size = st.slider(
        "Minimum cluster size to display",
        min_value=1,
        max_value=50,
        value=1,
        step=1,
        help="Hide clusters smaller than this number of nodes so the map can focus on larger connected components.",
        key="min_cluster_size_cluster_map",
    )
    cluster_fig = draw_cluster_map(
        selected_view_nodes,
        selected_core_edges,
        title="Connected local clusters in the current filtered graph",
        min_cluster_size=min_cluster_size,
    )
    st.plotly_chart(cluster_fig, width="stretch")
    st.caption("The graph should usually be read as many local interdependency clusters rather than one seamless national road network.")

    st.markdown("### Does the graph match a real road map?")
    supported_edges = int(osm_topology_status.loc[osm_topology_status["topology_level"] == "supported", "edges"].sum())
    weak_edges = int(osm_topology_status.loc[osm_topology_status["topology_level"] == "weakly_connected", "edges"].sum())
    disconnected_edges = int(osm_topology_status.loc[osm_topology_status["topology_level"] == "not_connected", "edges"].sum())
    total_tested_edges = int(osm_topology_status["edges"].sum())
    plausible_pct = (100.0 * supported_edges / total_tested_edges) if total_tested_edges else float("nan")

    same_route_long_total = int(same_route_real_axis_summary["edges"].sum())
    same_route_supported_total = int(same_route_real_axis_summary.loc[same_route_real_axis_summary["real_axis_verdict"] == "same_real_axis_supported", "edges"].sum())
    same_route_not_supported = int(same_route_real_axis_summary.loc[same_route_real_axis_summary["real_axis_verdict"] == "not_supported", "edges"].sum())
    local_same_route_share = float(
        same_route_edge_suspects_summary.loc[
            same_route_edge_suspects_summary["suspicious_locality"].isin(["very_local", "local"]),
            "share_pct",
        ].sum()
    )

    metric_card_row(
        [
            ("Locally audited edges", str(total_tested_edges), "Edges checked against compact OSM road subgraphs."),
            ("Plausible local edges", f"{plausible_pct:.1f}%", "Edges on the same OSM segment or connected by a short / reasonable road path."),
            ("Long same-route edges audited", str(same_route_long_total), "Targeted audit of corridor links longer than 30 km."),
            ("Long same-route edges still on the same real axis", str(same_route_supported_total), "Long corridor links that still look like the same real road corridor on OSM."),
            ("Long same-route edges not supported", str(same_route_not_supported), "Long corridor links that look doubtful on the real map."),
            ("Same-route edges under 10 km", f"{local_same_route_share:.1f}%", "Most corridor links are actually short and local; only a small tail is suspicious."),
        ]
    )
    st.info(
        """
**OSM interpretation**

Two different OSM checks were useful here:

1. **Local edge plausibility audit**: on compact local components, most tested graph edges were topologically plausible on a real road map.
2. **Long same-route corridor audit**: most very long same-route links still look like the same corridor in OSM, but they should be read as **same corridor membership**, not as **local neighbouring sections**.

This matters because a line that crosses a large part of the country may still belong to the same named route, but it does **not** mean a closure in one place creates an immediate local impact everywhere else along that route.
"""
    )

    axis_view = same_route_real_axis_summary.copy()
    axis_view["Verdict"] = axis_view["real_axis_verdict"].replace(
        {
            "same_real_axis_supported": "Same real axis supported",
            "connected_but_far": "Connected but far",
            "not_supported": "Not supported",
            "needs_manual_map_check": "Needs manual map check",
        }
    )
    fig_axis = px.bar(
        axis_view,
        x="Verdict",
        y="edges",
        color="local_neighbour_verdict",
        title="Audit of long same-route links against OpenStreetMap",
        labels={"edges": "Edges", "local_neighbour_verdict": "Neighbour interpretation"},
    )
    st.plotly_chart(fig_axis, width="stretch")

    st.markdown("### Do connected sections also evolve similarly?")
    st.info(
        """
This is a **different validation question** from OSM.

- **OSM validation** asks: *are these edges physically/topologically plausible on a real road map?*
- **Cracking-correlation validation** asks: *do linked sections empirically move together over time in the distress data?*

An edge can be topologically plausible but only weakly correlated in cracking evolution, because pavement deterioration also depends on materials, maintenance timing, trucks, and local conditions. So these two checks are **complementary**, not contradictory.
"""
    )
    if not cracking_corr_by_edge.empty:
        corr_view = cracking_corr_by_edge.copy()
        corr_view["Edge type"] = corr_view["edge_type"].replace(
            {
                "spatial": "Spatial",
                "same_route": "Same route",
                "same_functional_class": "Same functional class",
            }
        )
        st.dataframe(
            corr_view[[
                "Edge type",
                "tested_edges",
                "edges_with_change_corr",
                "median_change_corr",
                "mean_change_corr",
                "median_level_corr",
                "mean_level_corr",
                "median_distance_km",
            ]],
            use_container_width=True,
            hide_index=True,
        )
        fig_corr = px.bar(
            corr_view,
            x="Edge type",
            y=["median_change_corr", "median_level_corr"],
            barmode="group",
            title="Do linked sections have correlated cracking levels and cracking changes?",
            labels={"value": "Median correlation", "variable": "Correlation type"},
        )
        st.plotly_chart(fig_corr, width="stretch")
    if not cracking_corr_spatial_bins.empty:
        bin_view = cracking_corr_spatial_bins.copy().rename(columns={"distance_bin": "Spatial distance bin"})
        fig_spatial_corr = px.bar(
            bin_view,
            x="Spatial distance bin",
            y="median_change_corr",
            text="edges",
            title="For spatial links, cracking-change similarity is strongest at short distance",
            labels={"median_change_corr": "Median cracking-change correlation", "edges": "Edges"},
        )
        st.plotly_chart(fig_spatial_corr, width="stretch")
    st.success(
        """
**How to reconcile OSM and cracking correlation**

- OSM says whether an edge is **physically plausible** as a route or local connection.
- Cracking correlation says whether linked sections **behave similarly** in the distress data.
- The current evidence suggests that many graph edges are topologically plausible, while the **strongest empirical co-movement appears for short-distance links**.

So the graph is not invalidated by the correlation results. Instead, the correlation results help us say **which kinds of edges are more behaviourally convincing**, especially for local propagation arguments.
"""
    )

with tabs[2]:
    st.markdown("## Models and findings")
    st.markdown(
        """
This page focuses on the **latest cracking-prediction results** and removes older exploratory branches that are no longer part of the main study story.

The key question is:

> **Which model family predicts cracking best, and when does the graph genuinely help?**
"""
    )

    hpms_rows: list[dict[str, object]] = []
    hpms_results = ensemble_results.get("results", {}) if isinstance(ensemble_results, dict) else {}
    if hpms_results:
        rf_test = hpms_results.get("RF local", {}).get("test", {})
        rgcn_test = hpms_results.get("R-GCN", {}).get("test", {})
        ens_test = hpms_results.get("Stacked MLP", {}).get("test", {})
        hpms_rows.extend(
            [
                {
                    "Target": "HPMS16 cracking",
                    "Model": "RF local",
                    "Family": "Asset-level baseline",
                    "Graph": "No",
                    "Test R²": safe_float(rf_test.get("r2")),
                    "Test MAE": safe_float(rf_test.get("mae")),
                    "Test RMSE": safe_float(rf_test.get("rmse")),
                },
                {
                    "Target": "HPMS16 cracking",
                    "Model": "R-GCN baseline",
                    "Family": "Graph-aware",
                    "Graph": "full_refined",
                    "Test R²": safe_float(rgcn_test.get("r2")),
                    "Test MAE": safe_float(rgcn_test.get("mae")),
                    "Test RMSE": safe_float(rgcn_test.get("rmse")),
                },
                {
                    "Target": "HPMS16 cracking",
                    "Model": "Stacked MLP ensemble",
                    "Family": "Hybrid ensemble",
                    "Graph": "RF + R-GCN",
                    "Test R²": safe_float(ens_test.get("r2")),
                    "Test MAE": safe_float(ens_test.get("mae")),
                    "Test RMSE": safe_float(ens_test.get("rmse")),
                },
            ]
        )
    if materials_weight_sweep_hpms16:
        best_hpms_label = materials_weight_sweep_hpms16.get("best_label")
        best_hpms_row = next(
            (row for row in materials_weight_sweep_hpms16.get("results", []) if row.get("label") == best_hpms_label),
            {},
        )
        hpms_rows.append(
            {
                "Target": "HPMS16 cracking",
                "Model": f"Refined R-GCN ({best_hpms_label})",
                "Family": "Graph-aware",
                "Graph": "full_refined + reweighted materials",
                "Test R²": safe_float(materials_weight_sweep_hpms16.get("best_test_r2")),
                "Test MAE": safe_float(best_hpms_row.get("mae_test")),
                "Test RMSE": safe_float(best_hpms_row.get("rmse_test")),
            }
        )
    hpms_benchmark_df = pd.DataFrame(hpms_rows)

    mepdg_benchmark_df = pd.DataFrame(mepdg_benchmark.get("results", [])) if mepdg_benchmark else pd.DataFrame()
    if not mepdg_benchmark_df.empty:
        mepdg_benchmark_df = mepdg_benchmark_df.rename(
            columns={"model": "Model", "r2_test": "Test R²", "mae_test": "Test MAE", "rmse_test": "Test RMSE"}
        )
        mepdg_benchmark_df["Target"] = "MEPDG cracking"
        mepdg_benchmark_df["Family"] = mepdg_benchmark_df["Model"].map(
            {
                "RF local": "Asset-level baseline",
                "R-GCN baseline": "Graph-aware",
                "Stacked MLP ensemble": "Hybrid ensemble",
                "R-GCN best materials sweep (climate_pavement)": "Graph-aware",
            }
        ).fillna("Other")
        mepdg_benchmark_df["Graph"] = mepdg_benchmark_df["Model"].map(
            {
                "RF local": "No",
                "R-GCN baseline": "full_refined",
                "Stacked MLP ensemble": "RF + R-GCN",
                "R-GCN best materials sweep (climate_pavement)": "full_refined + reweighted materials",
            }
        ).fillna("n/a")

    cracking_benchmark = pd.concat(
        [
            hpms_benchmark_df[["Target", "Model", "Family", "Graph", "Test R²", "Test MAE", "Test RMSE"]]
            if not hpms_benchmark_df.empty
            else pd.DataFrame(),
            mepdg_benchmark_df[["Target", "Model", "Family", "Graph", "Test R²", "Test MAE", "Test RMSE"]]
            if not mepdg_benchmark_df.empty
            else pd.DataFrame(),
        ],
        ignore_index=True,
    )

    hpms_best = safe_float(materials_weight_sweep_hpms16.get("best_test_r2")) if materials_weight_sweep_hpms16 else None
    hpms_ensemble = safe_float(hpms_results.get("Stacked MLP", {}).get("test", {}).get("r2")) if hpms_results else None
    mepdg_best = safe_float(materials_weight_sweep.get("best_test_r2")) if materials_weight_sweep else None
    rf_ood = safe_float(part1_ood_ensemble_summary.get("rf_r2", {}).get("mean")) if part1_ood_ensemble_summary else None
    ens_ood = safe_float(part1_ood_ensemble_summary.get("ensemble_r2", {}).get("mean")) if part1_ood_ensemble_summary else None

    metric_card_row(
        [
            ("HPMS16 best overall R²", f"{hpms_ensemble:.3f}" if hpms_ensemble is not None else "n/a", "Latest HPMS16 best overall result: the stacked MLP ensemble."),
            ("HPMS16 best pure graph R²", f"{hpms_best:.3f}" if hpms_best is not None else "n/a", "Latest HPMS16 best graph-only result after refining the graph weights."),
            ("MEPDG best result R²", f"{mepdg_best:.3f}" if mepdg_best is not None else "n/a", "Latest MEPDG best result after graph refinement."),
            ("OOD mean R²: RF vs ensemble", f"{rf_ood:.3f} vs {ens_ood:.3f}" if rf_ood is not None and ens_ood is not None else "n/a", "Held-out-state mean R² on the latest 5-state subset."),
        ]
    )

    st.markdown("### 1. Cracking benchmark: current headline comparison")
    st.markdown(
        """
- **HPMS16 cracking** is the main benchmark across model families.
- **MEPDG cracking** is the specialist target where the refined graph currently performs best.
"""
    )
    if not cracking_benchmark.empty:
        benchmark_plot = px.bar(
            cracking_benchmark,
            x="Model",
            y="Test R²",
            color="Target",
            barmode="group",
            title="Cracking prediction benchmark across the main model families",
        )
        st.plotly_chart(benchmark_plot, width="stretch")
        st.dataframe(
            cracking_benchmark.sort_values(["Target", "Test R²"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        """
**What this benchmark says**

- On **HPMS16**, the best overall score is the **stacked ensemble**.
- On **HPMS16**, the **refined R-GCN** is now almost at the same level as the ensemble.
- On **MEPDG**, the best score is the **refined R-GCN**, not the ensemble.

So the clean presentation story is:
- **HPMS16** = benchmark between model families
- **MEPDG** = best specialist graph result
"""
    )

    st.markdown("### 2. Does the graph itself help?")
    rgcn_view = part1_rgcn.copy()
    rgcn_view["Graph variant"] = rgcn_view["graph_variant"].map(GRAPH_VARIANTS)
    fig_rgcn = px.bar(
        rgcn_view,
        x="Graph variant",
        y="test_r2",
        text="test_r2",
        title="Baseline R-GCN improves as the graph becomes richer",
        labels={"test_r2": "Test R²"},
    )
    fig_rgcn.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig_rgcn, width="stretch")

    if materials_weight_sweep_hpms16.get("results"):
        hpms_sweep_df = pd.DataFrame(materials_weight_sweep_hpms16["results"]).copy()
        hpms_sweep_df["Config"] = hpms_sweep_df["label"]
        fig_hpms_sweep = px.bar(
            hpms_sweep_df,
            x="Config",
            y="r2_test",
            title="HPMS16: reweighting the refined graph changes performance substantially",
            labels={"r2_test": "Test R²"},
        )
        st.plotly_chart(fig_hpms_sweep, width="stretch")

    st.success(
        f"""
**What the graph evidence supports**

- Baseline R-GCN rises from **{safe_float(rgcn_view.loc[rgcn_view['graph_variant'].eq('spatial'), 'test_r2'].iloc[0]):.3f}** on `spatial`
  to **{safe_float(rgcn_view.loc[rgcn_view['graph_variant'].eq('full_refined'), 'test_r2'].iloc[0]):.3f}** on `full_refined`.
- On **HPMS16**, refinement pushes the graph result to **{hpms_best:.3f}**.
- On **MEPDG**, the best refined graph reaches **{mepdg_best:.3f}**.

So the useful conclusion is not “any graph works”. It is that the **refined graph family** is the one that matters.
"""
    )

    st.markdown("### 3. Robustness on unseen states")
    ood_rows = []
    if part1_ood_ensemble_summary:
        ood_rows.extend(
            [
                {"Model": "RF local", "Mean OOD R²": safe_float(part1_ood_ensemble_summary.get("rf_r2", {}).get("mean"))},
                {"Model": "R-GCN", "Mean OOD R²": safe_float(part1_ood_ensemble_summary.get("rgcn_r2", {}).get("mean"))},
                {"Model": "Ensemble", "Mean OOD R²": safe_float(part1_ood_ensemble_summary.get("ensemble_r2", {}).get("mean"))},
            ]
        )
    ood_df = pd.DataFrame(ood_rows)
    if not ood_df.empty:
        fig_ood = px.bar(
            ood_df,
            x="Model",
            y="Mean OOD R²",
            title="Held-out-state robustness remains the weak point",
            labels={"Mean OOD R²": "Mean R² on held-out states"},
        )
        st.plotly_chart(fig_ood, width="stretch")
        st.dataframe(ood_df, use_container_width=True, hide_index=True)

    st.warning(
        """
The current graph story is strongest **in-domain**.

- The ensemble does **not** solve geographic transfer.
- The R-GCN is currently **weaker than RF** on the latest 5-state held-out subset.
- So the honest claim is improved cracking prediction **within the observed data regime**, not full inter-state generalisation.
"""
    )

    st.markdown("### 4. Final takeaways")
    st.info(
        """
**Supported by the current evidence**

- Graph-aware models can improve cracking prediction when the graph is **well refined**.
- The **full refined** graph is a better basis than the simpler graph variants.
- The strongest graph results are now the **refined R-GCN** results on **HPMS16** and **MEPDG**.

**Not supported by the current evidence**

- That graph models always beat tabular baselines.
- That the graph alone is the best HPMS16 headline model.
- That the current graph solves **OOD geographic transfer**.
"""
    )

if advanced_mode and len(tabs) > 3:
    with tabs[3]:
        page_intro_box("This appendix keeps a few raw diagnostics available without interrupting the main story.")
        st.markdown("### Data freshness")
        st.caption("The app now reads live files from disk for graph data, reports, and inspector tables. This table helps confirm which outputs were last updated.")
        st.dataframe(build_data_freshness_table(), use_container_width=True, hide_index=True)
        appendix_files = [
            "graph_diagnostics.csv",
            "graph_variant_model_comparison.csv",
            "treatment_feature_ablation.csv",
            "part1_rgcn_temporal.csv",
            "distress_target_profile.csv",
            "distress_model_comparison.csv",
            "part1_ood_temporal.csv",
            "part1_ood_static.csv",
            "osm_topology_status_summary.csv",
            "same_route_real_axis_summary.csv",
            "same_route_edge_suspects_summary.csv",
        ]
        for name in appendix_files:
            with st.expander(f"Show {name}"):
                st.dataframe(pd.read_csv(REPORT_DIR / name, low_memory=False), use_container_width=True)
