from __future__ import annotations

import json
from pathlib import Path

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

GRAPH_VARIANTS = {
    "spatial": "Spatial only",
    "spatial_route": "Spatial + Route",
    "full_refined": "Full refined",
}

GRAPH_VARIANT_STORY = {
    "spatial": "Nearby sections only. This is the clearest structure for proxy disruption prediction.",
    "spatial_route": "Nearby sections plus local same-route corridor continuity. This is useful for corridor-based interpretation.",
    "full_refined": "Spatial plus route plus filtered similarity in function, traffic, climate, and structure.",
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


@st.cache_data
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


@st.cache_data
def load_projects() -> pd.DataFrame:
    projects = pd.read_csv(DATA_DIR / "projects.csv", low_memory=False)
    projects["node_id"] = projects["node_id"].astype(str)
    return projects


@st.cache_data
def load_variant_bundle(variant: str) -> dict[str, object]:
    bundle: dict[str, object] = {}
    bundle["network_edges"] = pd.read_csv(tagged_graph_path("network_edges_research", variant, ".csv"), low_memory=False)
    bundle["od_pairs"] = pd.read_csv(tagged_graph_path("network_od_pairs", variant, ".csv"), low_memory=False)
    bundle["scenarios"] = pd.read_csv(tagged_graph_path("network_scenarios", variant, ".csv"), low_memory=False)
    bundle["predictions"] = pd.read_csv(tagged_graph_path("network_scenario_predictions", variant, ".csv"), low_memory=False)
    with open(tagged_graph_path("network_model_metrics", variant, ".json"), "r", encoding="utf-8") as fh:
        bundle["static_metrics"] = json.load(fh)
    return bundle


@st.cache_data
def load_report_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / name, low_memory=False)


@st.cache_data
def load_report_json(name: str) -> dict:
    with open(REPORT_DIR / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
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


@st.cache_data
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
        raw["node_id"] = raw.apply(lambda row: f"{str(row['STATE_CODE']).strip()}_{str(row['SHRP_ID']).strip()}", axis=1)
        raw["SURVEY_DATE"] = pd.to_datetime(raw["SURVEY_DATE"], errors="coerce")
        raw["YEAR"] = raw["SURVEY_DATE"].dt.year.astype("Int64")
        raw["distress_source"] = sheet_name.replace("ANALYSIS_DIS_", "")
        for col in metric_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        pieces.append(raw)
    return pd.concat(pieces, ignore_index=True)


@st.cache_data
def load_inspector_treatment_events() -> pd.DataFrame:
    events = pd.read_csv(REPORT_DIR / "experiment_section_event_table.csv", low_memory=False)
    events["node_id"] = events["node_id"].astype(str)
    events["event_start_date"] = pd.to_datetime(events["event_start_date"], errors="coerce")
    events["event_end_date"] = pd.to_datetime(events["event_end_date"], errors="coerce")
    return events


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
    max_edges: int = 1200,
    color_by_cluster: bool = False,
    height: int = 700,
) -> go.Figure:
    frame_nodes = nodes.copy()
    edge_frame = edges.head(max_edges).copy()
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


def draw_cluster_map(nodes: pd.DataFrame, edges: pd.DataFrame, title: str, height: int = 650) -> go.Figure:
    frame = assign_component_clusters(nodes, edges)
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
        "old_proxy": "Old PROJECT_HIST_AGE_EXP proxy",
        "experiment": "EXPERIMENT_SECTION treatment features",
    }
    table["Variant"] = table["treatment_mode"].map(friendly_variant)
    table["RF local R²"] = table["rf_test_r2"]
    table["GCN without project/treatment features R²"] = table["gcn_without_project_treatment_test_r2"]
    table["GCN with project/treatment features R²"] = table["gcn_with_project_treatment_test_r2"]
    table["GCN gain from treatment features"] = table["gcn_project_treatment_r2_gain"]
    return table[
        [
            "Variant",
            "RF local R²",
            "GCN without project/treatment features R²",
            "GCN with project/treatment features R²",
            "GCN gain from treatment features",
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
        st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(fig, use_container_width=True)
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
    fig = px.timeline(
        plot_df,
        x_start="event_start_date",
        x_end="plot_end_date",
        y="Treatment label",
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
    st.plotly_chart(fig, use_container_width=True)
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


def show_core_terms_box() -> None:
    st.info(CORE_TERMS_MARKDOWN)


nodes = add_project_counts(load_nodes(), load_projects())
projects = load_projects()
graph_diag = load_report_csv("graph_diagnostics.csv")
graph_distance = load_report_csv("graph_distance_summary.csv")
graph_variant_comparison = load_report_csv("graph_variant_model_comparison.csv")
climate_meta = load_report_json("climate_mapping_diagnostics.json")
monthly_ablation = load_report_csv("monthly_climate_ablation.csv")
treatment_ablation = load_report_csv("treatment_feature_ablation.csv")
treatment_semantics = load_report_csv("treatment_feature_semantics.csv")
treatment_counts = load_report_csv("experiment_treatment_group_counts.csv")
event_by_group = load_report_csv("experiment_event_study_by_group.csv")
event_climate_by_group = load_report_csv("experiment_event_study_climate_by_group.csv")
event_classifier = load_report_csv("experiment_event_study_treatment_classifier.csv")
event_importance = load_report_csv("experiment_event_study_feature_importance.csv")
event_neighbour_vs_control = load_report_csv("experiment_event_study_neighbour_vs_control.csv")
event_interpretation = load_report_json("experiment_event_study_interpretation.json")
case_studies = load_report_csv("dissertation_case_study_summaries.csv")
static_target_defs = load_report_csv("static_target_variable_definitions.csv")
static_target_defs_json = load_report_json("static_target_variable_definitions.json")
static_target_worked = load_report_csv("static_target_worked_example.csv")
monthly_feature_diag = load_report_csv("monthlyagg_feature_diagnostics.csv")
climate_mapping_diag = load_report_csv("climate_mapping_diagnostics.csv")
monthly_redundancy = load_report_csv("event_study_monthly_vs_annual_climate_redundancy.csv")

sidebar = st.sidebar
sidebar.title("Explore the story")
advanced_mode = sidebar.toggle("Advanced mode", value=False, help="Show a few extra controls and the technical appendix.")
graph_variant = sidebar.selectbox(
    "Graph variant",
    options=list(GRAPH_VARIANTS.keys()),
    index=0,
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
variant_scenarios = variant_bundle["scenarios"].copy()
variant_predictions = variant_bundle["predictions"].copy()

visible_nodes = filter_by_states(nodes, selected_states)
visible_node_ids = set(visible_nodes["node_id"].astype(str))
visible_edges = filter_edges_for_visible_nodes(network_edges, visible_node_ids)
visible_scenarios = filter_scenarios_to_states(variant_scenarios, selected_states, nodes)
if visible_scenarios.empty:
    visible_scenarios = variant_scenarios.copy()

section_lookup = build_section_lookup(visible_nodes if not visible_nodes.empty else nodes)
section_options = sorted(section_lookup.keys())
default_section = case_studies.iloc[0]["node_id"] if not case_studies.empty and case_studies.iloc[0]["node_id"] in section_lookup else (section_options[0] if section_options else None)
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

scenario_options = visible_scenarios.sort_values("delta_vht_proxy", ascending=False)["scenario_id"].tolist()
default_scenario = scenario_options[0] if scenario_options else None
selected_scenario_id = None
if scenario_options:
    selected_scenario_id = sidebar.selectbox("Selected scenario", options=scenario_options, index=0)

case_options = case_studies["node_id"].tolist()
selected_case_node = None
if case_options:
    selected_case_node = sidebar.selectbox(
        "Case-study section",
        options=case_options,
        index=0,
        format_func=lambda node_id: next(
            (
                f"{row.node_id} — {row.state_name} — {row.route_key if pd.notna(row.route_key) else 'no route'} — {TREATMENT_LABELS.get(row.treatment_group, row.treatment_group)}"
                for _, row in case_studies.iterrows()
                if row["node_id"] == node_id
            ),
            node_id,
        ),
    )

max_edges = 1200
if advanced_mode:
    max_edges = sidebar.slider("Maximum edges in web view", min_value=200, max_value=3000, value=1200, step=100)

st.title(APP_TITLE)
st.caption("A guided dissertation app that explains how asset-level road data were turned into graph-based maintenance insights.")

tabs = st.tabs(
    [
        "Road Section Inspector",
        "Project Story / Overview",
        "Data Sources",
        "Graph Construction",
        "Static Disruption Targets",
        "Static Graph Model Results",
        "Temporal Degradation and Treatment Features",
        "EXPERIMENT_SECTION Treatment Semantics",
        "Event-Study Findings",
        "Monthly Climate Findings",
        "What We Have Learned / Final Conclusions",
        "Technical Appendix",
    ]
)

# Page 1
with tabs[0]:
    page_intro_box(
        """
This inspector is the best place to start if you are discovering the project.

It connects the **network view** back to one real road section at a time:
- where the section is located,
- how its climate and traffic evolved over time,
- what distress surveys recorded,
- and what treatment or project history was recorded for it.
"""
    )
    show_core_terms_box()
    if not focus_node:
        st.info("No road section is currently available under the selected state filter.")
    else:
        focus_info = nodes[nodes["node_id"] == focus_node].iloc[0]
        inspector_nodes = visible_nodes if not visible_nodes.empty else nodes
        inspector_map = draw_node_map(
            inspector_nodes,
            "Click a road section to inspect it",
            color_col="project_count",
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

        st.caption("This national map is the main entry point for inspection: click a point to make it the selected road section.")
        metric_card_row(
            [
                ("Road section", str(focus_info["node_id"]), "Selected LTPP road section"),
                ("State", str(focus_info["state_name"]), "State where the section is located"),
                ("Route", str(focus_info.get("route_key", "n/a")), "Route or corridor label used in the graph"),
                ("Functional class", str(focus_info.get("functional_class", "n/a")), "Broad road role, such as interstate-like or arterial-like"),
                ("Climate grid ID", str(focus_info.get("merra_id", "n/a")), "Identifier of the linked MERRA climate cell"),
                ("Project records", str(int(focus_info.get("project_count", 0))), "Count of linked project or treatment records"),
            ]
        )
        tabs_inspector = st.tabs(["Yearly climate & traffic", "Distress timeline", "Treatment / maintenance"])
        with tabs_inspector[0]:
            st.info("These yearly series are normalised within the selected section so climate and traffic variables can be compared on the same chart even though they have different units.")
            render_yearly_inspector(focus_node)
        with tabs_inspector[1]:
            st.info("Distress data come from dated survey rows. In other words, this tab shows when inspectors actually measured cracking and related condition variables.")
            render_distress_inspector(focus_node)
        with tabs_inspector[2]:
            st.info("Treatment history comes from EXPERIMENT_SECTION and should be interpreted as broader treatment or project history, not as cracking-only maintenance.")
            render_treatment_inspector(focus_node)
        with st.expander("Show raw section attributes"):
            st.dataframe(focus_info.to_frame().reset_index().rename(columns={"index": "Field", 0: "Value"}), use_container_width=True, hide_index=True)

# Page 2
with tabs[1]:
    page_intro_box(
        """
This project starts with **individual pavement sections** and builds a **graph**
to understand how maintenance decisions interact across a network.
"""
    )
    show_core_terms_box()
    st.markdown(
        """
**Project flow**

`Data` → `Graph` → `Disruption scenarios` → `Models` → `Treatment / event analysis` → `Optimisation next step`
"""
    )

    best_static = best_static_rows(graph_variant_comparison)
    best_delta = best_static.loc[best_static["target"] == "delta_vht_proxy", "gcn_test_r2"]
    best_disruption = best_static.loc[best_static["target"] == "disruption_score", "gcn_test_r2"]
    exp_row = treatment_ablation[treatment_ablation["treatment_mode"] == "experiment"].iloc[0]
    metric_card_row(
        [
            ("LTPP road sections", compact_number(len(nodes)), "One node = one LTPP pavement section"),
            ("MERRA climate coverage", f"{climate_meta['merra_coverage_pct']:.1f}%", "Corrected after SHRP_ID normalisation"),
            ("Treatment / change events", compact_number(event_interpretation["n_events"]), "Events from EXPERIMENT_SECTION"),
            ("Best extra travel-time proxy R²", f"{safe_float(best_delta.iloc[0]):.3f}" if not best_delta.empty else "n/a", "Best static graph result"),
            ("RF local cracking R²", f"{safe_float(exp_row['rf_test_r2']):.3f}", "Best local one-year cracking predictor"),
            ("GCN with treatment features R²", f"{safe_float(exp_row['gcn_with_project_treatment_test_r2']):.3f}", "Temporal graph model with EXPERIMENT_SECTION features"),
        ]
    )

    edge_summary = graph_diag[["graph_variant", "edges"]].copy()
    edge_summary["Variant"] = edge_summary["graph_variant"].map(GRAPH_VARIANTS)
    edge_summary["Edges"] = edge_summary["edges"]
    fig_edges = px.bar(edge_summary, x="Variant", y="Edges", title="How many section-to-section relationships are in each graph variant?")
    st.plotly_chart(fig_edges, use_container_width=True)

    overview_map = draw_node_map(
        visible_nodes if not visible_nodes.empty else nodes,
        title="National view of road sections",
        color_col="project_count",
        highlight_nodes=scenario_nodes(
            visible_scenarios.loc[visible_scenarios["scenario_id"] == selected_scenario_id, "closed_node_ids"].iloc[0]
        ) if selected_scenario_id else None,
    )
    st.plotly_chart(overview_map, use_container_width=True)

    st.success(
        """
**Current conclusion**

- RF is best for local cracking prediction.
- The spatial graph performs best for most static disruption proxies.
- EXPERIMENT_SECTION treatment features clearly improve the temporal GCN.
- Monthly climate helps explain treatment categories, but not the main cracking model.
- Graph-aware maintenance portfolio optimisation is the next required step.
"""
    )

# Page 3
with tabs[2]:
    page_intro_box("This page explains what each dataset contributes to the overall pipeline.")
    show_core_terms_box()
    data_source_table = pd.DataFrame(
        [
            ["General Section Info.xlsx", "Section metadata, route context, construction and treatment history", "Defines sections, route context, and treatment / experiment history"],
            ["Analysis Ready Distress.xlsx", "Cracking and other pavement condition indicators", "Used for pre/post treatment analysis and one-year cracking prediction"],
            ["Annual Traffic Inputs Over Time.xlsx", "Annual truck traffic, AADT-style loading, ESAL/GESAL", "Defines section importance and annual traffic context"],
            ["MERRA annual climate files", "Annual temperature, humidity, precipitation, wind, and solar summaries", "Used in graph features and annual temporal modelling"],
            ["MERRA monthly climate files", "Monthly climate time series", "Used only for seasonal and extreme event-study climate summaries"],
            ["EXPERIMENT_SECTION", "Treatment/change events with dates and reason labels", "Main source for treatment semantics and treatment-aware temporal features"],
            ["graph_data/nodes.csv", "Prepared section-level node table", "Main node table for the graph and visualisation"],
            ["graph_data/edges.csv", "Prepared structural relationships between sections", "Defines graph topology variants"],
        ],
        columns=["Data source", "What it contains", "How it is used"],
    )
    st.dataframe(data_source_table, use_container_width=True, hide_index=True)
    st.info(
        """
**How to read the data**

- **Distress** means pavement condition, especially cracking.
- **Traffic** means annual loading and section importance.
- **Climate** means environmental exposure.
- **Coordinates and routes** are what allow the graph to be built.
"""
    )
    st.warning(
        "Traffic is annual, not monthly. Therefore, the project estimates potential disruption using graph-based proxies rather than observed short-term traffic diversion."
    )

# Page 4
with tabs[3]:
    page_intro_box(
        """
The project uses a **section-level interdependency graph**, not a full national operational road network.

- **One node = one LTPP pavement section**
- **One edge = one plausible interdependency between two sections**
"""
    )
    metric_card_row(
        [
            ("Visible road sections", compact_number(len(visible_nodes) if not visible_nodes.empty else len(nodes)), "Filtered by the current state selection"),
            ("Visible relationships", compact_number(len(visible_edges)), "Edges where both end nodes pass the state filter"),
            ("Selected states", str(len(selected_states)) if selected_states else "All", "How much of the map is currently visible"),
            ("Largest local component", compact_number(graph_diag.loc[graph_diag["graph_variant"] == graph_variant, "largest_component_nodes"].iloc[0]), "Largest connected set of sections"),
        ]
    )

    st.markdown("### Graph variants")
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
        variant_table[
            ["Graph variant", "Nodes", "Edges", "Average degree", "Isolated nodes", "Connected components", "Largest component share (%)"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"""
**What is currently displayed**

- **Graph variant:** {GRAPH_VARIANTS[graph_variant]}
- **Meaning:** {GRAPH_VARIANT_STORY[graph_variant]}
- **State filter:** {', '.join(selected_state_labels) if selected_state_labels else 'All states'}
- **Interpretation:** the LTPP sections form many **local interdependency networks**, not one complete national road network.
"""
    )

    selected_view_nodes = visible_nodes if not visible_nodes.empty else nodes
    st.plotly_chart(
        draw_network_web(
            selected_view_nodes,
            visible_edges,
            "Spider-web view of the selected graph",
            focus_node=focus_node,
            max_edges=max_edges,
            color_by_cluster=False,
            height=760,
        ),
        width="stretch",
        key="graph_construction_web",
    )
    st.caption("This view drops the basemap and emphasises the actual graph relationships between the visible sections.")

    st.plotly_chart(
        draw_cluster_map(
            selected_view_nodes,
            visible_edges,
            "Cluster view of the selected graph",
            height=760,
        ),
        width="stretch",
        key="graph_construction_clusters",
    )
    st.caption("Each colour corresponds to one connected local cluster of road sections in the currently selected graph and state filter.")

    with st.expander("Show raw graph diagnostics"):
        st.dataframe(graph_diag, use_container_width=True)
        st.dataframe(graph_distance, use_container_width=True)

# Page 5
with tabs[4]:
    page_intro_box(
        """
The static disruption model does **not** use observed traffic assignment. It uses **synthetic graph-based target variables**
to estimate how disruptive a set of simultaneous section closures could be.
"""
    )
    st.info(
        """
**Key terms on this page**

- **Origin-destination pair (OD pair)**: a possible trip from one important road section to another.
- **Shortest path**: the quickest available route through the graph.
- **Proxy**: a synthetic indicator used because observed short-term diversion traffic is not available in the dataset.
"""
    )
    st.markdown("### Target definitions")
    st.latex(r"\Delta VHT_{proxy} = \sum_{od} \left[w_{od} \times \max(T_1 - T_0, 0)\right] \quad \text{if connected}")
    st.latex(r"\Delta VHT_{proxy} = \sum_{od} \left[w_{od} \times 4 \times T_0\right] \quad \text{if disconnected}")
    st.latex(r"L_{conn} = 1 - \frac{A^{max}_1}{A^{max}_0}")
    st.latex(r"D_{OD} = \frac{\sum w_{od} I_{disconnected}}{\sum w_{od}}")
    st.latex(r"S = \Delta VHT_{proxy} \times (1 + L_{conn} + D_{OD})")

    st.markdown(
        """
- **T0** = baseline shortest-path travel time
- **T1** = post-closure shortest-path travel time
- **w_od** = weight of one origin-destination pair, so more important trips count more heavily
- **A0_max** = baseline largest connected component asset weight
- **A1_max** = post-closure largest connected component asset weight
"""
    )

    defs = static_target_defs.copy()
    defs["Target"] = defs["target_variable"].map(lambda x: FRIENDLY_TARGETS.get(x, x))
    defs["Meaning"] = defs["plain_english"]
    defs["Formula"] = defs["formula"]
    st.dataframe(defs[["Target", "Formula", "Meaning", "higher_means", "observed_or_proxy"]], use_container_width=True, hide_index=True)

    st.warning("These are synthetic graph-based proxies, not observed traffic impacts.")

    st.markdown("### Worked scenario example")
    summary_example = static_target_worked[static_target_worked["row_type"] == "scenario_summary"].copy()
    if not summary_example.empty:
        row = summary_example.iloc[0]
        metric_card_row(
            [
                ("Scenario", str(row["scenario_id"]), "Worked example from the stored scenario outputs"),
                ("Closed nodes", str(row["closed_node_ids"]), "Sections removed from the residual graph"),
                ("Extra travel-time proxy", compact_number(row["delta_vht_proxy"]), "Synthetic extra travel penalty"),
                ("Connectivity loss share", f"{safe_float(row['connectivity_loss_pct']):.4f}", "Largest-component asset loss"),
                ("Disconnected origin-destination share", f"{safe_float(row['disconnected_od_pct']):.4f}", "Weighted share of unreachable origin-destination pairs"),
                ("Overall disruption score", compact_number(row["disruption_score"]), "Composite disruption proxy"),
            ]
        )
    top_pairs = static_target_worked[static_target_worked["row_type"] == "top_affected_od_pairs"].copy()
    if not top_pairs.empty:
        st.dataframe(
            top_pairs[
                [
                    "origin",
                    "destination",
                    "base_travel_time_hours",
                    "new_travel_time_hours",
                    "od_weight",
                    "od_disconnected",
                    "od_penalty_hours",
                ]
            ].head(15),
            use_container_width=True,
            hide_index=True,
        )
    st.info(
        """
**What this means**

The static model learns how closure combinations affect a **synthetic, demand-weighted set of important OD pairs**.
This makes the targets defensible as disruption proxies, but they should not be described as observed traffic impacts.
"""
    )

# Page 6
with tabs[5]:
    page_intro_box(
        """
The static graph model learns:

**scenario + graph + section features → predicted network disruption**
"""
    )
    static_table = build_static_result_table(graph_variant_comparison)
    st.dataframe(static_table, use_container_width=True, hide_index=True)

    static_rows = graph_variant_comparison[graph_variant_comparison["model_family"] == "static"].copy()
    static_rows["Graph variant"] = static_rows["graph_variant"].map(GRAPH_VARIANTS)
    static_rows["Target"] = static_rows["target"].map(lambda x: FRIENDLY_TARGETS.get(x, x))
    fig_static = px.bar(
        static_rows,
        x="Graph variant",
        y="gcn_test_r2",
        color="Target",
        barmode="group",
        title="Static graph-model R² by graph variant and disruption target",
    )
    st.plotly_chart(fig_static, use_container_width=True)

    predictions = variant_predictions.copy()
    for raw, friendly in FRIENDLY_TARGETS.items():
        if raw in predictions.columns:
            predictions[friendly] = predictions[raw]
    c1, c2 = st.columns(2)
    with c1:
        fig_scatter_1 = px.scatter(
            predictions,
            x="delta_vht_proxy",
            y="pred_delta_vht_proxy",
            color="split",
            title="Actual vs predicted extra travel-time proxy",
            labels={"delta_vht_proxy": "Actual", "pred_delta_vht_proxy": "Predicted"},
        )
        st.plotly_chart(fig_scatter_1, use_container_width=True)
    with c2:
        fig_scatter_2 = px.scatter(
            predictions,
            x="disruption_score",
            y="pred_disruption_score",
            color="split",
            title="Actual vs predicted overall disruption score",
            labels={"disruption_score": "Actual", "pred_disruption_score": "Predicted"},
        )
        st.plotly_chart(fig_scatter_2, use_container_width=True)

    st.success(
        """
**What this means**

- The **spatial graph** performs best for most static disruption targets.
- For this shortest-path disruption task, nearby sections mattered more than richer similarity links.
- The model estimates **potential disruption of simultaneous outages**, not observed diversion behaviour.
"""
    )

# Page 7
with tabs[6]:
    page_intro_box(
        """
The temporal model asks a different question:

**Can neighbour and treatment context help explain next-year cracking?**
"""
    )
    st.info(
        """
**How to read these model names**

- **RF local** means a **Random Forest** model using only section-level features.
- **GCN** means a **Graph Convolutional Network** that also uses neighbouring sections in the graph.
- **R²** measures predictive fit. Higher is better, with `1` meaning perfect fit.
"""
    )
    temporal_table = build_temporal_result_table(treatment_ablation)
    st.dataframe(temporal_table, use_container_width=True, hide_index=True)

    latest = treatment_ablation[treatment_ablation["treatment_mode"] == "experiment"].iloc[0]
    metric_card_row(
        [
            ("RF local R²", f"{safe_float(latest['rf_test_r2']):.3f}", "Asset-level baseline"),
            ("GCN without project/treatment features R²", f"{safe_float(latest['gcn_without_project_treatment_test_r2']):.3f}", "Graph model without treatment context"),
            ("GCN with project/treatment features R²", f"{safe_float(latest['gcn_with_project_treatment_test_r2']):.3f}", "Graph model with EXPERIMENT_SECTION features"),
            ("GCN gain from treatment features", f"{safe_float(latest['gcn_project_treatment_r2_gain']):.3f}", "Added relational signal from treatment/project context"),
        ]
    )

    temporal_long = pd.DataFrame(
        {
            "Model": [
                "RF local",
                "GCN without project/treatment features",
                "GCN with EXPERIMENT_SECTION treatment features",
            ],
            "Test R²": [
                latest["rf_test_r2"],
                latest["gcn_without_project_treatment_test_r2"],
                latest["gcn_with_project_treatment_test_r2"],
            ],
        }
    )
    fig_temp = px.bar(temporal_long, x="Model", y="Test R²", title="One-year cracking prediction results")
    st.plotly_chart(fig_temp, use_container_width=True)

    st.info(
        """
**What this means**

- **RF local** remains the strongest model for one-year cracking prediction.
- The graph model should **not** be framed as “beating RF”.
- Instead, the useful result is that **EXPERIMENT_SECTION treatment features substantially improve the GCN**, which means treatment / project context contains relational signal.
"""
    )

# Page 8
with tabs[7]:
    page_intro_box(
        """
`PROJECT_HIST_AGE_EXP` turned out to be too broad for treatment semantics.
`EXPERIMENT_SECTION` provides a better treatment/project history representation.
"""
    )
    key_cols = pd.DataFrame(
        {
            "Column": [
                "CONSTRUCTION_NO",
                "CN_ASSIGN_DATE",
                "ASSIGN_DATE",
                "DEASSIGN_DATE",
                "CN_CHANGE_REASON",
                "CN_CHANGE_REASON_EXP",
                "STATUS / STATUS_EXP",
                "EXPERIMENT_NO / EXPERIMENT_NO_EXP",
            ],
            "Why it matters": [
                "construction phase / version",
                "construction-number change date",
                "assignment date",
                "deassignment date",
                "coded change reason",
                "decoded treatment/change label",
                "status semantics",
                "experiment semantics",
            ],
        }
    )
    st.dataframe(key_cols, use_container_width=True, hide_index=True)

    group_counts = treatment_counts[treatment_counts["breakdown"] == "broad_treatment_group"].copy()
    group_counts["Treatment group"] = group_counts["broad_treatment_group"].map(lambda x: TREATMENT_LABELS.get(x, x))
    fig_treat = px.bar(group_counts, x="Treatment group", y="count", title="How many treatment/change events fall into each broad treatment group?")
    st.plotly_chart(fig_treat, use_container_width=True)

    st.warning(
        """
**Interpretation**

- Not all treatment/project events are caused by cracking.
- These records represent broader **maintenance, rehabilitation, construction-number changes, and experiment changes**.
- In the dissertation, use the language **treatment/project history**, not “cracking-caused maintenance”.
"""
    )

# Page 9
with tabs[8]:
    page_intro_box(
        """
This event-study analysis asks whether different treatment groups show different before/after patterns
and whether neighbouring sections move in similar ways.
"""
    )
    group_view = event_by_group.copy()
    group_view["Treatment group"] = group_view["broad_treatment_group"].map(lambda x: TREATMENT_LABELS.get(x, x))
    chart_df = group_view[
        [
            "Treatment group",
            "pre_cracking_3yr_median",
            "post_cracking_3yr_median",
            "treated_minus_neighbour_cracking_change_median",
            "traffic_change_pre_to_post_median",
        ]
    ].copy()
    chart_long = chart_df.melt(id_vars="Treatment group", var_name="Metric", value_name="Value")
    chart_long["Metric"] = chart_long["Metric"].replace(
        {
            "pre_cracking_3yr_median": "Pre-event cracking (median)",
            "post_cracking_3yr_median": "Post-event cracking (median)",
            "treated_minus_neighbour_cracking_change_median": "Treated minus neighbour cracking change",
            "traffic_change_pre_to_post_median": "Traffic change pre to post",
        }
    )
    fig_event = px.bar(chart_long, x="Treatment group", y="Value", color="Metric", barmode="group", title="Treatment groups show different before/after profiles")
    st.plotly_chart(fig_event, use_container_width=True)

    if selected_case_node:
        chosen_case = case_studies[case_studies["node_id"] == selected_case_node].iloc[0]
        st.markdown("### Dissertation-ready case study")
        metric_card_row(
            [
                ("Section", str(chosen_case["node_id"]), "Selected case-study node"),
                ("State", str(chosen_case["state_name"]), "State context"),
                ("Route", str(chosen_case["route_key"]), "Route / corridor label"),
                ("Treatment group", TREATMENT_LABELS.get(chosen_case["treatment_group"], str(chosen_case["treatment_group"])), "Broad treatment category"),
            ]
        )
        st.write(chosen_case["short_interpretation"])
        st.caption(f"Limitation: {chosen_case['limitations_caution']}")

    st.info(
        """
**Main event-study reading**

- **Asphalt overlay** and **reconstruction / major rehab** show the clearest treated-section improvement.
- **Crack sealing** starts with low cracking and tends to remain stable.
- **Seal coat** looks more preventive and more associated with warmer / drier / sunnier contexts.
- **Patching** is mixed.
- Neighbours do move in some cases, but the evidence is not strong enough to claim a general causal spillover effect.
"""
    )
    st.warning("This event-study is exploratory, not causal.")

# Page 10
with tabs[9]:
    st.info(
        """
**Climate interpretation on this page**

Monthly climate is useful because it captures **seasonality, extremes, and short-term exposure windows** that annual averages can hide.
Examples include heat spikes, freeze-thaw months, very wet months, or unusually high solar exposure before a treatment event.
"""
    )
    page_intro_box(
        """
Monthly climate was kept out of the main annual cracking model, but it was tested as an explanatory layer for treatment types.
"""
    )
    classifier_keep = event_classifier[
        event_classifier["feature_set"].isin(
            [
                "annual_climate_only",
                "monthly_climate_only",
                "distress_plus_traffic_plus_annual",
                "distress_plus_traffic_plus_monthly",
                "distress_plus_traffic_plus_annual_plus_monthly",
            ]
        )
        & (event_classifier["model"] == "random_forest")
    ].copy()
    classifier_keep["Feature set"] = classifier_keep["feature_set"].replace(
        {
            "annual_climate_only": "Annual climate only",
            "monthly_climate_only": "Monthly climate only",
            "distress_plus_traffic_plus_annual": "Distress + traffic + annual climate",
            "distress_plus_traffic_plus_monthly": "Distress + traffic + monthly climate",
            "distress_plus_traffic_plus_annual_plus_monthly": "Distress + traffic + annual + monthly climate",
        }
    )
    fig_cls = px.bar(
        classifier_keep,
        x="Feature set",
        y=["macro_f1", "balanced_accuracy"],
        barmode="group",
        title="Does monthly climate help explain treatment categories?",
    )
    st.plotly_chart(fig_cls, use_container_width=True)

    top_monthly = event_importance[event_importance["feature_set"] == "monthly_climate_only"].sort_values("importance", ascending=False).head(12)
    fig_imp = px.bar(top_monthly, x="importance", y="feature", orientation="h", title="Most informative monthly climate features")
    st.plotly_chart(fig_imp, use_container_width=True)

    redundancy_counts = monthly_redundancy["keep_redundant_recommendation"].value_counts().rename_axis("Recommendation").reset_index(name="Count")
    fig_red = px.pie(redundancy_counts, names="Recommendation", values="Count", title="How many monthly climate features are useful vs redundant?")
    st.plotly_chart(fig_red, use_container_width=True)

    st.success(
        """
**What this means**

- Monthly climate helps explain **treatment categories** better than annual climate alone.
- The most useful monthly variables are about **shortwave exposure, temperature extremes/variation, freeze-thaw, and humidity variability**.
- Long 12–36 month monthly averages are often redundant with annual climate.
- Monthly climate **did not improve the main annual cracking model** after the corrected MERRA mapping.
"""
    )

# Page 11
with tabs[10]:
    page_intro_box("This page summarises the full dissertation story from asset-level records to network-aware maintenance planning.")
    st.markdown(
        """
### Final conclusions

1. The LTPP data can be transformed from asset-level records into a **section-level interdependency graph**.
2. The graph is fragmented, so it should be interpreted as **local interdependency networks**, not a full national road network.
3. **Spatial graph structure** works best for proxy disruption prediction.
4. **RF local** is best for one-year cracking prediction.
5. **EXPERIMENT_SECTION treatment features** improve the GCN, showing useful treatment-context signal.
6. **Monthly climate** adds information for treatment-category explanation.
7. Annual traffic data do not allow direct measurement of short-term traffic diversion.
8. Therefore, disruption is modelled using **graph-based synthetic proxies**.
9. The next step is **graph-aware maintenance portfolio optimisation**.
"""
    )
    st.success(
        """
**Next-step pipeline**

`Condition need from RF`
  
`+`
  
`Network disruption from graph model`
  
`+`
  
`Treatment semantics from EXPERIMENT_SECTION`
  
`+`
  
`Budget and conflict constraints`
  
`=`
  
`Maintenance portfolio optimisation`
"""
    )

# Page 12
with tabs[11]:
    page_intro_box("This appendix keeps the raw diagnostics available without interrupting the main dissertation narrative.")
    appendix_files = [
        "graph_diagnostics.csv",
        "graph_distance_summary.csv",
        "climate_mapping_diagnostics.csv",
        "monthlyagg_feature_diagnostics.csv",
        "graph_variant_model_comparison.csv",
        "monthly_climate_ablation.csv",
        "treatment_feature_ablation.csv",
        "experiment_event_study_by_group.csv",
        "experiment_event_study_climate_by_group.csv",
        "experiment_event_study_treatment_classifier.csv",
        "experiment_event_study_feature_importance.csv",
        "experiment_event_study_neighbour_vs_control.csv",
        "static_target_variable_definitions.csv",
        "static_target_worked_example.csv",
    ]
    for name in appendix_files:
        with st.expander(f"Show {name}"):
            path = REPORT_DIR / name
            if path.suffix.lower() == ".csv":
                st.dataframe(pd.read_csv(path, low_memory=False), use_container_width=True)
            else:
                st.code(path.read_text(), language="json")

    if advanced_mode:
        with st.expander("Show selected variant scenario predictions"):
            st.dataframe(variant_predictions.head(200), use_container_width=True)
