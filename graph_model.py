"""Train a graph surrogate for network-wide maintenance disruption scenarios.

This version answers the thesis question more directly than the previous
node-overlap prototype:
- define explicit network targets (delta travel-hours proxy, connectivity loss,
  disconnected OD share),
- generate thousands of simultaneous project-closure scenarios,
- train a graph model on scenario masks,
- export artifacts for a Streamlit app that visualizes scenario-level impacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation import compare_models, compute_metrics, dataframe_to_markdown, save_metrics_table

ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"
SEED = 42
SCENARIO_COUNT = 1500
MIN_CLOSED = 1
MAX_CLOSED = 5
OD_MAX_PER_STATE = 16
OD_MAX_PAIRS = 120
DEFAULT_SPEED_MPH = {"corridor": 60.0, "spatial": 45.0, "diversion": 35.0}
EDGE_PENALTY = {"corridor": 1.0, "spatial": 1.25, "diversion": 1.6}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(message: str) -> None:
    print(f"[network_surrogate] {message}")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def pick_numeric_col(frame: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def load_graph_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    projects = pd.read_csv(GRAPH_DIR / "projects.csv", low_memory=False)
    project_conflicts = pd.read_csv(GRAPH_DIR / "project_conflicts.csv", low_memory=False)
    return nodes, edges, projects, project_conflicts


def filter_edges_for_variant(edges: pd.DataFrame, graph_variant: str) -> pd.DataFrame:
    variant_to_types = {
        "spatial": {"spatial"},
        "spatial_route": {"spatial", "same_route"},
        "full_refined": {"spatial", "same_route", "same_functional_class"},
    }
    allowed = variant_to_types[graph_variant]
    return edges[edges["edge_type"].isin(allowed)].copy()


def prepare_nodes(nodes: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    work = nodes.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["NO_OF_LANES"] = pd.to_numeric(work.get("NO_OF_LANES"), errors="coerce").fillna(1.0)
    work["SECTION_LENGTH"] = pd.to_numeric(work.get("SECTION_LENGTH"), errors="coerce").fillna(1.0)
    work["SPEED_LIMIT"] = pd.to_numeric(work.get("SPEED_LIMIT"), errors="coerce").fillna(55.0)
    work["MILEPOINT"] = pd.to_numeric(work.get("MILEPOINT"), errors="coerce")
    work["functional_class"] = pd.to_numeric(work.get("functional_class"), errors="coerce").fillna(-1.0)
    work["route_key"] = work.get("route_key", pd.Series("unknown", index=work.index)).fillna("unknown").astype(str)
    work["state_code"] = work.get("state_code", pd.Series("unknown", index=work.index)).astype(str)

    annual_truck_volume = pick_numeric_col(work, ["traffic_trf_trend_1_ANNUAL_TRUCK_VOLUME_TREND"])
    truck_aadtt = pick_numeric_col(work, ["traffic_trf_trend_1_AADTT_ALL_TRUCKS_TREND"])
    esal = pick_numeric_col(work, ["traffic_trf_trend_ANNUAL_ESAL_TREND", "traffic_trf_trend_ANNUAL_GESAL_TREND"])
    raw_demand = annual_truck_volume + 365.0 * truck_aadtt + 0.02 * esal
    raw_demand = raw_demand.where(raw_demand > 0, annual_truck_volume.where(annual_truck_volume > 0, 1.0))
    work["demand_weight"] = raw_demand.fillna(1.0).clip(lower=1.0)
    work["asset_weight"] = work["demand_weight"] * work["SECTION_LENGTH"].clip(lower=0.1)

    project_counts = projects.groupby("node_id").size().rename("project_count")
    work = work.merge(project_counts, left_on="node_id", right_index=True, how="left")
    work["project_count"] = work["project_count"].fillna(0).astype(int)
    work = work.dropna(subset=["latitude", "longitude"]).copy()
    return work


def build_corridor_edges(nodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for route_key, group in nodes.groupby("route_key"):
        if route_key == "unknown" or len(group) < 2:
            continue
        group = group.copy()
        if group["MILEPOINT"].notna().sum() >= 2:
            group = group.sort_values("MILEPOINT")
        else:
            anchor_lat = group["latitude"].mean()
            group["_anchor_distance"] = group.apply(
                lambda row: haversine_km(anchor_lat, group["longitude"].mean(), row["latitude"], row["longitude"]),
                axis=1,
            )
            group = group.sort_values("_anchor_distance")
        entries = list(group.itertuples(index=False))
        for left, right in zip(entries, entries[1:]):
            distance_km = haversine_km(left.latitude, left.longitude, right.latitude, right.longitude)
            rows.append(
                {
                    "source": str(left.node_id),
                    "target": str(right.node_id),
                    "edge_role": "corridor",
                    "distance_km": float(distance_km),
                    "diversion_potential": 1.0,
                }
            )
    return pd.DataFrame(rows)


def build_spatial_and_diversion_edges(nodes: pd.DataFrame, raw_edges: pd.DataFrame, include_diversion: bool) -> pd.DataFrame:
    coords = nodes.set_index("node_id")[["latitude", "longitude", "state_code", "SPEED_LIMIT"]].copy()
    spatial_rows: list[dict[str, object]] = []
    diversion_rows: list[dict[str, object]] = []

    spatial = raw_edges[raw_edges["edge_type"] == "spatial"].copy()
    spatial = spatial[spatial["source"].isin(coords.index) & spatial["target"].isin(coords.index)].copy()
    spatial["distance_km"] = pd.to_numeric(spatial["distance_km"], errors="coerce")
    spatial = spatial[spatial["distance_km"].fillna(9999) <= 120.0].copy()
    for node_id, group in spatial.groupby("source"):
        subset = group.sort_values(["distance_km", "diversion_potential"], ascending=[True, False]).head(4)
        for row in subset.itertuples(index=False):
            spatial_rows.append(
                {
                    "source": str(row.source),
                    "target": str(row.target),
                    "edge_role": "spatial",
                    "distance_km": float(row.distance_km),
                    "diversion_potential": float(getattr(row, "diversion_potential", 0.2) or 0.2),
                }
            )

    if include_diversion:
        func_edges = raw_edges[raw_edges["edge_type"] == "same_functional_class"].copy()
        func_edges = func_edges[func_edges["source"].isin(coords.index) & func_edges["target"].isin(coords.index)].copy()
        func_edges["distance_km"] = [
            haversine_km(coords.at[src, "latitude"], coords.at[src, "longitude"], coords.at[dst, "latitude"], coords.at[dst, "longitude"])
            for src, dst in zip(func_edges["source"], func_edges["target"])
        ]
        func_edges = func_edges[func_edges["distance_km"] <= 150.0].copy()
        for node_id, group in func_edges.groupby("source"):
            subset = group.sort_values("distance_km").head(3)
            for row in subset.itertuples(index=False):
                diversion_rows.append(
                    {
                        "source": str(row.source),
                        "target": str(row.target),
                        "edge_role": "diversion",
                        "distance_km": float(row.distance_km),
                        "diversion_potential": float(getattr(row, "diversion_potential", 0.1) or 0.1),
                    }
                )

    return pd.concat([pd.DataFrame(spatial_rows), pd.DataFrame(diversion_rows)], ignore_index=True)


def finalize_network_edges(nodes: pd.DataFrame, raw_edges: pd.DataFrame, graph_variant: str) -> pd.DataFrame:
    corridor = build_corridor_edges(nodes) if graph_variant in {"spatial_route", "full_refined"} else pd.DataFrame()
    extras = build_spatial_and_diversion_edges(
        nodes,
        raw_edges,
        include_diversion=graph_variant == "full_refined",
    )
    edges = pd.concat([corridor, extras], ignore_index=True)
    if edges.empty:
        raise ValueError("No network edges were generated for the disruption graph.")

    ordered_pairs = edges.apply(
        lambda row: tuple(sorted((str(row["source"]), str(row["target"])))),
        axis=1,
    )
    edges["source"] = ordered_pairs.map(lambda pair: pair[0])
    edges["target"] = ordered_pairs.map(lambda pair: pair[1])
    edges = edges.drop_duplicates(subset=["source", "target", "edge_role"]).copy()

    speed_lookup = nodes.set_index("node_id")["SPEED_LIMIT"].to_dict()
    lane_lookup = nodes.set_index("node_id")["NO_OF_LANES"].to_dict()
    demand_lookup = nodes.set_index("node_id")["demand_weight"].to_dict()

    def edge_speed(row: pd.Series) -> float:
        src_speed = float(speed_lookup.get(row["source"], DEFAULT_SPEED_MPH[row["edge_role"]]))
        dst_speed = float(speed_lookup.get(row["target"], DEFAULT_SPEED_MPH[row["edge_role"]]))
        speed = np.nanmean([src_speed, dst_speed])
        if not np.isfinite(speed) or speed <= 0:
            speed = DEFAULT_SPEED_MPH[row["edge_role"]]
        return float(speed)

    edges["speed_mph"] = edges.apply(edge_speed, axis=1)
    edges["travel_time_hours"] = (
        edges["distance_km"].clip(lower=0.05) / (edges["speed_mph"].clip(lower=15.0) * 1.60934)
    ) * edges["edge_role"].map(EDGE_PENALTY)
    edges["capacity_proxy"] = (
        edges["source"].map(lane_lookup).fillna(1.0) + edges["target"].map(lane_lookup).fillna(1.0)
    ) / 2.0
    edges["demand_proxy"] = np.sqrt(
        edges["source"].map(demand_lookup).fillna(1.0) * edges["target"].map(demand_lookup).fillna(1.0)
    )
    edges = edges.sort_values(["edge_role", "source", "target"]).reset_index(drop=True)
    return edges


def build_network_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(
            str(row.node_id),
            demand_weight=float(row.demand_weight),
            asset_weight=float(row.asset_weight),
            state_code=str(row.state_code),
        )
    for row in edges.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            edge_role=str(row.edge_role),
            travel_time_hours=float(row.travel_time_hours),
            distance_km=float(row.distance_km),
            diversion_potential=float(row.diversion_potential),
        )
    return graph


def build_od_pairs(nodes: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    candidates = nodes[nodes["project_count"] > 0].copy()
    candidates = candidates.sort_values(["state_code", "demand_weight"], ascending=[True, False])
    rows: list[dict[str, object]] = []
    for state_code, group in candidates.groupby("state_code"):
        chosen = group.head(OD_MAX_PER_STATE)
        if len(chosen) < 2:
            continue
        entries = list(chosen.itertuples(index=False))
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                left = entries[i]
                right = entries[j]
                if left.node_id not in graph or right.node_id not in graph:
                    continue
                try:
                    base_time = nx.shortest_path_length(
                        graph,
                        source=str(left.node_id),
                        target=str(right.node_id),
                        weight="travel_time_hours",
                    )
                except nx.NetworkXNoPath:
                    continue
                if not np.isfinite(base_time):
                    continue
                flow = float(np.sqrt(left.demand_weight * right.demand_weight))
                rows.append(
                    {
                        "origin": str(left.node_id),
                        "destination": str(right.node_id),
                        "state_code": str(state_code),
                        "base_travel_time_hours": float(base_time),
                        "od_weight": float(flow),
                    }
                )
    od_pairs = pd.DataFrame(rows)
    if od_pairs.empty:
        raise ValueError("No valid OD pairs were built; cannot define network targets.")
    od_pairs["weighted_base_hours"] = od_pairs["base_travel_time_hours"] * od_pairs["od_weight"]
    od_pairs = od_pairs.sort_values("weighted_base_hours", ascending=False).head(OD_MAX_PAIRS).reset_index(drop=True)
    return od_pairs


def build_conflict_map(project_conflicts: pd.DataFrame) -> dict[str, list[str]]:
    if project_conflicts.empty:
        return {}
    group = (
        project_conflicts.groupby(["source_node", "target_node"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    conflict_map: dict[str, list[str]] = {}
    for row in group.itertuples(index=False):
        conflict_map.setdefault(str(row.source_node), []).append(str(row.target_node))
        conflict_map.setdefault(str(row.target_node), []).append(str(row.source_node))
    return conflict_map


def generate_scenarios(nodes: pd.DataFrame, project_conflicts: pd.DataFrame, count: int) -> pd.DataFrame:
    project_nodes = nodes[nodes["project_count"] > 0]["node_id"].tolist()
    project_meta = nodes.set_index("node_id")[["route_key", "state_code", "demand_weight"]].to_dict("index")
    by_route: dict[tuple[str, str], list[str]] = {}
    for node_id, meta in project_meta.items():
        by_route.setdefault((str(meta["state_code"]), str(meta["route_key"])), []).append(node_id)
    conflict_map = build_conflict_map(project_conflicts)

    seen: set[tuple[str, ...]] = set()
    rows: list[dict[str, object]] = []
    tries = 0
    while len(rows) < count and tries < count * 20:
        tries += 1
        size = random.randint(MIN_CLOSED, MAX_CLOSED)
        seed_node = random.choice(project_nodes)
        mode = random.random()
        chosen = {seed_node}
        if mode < 0.35 and seed_node in conflict_map:
            pool = [n for n in conflict_map[seed_node] if n in project_meta]
            random.shuffle(pool)
            chosen.update(pool[: size - 1])
        elif mode < 0.7:
            meta = project_meta[seed_node]
            pool = by_route.get((str(meta["state_code"]), str(meta["route_key"])), []).copy()
            random.shuffle(pool)
            chosen.update([n for n in pool if n != seed_node][: size - 1])
        else:
            weighted = nodes[nodes["project_count"] > 0].sample(
                n=min(size, len(project_nodes)),
                weights=nodes.loc[nodes["project_count"] > 0, "demand_weight"],
                replace=False,
                random_state=random.randint(0, 1_000_000),
            )
            chosen.update(weighted["node_id"].tolist())

        scenario_nodes = tuple(sorted(chosen))
        if len(scenario_nodes) < MIN_CLOSED or scenario_nodes in seen:
            continue
        seen.add(scenario_nodes)
        rows.append(
            {
                "scenario_id": f"scenario_{len(rows):04d}",
                "num_closed_nodes": int(len(scenario_nodes)),
                "closed_node_ids": ";".join(scenario_nodes),
            }
        )
    return pd.DataFrame(rows)


def compute_network_targets(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    od_pairs: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    base_total_asset = float(nodes["asset_weight"].sum())
    base_vht_proxy = float((od_pairs["base_travel_time_hours"] * od_pairs["od_weight"]).sum())
    node_asset = nodes.set_index("node_id")["asset_weight"].to_dict()
    base_components = list(nx.connected_components(graph))
    base_largest_asset = max(
        (sum(node_asset.get(node, 0.0) for node in comp) for comp in base_components),
        default=base_total_asset,
    )
    total_od_weight = float(od_pairs["od_weight"].sum())
    unique_origins = sorted(od_pairs["origin"].unique().tolist())
    od_grouped = {origin: od_pairs[od_pairs["origin"] == origin].copy() for origin in unique_origins}
    rows: list[dict[str, object]] = []

    for row in scenarios.itertuples(index=False):
        closed_nodes = [node for node in str(row.closed_node_ids).split(";") if node]
        residual = graph.copy()
        residual.remove_nodes_from(closed_nodes)

        if residual.number_of_nodes() == 0 or residual.number_of_edges() == 0:
            connectivity_loss = 1.0
            delta_vht_proxy = base_vht_proxy * 4.0
            disconnected_od_pct = 1.0
        else:
            components = list(nx.connected_components(residual))
            largest_asset = max(
                (sum(node_asset.get(node, 0.0) for node in comp) for comp in components),
                default=0.0,
            )
            connectivity_loss = 1.0 - (largest_asset / base_largest_asset if base_largest_asset > 0 else 1.0)

            travel_penalty = 0.0
            disconnected_weight = 0.0
            for origin in unique_origins:
                od_subset = od_grouped[origin]
                if origin not in residual:
                    disconnected_weight += float(od_subset["od_weight"].sum())
                    for od in od_subset.itertuples(index=False):
                        travel_penalty += float(od.od_weight * od.base_travel_time_hours * 4.0)
                    continue
                path_lengths = nx.single_source_dijkstra_path_length(
                    residual,
                    source=origin,
                    weight="travel_time_hours",
                )
                for od in od_subset.itertuples(index=False):
                    base_time = float(od.base_travel_time_hours)
                    weight = float(od.od_weight)
                    new_time = path_lengths.get(str(od.destination))
                    if new_time is None:
                        disconnected_weight += weight
                        travel_penalty += weight * base_time * 4.0
                    else:
                        travel_penalty += weight * max(float(new_time) - base_time, 0.0)

            delta_vht_proxy = float(travel_penalty)
            disconnected_od_pct = float(disconnected_weight / total_od_weight) if total_od_weight > 0 else 0.0

        rows.append(
            {
                "scenario_id": str(row.scenario_id),
                "num_closed_nodes": int(row.num_closed_nodes),
                "closed_node_ids": str(row.closed_node_ids),
                "delta_vht_proxy": float(delta_vht_proxy),
                "connectivity_loss_pct": float(connectivity_loss),
                "disconnected_od_pct": float(disconnected_od_pct),
                "disruption_score": float(delta_vht_proxy * (1.0 + connectivity_loss + disconnected_od_pct)),
            }
        )
        if len(rows) % 100 == 0:
            log(f"computed network targets for {len(rows)}/{len(scenarios)} scenarios")

    summary = {
        "base_vht_proxy": base_vht_proxy,
        "base_total_asset": base_total_asset,
        "base_largest_component_asset": float(base_largest_asset),
        "total_od_weight": total_od_weight,
        "num_od_pairs": int(len(od_pairs)),
        "num_scenarios": int(len(rows)),
    }
    return pd.DataFrame(rows), summary


def build_node_feature_table(nodes: pd.DataFrame, network_edges: pd.DataFrame) -> pd.DataFrame:
    climate_prefixes = ("temp_bind_", "humid_", "precip_", "wind_", "solar_", "temp_year_")
    climate_cols = [
        col for col in nodes.columns
        if col.startswith(climate_prefixes) and pd.api.types.is_numeric_dtype(nodes[col])
    ]
    degree = pd.concat(
        [
            network_edges[["source", "edge_role"]].rename(columns={"source": "node_id"}),
            network_edges[["target", "edge_role"]].rename(columns={"target": "node_id"}),
        ],
        ignore_index=True,
    )
    degree = degree.groupby(["node_id", "edge_role"]).size().unstack(fill_value=0)
    degree.columns = [f"degree_{col}" for col in degree.columns]
    work = nodes[
        [
            "node_id",
            "latitude",
            "longitude",
            "NO_OF_LANES",
            "SECTION_LENGTH",
            "SPEED_LIMIT",
            "functional_class",
            "demand_weight",
            "asset_weight",
            "project_count",
            *climate_cols,
        ]
    ].copy()
    work = work.merge(degree.reset_index(), on="node_id", how="left").fillna(0.0)
    return work


def build_dense_adjacency(nodes: pd.DataFrame, network_edges: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    node_ids = nodes["node_id"].tolist()
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
    for row in network_edges.itertuples(index=False):
        i = index[str(row.source)]
        j = index[str(row.target)]
        weight = float(1.0 / max(row.travel_time_hours, 1e-3))
        adjacency[i, j] = max(adjacency[i, j], weight)
        adjacency[j, i] = max(adjacency[j, i], weight)
    adjacency += np.eye(len(node_ids), dtype=np.float32)
    deg = adjacency.sum(axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    normalized = adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
    return normalized.astype(np.float32), index


def build_scenario_graph_filter_features(
    node_features: pd.DataFrame,
    scenarios: pd.DataFrame,
    adjacency: np.ndarray,
    node_index: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    feature_table = node_features.set_index("node_id").copy()
    demand = feature_table["demand_weight"].to_numpy(dtype=np.float32)
    asset = feature_table["asset_weight"].to_numpy(dtype=np.float32)
    project = feature_table["project_count"].to_numpy(dtype=np.float32)
    lanes = feature_table["NO_OF_LANES"].to_numpy(dtype=np.float32)
    speed = feature_table["SPEED_LIMIT"].to_numpy(dtype=np.float32)
    route_degree = feature_table.get("degree_corridor", pd.Series(0.0, index=feature_table.index)).to_numpy(dtype=np.float32)
    spatial_degree = feature_table.get("degree_spatial", pd.Series(0.0, index=feature_table.index)).to_numpy(dtype=np.float32)
    diversion_degree = feature_table.get("degree_diversion", pd.Series(0.0, index=feature_table.index)).to_numpy(dtype=np.float32)
    total_degree = route_degree + spatial_degree + diversion_degree
    climate_cols = [
        col for col in feature_table.columns
        if col.startswith(("temp_bind_", "humid_", "precip_", "wind_", "solar_", "temp_year_"))
    ]
    climate_matrix = feature_table[climate_cols].fillna(feature_table[climate_cols].median()).to_numpy(dtype=np.float32) if climate_cols else np.zeros((len(feature_table), 0), dtype=np.float32)

    mask_matrix = np.zeros((len(scenarios), len(node_features)), dtype=np.float32)
    for s_idx, row in enumerate(scenarios.itertuples(index=False)):
        for node_id in str(row.closed_node_ids).split(";"):
            idx = node_index.get(node_id)
            if idx is not None:
                mask_matrix[s_idx, idx] = 1.0

    first_hop = mask_matrix @ adjacency.T
    second_hop = first_hop @ adjacency.T

    features = pd.DataFrame(
        {
            "num_closed_nodes": mask_matrix.sum(axis=1),
            "closed_demand": mask_matrix @ demand,
            "closed_asset": mask_matrix @ asset,
            "closed_project_load": mask_matrix @ project,
            "closed_lane_exposure": mask_matrix @ lanes,
            "closed_speed_exposure": mask_matrix @ speed,
            "closed_total_degree": mask_matrix @ total_degree,
            "first_hop_demand": first_hop @ demand,
            "first_hop_asset": first_hop @ asset,
            "first_hop_degree": first_hop @ total_degree,
            "second_hop_demand": second_hop @ demand,
            "second_hop_asset": second_hop @ asset,
            "second_hop_degree": second_hop @ total_degree,
            "first_hop_spatial": first_hop @ spatial_degree,
            "first_hop_diversion": first_hop @ diversion_degree,
            "route_concentration": mask_matrix @ route_degree,
        }
    )
    if climate_cols:
        for col_idx, col_name in enumerate(climate_cols):
            safe = col_name.replace("temp_bind_", "tb_").replace("temp_year_", "ty_").replace("precip_", "pr_").replace("humid_", "hu_").replace("wind_", "wi_").replace("solar_", "so_")
            values = climate_matrix[:, col_idx]
            features[f"closed_{safe}"] = mask_matrix @ values
            features[f"firsthop_{safe}"] = first_hop @ values
    scaler = StandardScaler()
    X = scaler.fit_transform(features).astype(np.float32)
    return X, features.columns.tolist()


class GraphFilterNet(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class FitArtifacts:
    model: GraphFilterNet
    target_scaler: StandardScaler
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    predictions: np.ndarray
    ridge_predictions: np.ndarray


def train_scenario_model(
    scenario_features: np.ndarray,
    targets: np.ndarray,
) -> FitArtifacts:
    indices = np.arange(len(scenario_features))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=SEED)

    x_t = torch.tensor(scenario_features, dtype=torch.float32)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(targets)
    y_t = torch.tensor(y_scaled, dtype=torch.float32)

    model = GraphFilterNet(input_dim=scenario_features.shape[-1], hidden_dim=64, out_dim=targets.shape[1], dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    patience = 25
    no_improve = 0

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)

    for epoch in range(1, 181):
        model.train()
        optimizer.zero_grad()
        pred = model(x_t)
        loss = F.mse_loss(pred[train_idx_t], y_t[train_idx_t])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_loss = F.mse_loss(model(x_t)[val_idx_t], y_t[val_idx_t]).item()
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 20 == 0:
            log(f"epoch={epoch:03d} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        pred_scaled = model(x_t).cpu().numpy()
    predictions = target_scaler.inverse_transform(pred_scaled)

    ridge = Ridge(alpha=1.0)
    ridge.fit(scenario_features[train_idx], y_scaled[train_idx])
    ridge_predictions = target_scaler.inverse_transform(ridge.predict(scenario_features))

    return FitArtifacts(
        model=model,
        target_scaler=target_scaler,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        predictions=predictions,
        ridge_predictions=ridge_predictions,
    )


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return compute_metrics(y_true, y_pred)


def build_metrics(target_names: list[str], targets: np.ndarray, artifacts: FitArtifacts) -> dict[str, object]:
    metrics: dict[str, object] = {"targets": {}}
    for target_idx, target_name in enumerate(target_names):
        y = targets[:, target_idx]
        y_pred = artifacts.predictions[:, target_idx]
        ridge_pred = artifacts.ridge_predictions[:, target_idx]
        metrics["targets"][target_name] = {
            "gcn": {
                "train": metric_block(y[artifacts.train_idx], y_pred[artifacts.train_idx]),
                "val": metric_block(y[artifacts.val_idx], y_pred[artifacts.val_idx]),
                "test": metric_block(y[artifacts.test_idx], y_pred[artifacts.test_idx]),
            },
            "ridge": {
                "train": metric_block(y[artifacts.train_idx], ridge_pred[artifacts.train_idx]),
                "val": metric_block(y[artifacts.val_idx], ridge_pred[artifacts.val_idx]),
                "test": metric_block(y[artifacts.test_idx], ridge_pred[artifacts.test_idx]),
            },
        }
    return metrics


def build_node_importance(
    nodes: pd.DataFrame,
    scenarios: pd.DataFrame,
    scenario_results: pd.DataFrame,
    scenario_predictions: pd.DataFrame,
) -> pd.DataFrame:
    del nodes, scenario_results
    single_node = scenarios[scenarios["num_closed_nodes"] == 1].copy()
    if single_node.empty:
        return pd.DataFrame(columns=["node_id"])
    merged = single_node.merge(
        scenario_predictions[["scenario_id", "pred_delta_vht_proxy", "pred_connectivity_loss_pct", "pred_disconnected_od_pct", "pred_disruption_score"]],
        on="scenario_id",
        how="left",
    )
    merged["node_id"] = merged["closed_node_ids"]
    cols = [
        "node_id",
        "delta_vht_proxy",
        "pred_delta_vht_proxy",
        "connectivity_loss_pct",
        "pred_connectivity_loss_pct",
        "disconnected_od_pct",
        "pred_disconnected_od_pct",
        "disruption_score",
        "pred_disruption_score",
    ]
    return merged[cols].sort_values("delta_vht_proxy", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the static scenario-impact graph surrogate.")
    parser.add_argument(
        "--graph-variant",
        choices=["spatial", "spatial_route", "full_refined"],
        default="full_refined",
    )
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def tagged_name(base: str, output_tag: str, suffix: str) -> Path:
    stem = f"{base}_{output_tag}" if output_tag else base
    return GRAPH_DIR / f"{stem}{suffix}"


def main() -> None:
    args = parse_args()
    set_seed()
    nodes_raw, edges_raw, projects, project_conflicts = load_graph_data()
    edges_raw = filter_edges_for_variant(edges_raw, args.graph_variant)
    nodes = prepare_nodes(nodes_raw, projects)
    log(f"Prepared {len(nodes)} nodes with demand and project attributes")

    network_edges = finalize_network_edges(nodes, edges_raw, args.graph_variant)
    network_edges.to_csv(tagged_name("network_edges_research", args.output_tag, ".csv"), index=False)
    log(f"Derived research network with {len(network_edges)} edges for variant={args.graph_variant}")

    graph = build_network_graph(nodes, network_edges)
    od_pairs = build_od_pairs(nodes, graph)
    od_pairs.to_csv(tagged_name("network_od_pairs", args.output_tag, ".csv"), index=False)
    log(f"Built {len(od_pairs)} OD pairs for the disruption proxy")

    scenarios = generate_scenarios(nodes, project_conflicts, SCENARIO_COUNT)
    scenario_targets, network_summary = compute_network_targets(graph, nodes, od_pairs, scenarios)
    scenarios = scenarios.merge(scenario_targets, on=["scenario_id", "num_closed_nodes", "closed_node_ids"], how="left")
    scenarios.to_csv(tagged_name("network_scenarios", args.output_tag, ".csv"), index=False)
    log(f"Generated {len(scenarios)} scenarios with explicit network targets")

    node_features = build_node_feature_table(nodes, network_edges)
    node_features.to_csv(tagged_name("network_node_features", args.output_tag, ".csv"), index=False)
    adjacency, node_index = build_dense_adjacency(nodes, network_edges)
    scenario_features, feature_names = build_scenario_graph_filter_features(node_features, scenarios, adjacency, node_index)
    pd.DataFrame(scenario_features, columns=feature_names).to_csv(tagged_name("network_scenario_features", args.output_tag, ".csv"), index=False)

    target_names = ["delta_vht_proxy", "connectivity_loss_pct", "disconnected_od_pct", "disruption_score"]
    targets = scenarios[target_names].to_numpy(dtype=np.float32)
    artifacts = train_scenario_model(scenario_features, targets)
    metrics = build_metrics(target_names, targets, artifacts)

    prediction_frame = scenarios[["scenario_id", "num_closed_nodes", "closed_node_ids", *target_names]].copy()
    for idx, name in enumerate(target_names):
        prediction_frame[f"pred_{name}"] = artifacts.predictions[:, idx]
        prediction_frame[f"ridge_{name}"] = artifacts.ridge_predictions[:, idx]
    prediction_frame["split"] = "train"
    prediction_frame.loc[artifacts.val_idx, "split"] = "val"
    prediction_frame.loc[artifacts.test_idx, "split"] = "test"
    prediction_frame.to_csv(tagged_name("network_scenario_predictions", args.output_tag, ".csv"), index=False)

    node_impacts = build_node_importance(nodes, scenarios, scenario_targets, prediction_frame)
    node_impacts.to_csv(tagged_name("network_node_impacts", args.output_tag, ".csv"), index=False)

    top_scenarios = prediction_frame.sort_values("delta_vht_proxy", ascending=False).head(25)
    top_scenarios[["scenario_id", "closed_node_ids", "delta_vht_proxy", "pred_delta_vht_proxy", "connectivity_loss_pct", "pred_connectivity_loss_pct", "disconnected_od_pct", "pred_disconnected_od_pct"]].to_csv(
        tagged_name("scenario_network_impacts", args.output_tag, ".csv"),
        index=False,
    )

    summary = {
        "question_alignment": {
            "graph_representation": True,
            "projects_mapped_to_nodes": True,
            "interdependencies_encoded": ["corridor", "spatial_diversion", "same_state_diversion", "simultaneous_project_conflicts"],
            "targets": target_names,
            "scenario_count": int(len(scenarios)),
        },
        "network": {
            "graph_variant": args.graph_variant,
            "nodes": int(len(nodes)),
            "edges": int(len(network_edges)),
            "od_pairs": int(len(od_pairs)),
            **network_summary,
        },
        "model_metrics": metrics,
        "feature_names": feature_names,
        "outputs": {
            "network_edges": str(tagged_name("network_edges_research", args.output_tag, ".csv")),
            "od_pairs": str(tagged_name("network_od_pairs", args.output_tag, ".csv")),
            "scenarios": str(tagged_name("network_scenarios", args.output_tag, ".csv")),
            "predictions": str(tagged_name("network_scenario_predictions", args.output_tag, ".csv")),
            "node_impacts": str(tagged_name("network_node_impacts", args.output_tag, ".csv")),
        },
    }
    metrics_path = tagged_name("network_model_metrics", args.output_tag, ".json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    comparison_tables: dict[str, object] = {}
    for target_name in target_names:
        target_results = {
            "GCN": metrics["targets"][target_name]["gcn"],
            "Ridge": metrics["targets"][target_name]["ridge"],
        }
        comparison_df = compare_models(target_results)
        comparison_tables[target_name] = comparison_df.to_dict(orient="records")
        base_name = f"{target_name}_{args.output_tag}" if args.output_tag else target_name
        save_metrics_table(
            comparison_df,
            tagged_name(f"{base_name}_static_model_metrics_v2", "", ".json"),
            tagged_name(f"{base_name}_static_model_metrics_v2", "", ".md"),
        )
        print(f"\n### Static comparison for {target_name}")
        print(dataframe_to_markdown(comparison_df))

    gcn_metrics_v2 = {
        "graph_variant": args.graph_variant,
        "model": "GCN",
        "targets": {target_name: metrics["targets"][target_name]["gcn"] for target_name in target_names},
    }
    ridge_metrics_v2 = {
        "graph_variant": args.graph_variant,
        "model": "Ridge",
        "targets": {target_name: metrics["targets"][target_name]["ridge"] for target_name in target_names},
    }
    gcn_metrics_v2_path = tagged_name("gcn_metrics_v2", args.output_tag, ".json")
    ridge_metrics_v2_path = tagged_name("ridge_metrics_v2", args.output_tag, ".json")
    with open(gcn_metrics_v2_path, "w", encoding="utf-8") as fh:
        json.dump(gcn_metrics_v2, fh, indent=2)
    with open(ridge_metrics_v2_path, "w", encoding="utf-8") as fh:
        json.dump(ridge_metrics_v2, fh, indent=2)

    torch.save(
        {
            "model_state_dict": artifacts.model.state_dict(),
            "scenario_feature_names": feature_names,
            "feature_names": feature_names,
            "target_names": target_names,
        },
        tagged_name("network_gnn_state", args.output_tag, ".pt"),
    )

    log(json.dumps(summary["network"], indent=2))
    for target_name in target_names:
        gcn_test = metrics["targets"][target_name]["gcn"]["test"]
        ridge_test = metrics["targets"][target_name]["ridge"]["test"]
        log(
            f"{target_name}: GNN test R2={gcn_test['r2']:.3f}, RMSE={gcn_test['rmse']:.3f} | "
            f"Ridge test R2={ridge_test['r2']:.3f}, RMSE={ridge_test['rmse']:.3f}"
        )


if __name__ == "__main__":
    main()
