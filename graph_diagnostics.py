"""Graph diagnostics for dissertation reporting.

Purpose:
- quantify the sparsity and locality of the refined section-level graph,
- compare graph variants used by the static and temporal models.

Inputs:
- graph_data/nodes.csv
- graph_data/edges.csv

Outputs:
- reports/graph_diagnostics.csv
- reports/graph_diagnostics.json
- reports/graph_distance_summary.csv

Command:
- ./.venv/bin/python graph_diagnostics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"


def filter_edges_for_variant(edges: pd.DataFrame, graph_variant: str) -> pd.DataFrame:
    variant_to_types = {
        "spatial": {"spatial"},
        "spatial_route": {"spatial", "same_route"},
        "full_refined": {"spatial", "same_route", "same_functional_class"},
    }
    return edges[edges["edge_type"].isin(variant_to_types[graph_variant])].copy()


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].astype(str).tolist())
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.source), str(row.target), edge_type=str(row.edge_type), distance_km=float(getattr(row, "distance_km", np.nan)))
    return graph


def graph_summary(nodes: pd.DataFrame, edges: pd.DataFrame, graph_variant: str) -> dict[str, object]:
    graph = build_graph(nodes, edges)
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    isolated = int((degrees == 0).sum())
    components = list(nx.connected_components(graph))
    component_sizes = sorted((len(comp) for comp in components), reverse=True)
    largest_component = int(component_sizes[0]) if component_sizes else 0
    largest_share = float(largest_component / len(nodes)) if len(nodes) else 0.0
    edge_type_counts = edges["edge_type"].value_counts().to_dict()
    return {
        "graph_variant": graph_variant,
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "edge_type_counts": edge_type_counts,
        "average_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "median_degree": float(np.median(degrees)) if len(degrees) else 0.0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "isolated_nodes": isolated,
        "isolated_share_pct": float(100.0 * isolated / len(nodes)) if len(nodes) else 0.0,
        "connected_components": int(len(components)),
        "largest_component_nodes": largest_component,
        "largest_component_share_pct": float(100.0 * largest_share),
    }


def distance_summary(edges: pd.DataFrame, graph_variant: str) -> pd.DataFrame:
    work = edges.copy()
    work["distance_km"] = pd.to_numeric(work["distance_km"], errors="coerce")
    summary = (
        work.groupby("edge_type")["distance_km"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    summary.insert(0, "graph_variant", graph_variant)
    return summary


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)

    summaries: list[dict[str, object]] = []
    distance_frames: list[pd.DataFrame] = []
    for graph_variant in ["spatial", "spatial_route", "full_refined"]:
        variant_edges = filter_edges_for_variant(edges, graph_variant)
        summaries.append(graph_summary(nodes, variant_edges, graph_variant))
        distance_frames.append(distance_summary(variant_edges, graph_variant))

    summary_frame = pd.DataFrame(
        [
            {
                "graph_variant": row["graph_variant"],
                "nodes": row["nodes"],
                "edges": row["edges"],
                "average_degree": row["average_degree"],
                "median_degree": row["median_degree"],
                "max_degree": row["max_degree"],
                "isolated_nodes": row["isolated_nodes"],
                "isolated_share_pct": row["isolated_share_pct"],
                "connected_components": row["connected_components"],
                "largest_component_nodes": row["largest_component_nodes"],
                "largest_component_share_pct": row["largest_component_share_pct"],
                "spatial_edges": row["edge_type_counts"].get("spatial", 0),
                "same_route_edges": row["edge_type_counts"].get("same_route", 0),
                "same_functional_class_edges": row["edge_type_counts"].get("same_functional_class", 0),
            }
            for row in summaries
        ]
    )
    distance_frame = pd.concat(distance_frames, ignore_index=True)

    summary_frame.to_csv(REPORT_DIR / "graph_diagnostics.csv", index=False)
    distance_frame.to_csv(REPORT_DIR / "graph_distance_summary.csv", index=False)
    with open(REPORT_DIR / "graph_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump({"variants": summaries}, fh, indent=2)

    print(summary_frame.to_string(index=False))
    print("\nDistance summary")
    print(distance_frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
