"""Faster OSM audit for spatial edges.

This audit is designed for the dissertation question:
"If I close this segment, is the neighbouring spatial edge a plausible local
 road neighbour on the real drive network?"

Compared with the initial version, this script is faster because it:
- samples edges stratified by distance bin
- groups sampled edges by coarse geographic tile
- downloads one OSM drive graph per group instead of per edge
- processes groups in parallel
- logs progress every 10 audited edges
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import sys
import threading
import time
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part1_extensions import (
    GRAPH_DIR,
    REPORT_DIR,
    OSM_MAX_COMPONENT_DIAGONAL_KM,
    OSM_MAX_SNAP_M,
    bbox_diagonal_km,
    classify_topology_status,
    edge_distance_bin,
    log,
    osm_edge_signature,
)


DISTANCE_BIN_ORDER = ["0-1 km", "1-5 km", "5-10 km", "10-25 km", "25-50 km", ">50 km"]
DEFAULT_WORKERS = 2
BUFFER_DEG = 0.08
CACHE_DIR = ROOT / ".osm_cache"
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
_OSM_SETTINGS_LOCK = threading.Lock()

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.cache_folder = str(CACHE_DIR)
ox.settings.requests_timeout = 90


def _edge_attrs_for_step(graph: nx.MultiDiGraph, u: int, v: int) -> dict[str, object]:
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        return {}
    return min(edge_data.values(), key=lambda d: float(d.get("length", math.inf)))


def _tokenize(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        raw_items = text.replace("|", ";").replace("/", ";").split(";")
    out = []
    for item in raw_items:
        token = str(item).strip().lower()
        if token and token != "nan":
            out.append(token)
    return out


def shortest_path_with_metrics(
    graph: nx.MultiDiGraph,
    src_edge: tuple[int, int, int],
    dst_edge: tuple[int, int, int],
) -> dict[str, object]:
    if src_edge == dst_edge:
        sig = osm_edge_signature(graph, src_edge)
        dominant_ref = sorted(sig["ref"])[0] if sig["ref"] else ""
        dominant_name = sorted(sig["name"])[0] if sig["name"] else ""
        return {
            "path_len_m": 0.0,
            "route_change_count_ref": 0,
            "route_change_count_name": 0,
            "unique_ref_count": len(sig["ref"]),
            "unique_name_count": len(sig["name"]),
            "dominant_ref": dominant_ref,
            "dominant_name": dominant_name,
            "step_count": 0,
        }

    best = None
    for left in (src_edge[0], src_edge[1]):
        for right in (dst_edge[0], dst_edge[1]):
            try:
                path = nx.shortest_path(graph, left, right, weight="length")
            except Exception:
                continue
            if len(path) < 2:
                continue
            total_m = 0.0
            ref_sequence: list[str] = []
            name_sequence: list[str] = []
            for u, v in zip(path[:-1], path[1:]):
                attrs = _edge_attrs_for_step(graph, u, v)
                total_m += float(attrs.get("length", 0.0))
                refs = _tokenize(attrs.get("ref"))
                names = _tokenize(attrs.get("name"))
                ref_sequence.append(refs[0] if refs else "")
                name_sequence.append(names[0] if names else "")
            if total_m <= 0:
                continue
            candidate = {
                "path_len_m": total_m,
                "ref_sequence": ref_sequence,
                "name_sequence": name_sequence,
                "step_count": len(path) - 1,
            }
            if best is None or total_m < best["path_len_m"]:
                best = candidate

    if best is None:
        return {
            "path_len_m": math.nan,
            "route_change_count_ref": math.nan,
            "route_change_count_name": math.nan,
            "unique_ref_count": math.nan,
            "unique_name_count": math.nan,
            "dominant_ref": "",
            "dominant_name": "",
            "step_count": math.nan,
        }

    def count_changes(sequence: list[str]) -> int:
        cleaned = [item for item in sequence if item]
        if len(cleaned) <= 1:
            return 0
        return sum(1 for prev, cur in zip(cleaned[:-1], cleaned[1:]) if cur != prev)

    ref_counts = pd.Series([item for item in best["ref_sequence"] if item]).value_counts()
    name_counts = pd.Series([item for item in best["name_sequence"] if item]).value_counts()
    return {
        "path_len_m": float(best["path_len_m"]),
        "route_change_count_ref": int(count_changes(best["ref_sequence"])),
        "route_change_count_name": int(count_changes(best["name_sequence"])),
        "unique_ref_count": int(ref_counts.size),
        "unique_name_count": int(name_counts.size),
        "dominant_ref": str(ref_counts.index[0]) if not ref_counts.empty else "",
        "dominant_name": str(name_counts.index[0]) if not name_counts.empty else "",
        "step_count": int(best["step_count"]),
    }


def spatial_audit_verdict(
    topology_status: str,
    detour_ratio: float,
    route_change_count_ref: float,
    route_change_count_name: float,
    distance_km: float,
) -> str:
    if topology_status == "same_osm_edge":
        return "very_strong_local_neighbour"
    if topology_status == "unreachable":
        return "not_supported"
    if topology_status == "snap_too_far":
        return "map_match_invalid"
    if not np.isfinite(detour_ratio):
        return "unknown"
    finite_changes = [x for x in [route_change_count_ref, route_change_count_name] if np.isfinite(x)]
    route_changes = min(finite_changes) if finite_changes else math.nan
    if detour_ratio <= 1.5 and (not np.isfinite(route_changes) or route_changes <= 1):
        return "strong_local_neighbour"
    if detour_ratio <= 3.0 and (not np.isfinite(route_changes) or route_changes <= 2):
        return "plausible_local_neighbour"
    if distance_km <= 5 and detour_ratio <= 5.0:
        return "ambiguous_but_close"
    return "weak_local_neighbour"


def sample_spatial_edges(edges: pd.DataFrame, max_edges: int, seed: int) -> pd.DataFrame:
    spatial = edges[edges["edge_type"].eq("spatial")].copy()
    spatial["distance_km"] = pd.to_numeric(spatial["distance_km"], errors="coerce")
    spatial = spatial[np.isfinite(spatial["distance_km"])].copy()
    spatial["distance_bin"] = spatial["distance_km"].map(edge_distance_bin)
    spatial["distance_bin"] = pd.Categorical(spatial["distance_bin"], categories=DISTANCE_BIN_ORDER, ordered=True)

    rng = np.random.default_rng(seed)
    groups = {bin_name: grp.copy() for bin_name, grp in spatial.groupby("distance_bin", observed=True)}
    total = min(max_edges, len(spatial))
    if total == len(spatial):
        return spatial.sort_values(["distance_bin", "distance_km"]).reset_index(drop=True)

    counts = {name: len(grp) for name, grp in groups.items()}
    total_available = sum(counts.values())
    raw_alloc = {name: total * count / total_available for name, count in counts.items()}
    alloc = {name: min(counts[name], int(math.floor(raw_alloc[name]))) for name in counts}
    remaining = total - sum(alloc.values())
    remainders = sorted(
        counts,
        key=lambda name: (raw_alloc[name] - alloc[name], counts[name] - alloc[name]),
        reverse=True,
    )
    for name in remainders:
        if remaining <= 0:
            break
        if alloc[name] < counts[name]:
            alloc[name] += 1
            remaining -= 1

    sampled = []
    for name in DISTANCE_BIN_ORDER:
        grp = groups.get(name)
        n = alloc.get(name, 0)
        if grp is None or n <= 0:
            continue
        take_idx = rng.choice(len(grp), size=n, replace=False)
        sampled.append(grp.iloc[take_idx])
    return pd.concat(sampled, ignore_index=True).sort_values(["distance_bin", "distance_km"]).reset_index(drop=True)


def download_graph_with_retry(west: float, south: float, east: float, north: float, max_attempts: int = 6):
    delay = 3.0
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        mirror = OVERPASS_MIRRORS[(attempt - 1) % len(OVERPASS_MIRRORS)]
        try:
            with _OSM_SETTINGS_LOCK:
                ox.settings.overpass_url = mirror
                graph = ox.graph_from_bbox(
                    (west, south, east, north),
                    network_type="drive",
                    simplify=True,
                    retain_all=True,
                    truncate_by_edge=True,
                )
            return graph
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            recoverable = any(
                token in msg
                for token in [
                    "max retries",
                    "connectionpool",
                    "failed to establish a new connection",
                    "connection refused",
                    "429",
                    "timeout",
                    "remote disconnected",
                    "connection aborted",
                    "504",
                    "502",
                    "503",
                ]
            )
            if attempt == max_attempts or not recoverable:
                raise
            time.sleep(delay)
            delay = min(delay * 1.7, 30.0)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OSM download failed without exception")


def group_key(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[int, int, int, int]:
    lat_floor_a = int(math.floor(lat1))
    lon_floor_a = int(math.floor(lon1))
    lat_floor_b = int(math.floor(lat2))
    lon_floor_b = int(math.floor(lon2))
    return (
        min(lat_floor_a, lat_floor_b),
        min(lon_floor_a, lon_floor_b),
        max(lat_floor_a, lat_floor_b),
        max(lon_floor_a, lon_floor_b),
    )


def group_bbox(records: list[dict[str, object]], buffer_deg: float = BUFFER_DEG) -> tuple[float, float, float, float]:
    lats = []
    lons = []
    for rec in records:
        lats.extend([float(rec["src_lat"]), float(rec["dst_lat"])])
        lons.extend([float(rec["src_lon"]), float(rec["dst_lon"])])
    return max(lats) + buffer_deg, min(lats) - buffer_deg, max(lons) + buffer_deg, min(lons) - buffer_deg


def build_sample_records(sampled: pd.DataFrame, node_lookup: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in sampled.itertuples(index=False):
        src = str(row.source)
        dst = str(row.target)
        src_lat = float(node_lookup.at[src, "latitude"])
        src_lon = float(node_lookup.at[src, "longitude"])
        dst_lat = float(node_lookup.at[dst, "latitude"])
        dst_lon = float(node_lookup.at[dst, "longitude"])
        records.append(
            {
                "source": src,
                "target": dst,
                "distance_km": float(row.distance_km),
                "distance_bin": str(row.distance_bin),
                "src_lat": src_lat,
                "src_lon": src_lon,
                "dst_lat": dst_lat,
                "dst_lon": dst_lon,
                "group_key": group_key(src_lat, src_lon, dst_lat, dst_lon),
            }
        )
    return records


def process_group(gkey: tuple[int, int, int, int], records: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    north, south, east, west = group_bbox(records, buffer_deg=BUFFER_DEG)
    diagonal_km = bbox_diagonal_km(north, south, east, west)

    if diagonal_km > OSM_MAX_COMPONENT_DIAGONAL_KM:
        for rec in records:
            rows.append(
                {
                    "source": rec["source"],
                    "target": rec["target"],
                    "distance_km": rec["distance_km"],
                    "distance_bin": rec["distance_bin"],
                    "bbox_diagonal_km": diagonal_km,
                    "topology_status": "bbox_too_large",
                    "path_exists": 0,
                    "audit_verdict": "needs_manual_map_check",
                }
            )
        return rows, failures

    try:
        # The timeout helper relies on signal.alarm, which cannot be used from
        # worker threads. For the parallel audit, rely on OSMnx's request
        # timeout instead and fetch directly inside each thread.
        osm_graph = download_graph_with_retry(west, south, east, north, max_attempts=6)
        osm_graph_proj = ox.project_graph(osm_graph)
        transformer = Transformer.from_crs("EPSG:4326", osm_graph_proj.graph["crs"], always_xy=True)
        src_x, src_y = transformer.transform(
            [float(rec["src_lon"]) for rec in records],
            [float(rec["src_lat"]) for rec in records],
        )
        dst_x, dst_y = transformer.transform(
            [float(rec["dst_lon"]) for rec in records],
            [float(rec["dst_lat"]) for rec in records],
        )
        src_edge_arr, src_dist_arr = ox.distance.nearest_edges(osm_graph_proj, X=list(src_x), Y=list(src_y), return_dist=True)
        dst_edge_arr, dst_dist_arr = ox.distance.nearest_edges(osm_graph_proj, X=list(dst_x), Y=list(dst_y), return_dist=True)
    except Exception as exc:
        for rec in records:
            failures.append({"source": rec["source"], "target": rec["target"], "error": str(exc)})
            rows.append(
                {
                    "source": rec["source"],
                    "target": rec["target"],
                    "distance_km": rec["distance_km"],
                    "distance_bin": rec["distance_bin"],
                    "bbox_diagonal_km": diagonal_km,
                    "topology_status": "fetch_or_match_failed",
                    "path_exists": 0,
                    "audit_verdict": "needs_manual_map_check",
                }
            )
        return rows, failures

    for rec, src_edge_raw, src_snap_m, dst_edge_raw, dst_snap_m in zip(records, src_edge_arr, src_dist_arr, dst_edge_arr, dst_dist_arr):
        src_edge = tuple(src_edge_raw)
        dst_edge = tuple(dst_edge_raw)
        same_osm_edge = int(src_edge == dst_edge)
        snap_ok = bool(float(src_snap_m) <= OSM_MAX_SNAP_M and float(dst_snap_m) <= OSM_MAX_SNAP_M)
        if snap_ok:
            path_metrics = shortest_path_with_metrics(osm_graph_proj, src_edge, dst_edge)
            path_len_m = float(path_metrics["path_len_m"]) if np.isfinite(path_metrics["path_len_m"]) else math.nan
        else:
            path_metrics = {
                "path_len_m": math.nan,
                "route_change_count_ref": math.nan,
                "route_change_count_name": math.nan,
                "unique_ref_count": math.nan,
                "unique_name_count": math.nan,
                "dominant_ref": "",
                "dominant_name": "",
                "step_count": math.nan,
            }
            path_len_m = math.nan
        detour_ratio = (
            float(path_len_m / (float(rec["distance_km"]) * 1000.0))
            if np.isfinite(path_len_m) and float(rec["distance_km"]) > 0
            else math.nan
        )
        topology_status = classify_topology_status(snap_ok, bool(same_osm_edge), path_len_m, detour_ratio)
        src_sig = osm_edge_signature(osm_graph_proj, src_edge)
        dst_sig = osm_edge_signature(osm_graph_proj, dst_edge)
        rows.append(
            {
                "source": rec["source"],
                "target": rec["target"],
                "distance_km": rec["distance_km"],
                "distance_bin": rec["distance_bin"],
                "bbox_diagonal_km": diagonal_km,
                "src_snap_m": float(src_snap_m),
                "dst_snap_m": float(dst_snap_m),
                "same_osm_edge": same_osm_edge,
                "path_exists": int(np.isfinite(path_len_m)),
                "osm_path_km": float(path_len_m / 1000.0) if np.isfinite(path_len_m) else math.nan,
                "detour_ratio": detour_ratio,
                "topology_status": topology_status,
                "route_change_count_ref": path_metrics["route_change_count_ref"],
                "route_change_count_name": path_metrics["route_change_count_name"],
                "unique_ref_count": path_metrics["unique_ref_count"],
                "unique_name_count": path_metrics["unique_name_count"],
                "dominant_ref": path_metrics["dominant_ref"],
                "dominant_name": path_metrics["dominant_name"],
                "src_ref": ";".join(sorted(src_sig["ref"])),
                "dst_ref": ";".join(sorted(dst_sig["ref"])),
                "src_name": ";".join(sorted(src_sig["name"])),
                "dst_name": ";".join(sorted(dst_sig["name"])),
                "endpoint_ref_overlap": int(bool(src_sig["ref"] & dst_sig["ref"])),
                "endpoint_name_overlap": int(bool(src_sig["name"] & dst_sig["name"])),
                "path_step_count": path_metrics["step_count"],
                "audit_verdict": spatial_audit_verdict(
                    topology_status=topology_status,
                    detour_ratio=detour_ratio,
                    route_change_count_ref=float(path_metrics["route_change_count_ref"])
                    if np.isfinite(path_metrics["route_change_count_ref"])
                    else math.nan,
                    route_change_count_name=float(path_metrics["route_change_count_name"])
                    if np.isfinite(path_metrics["route_change_count_name"])
                    else math.nan,
                    distance_km=float(rec["distance_km"]),
                ),
            }
        )
    return rows, failures


def run_audit(max_edges: int, seed: int, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    node_lookup = nodes.set_index("node_id")[["latitude", "longitude"]].copy()

    sampled = sample_spatial_edges(edges, max_edges=max_edges, seed=seed)
    records = build_sample_records(sampled, node_lookup)
    grouped: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}
    for rec in records:
        grouped.setdefault(rec["group_key"], []).append(rec)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    completed = 0
    next_report = 1

    log(
        f"Spatial OSM audit grouped {len(records)} sampled edges into {len(grouped)} OSM fetch groups "
        f"(workers={workers})"
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_group, gkey, recs): gkey for gkey, recs in grouped.items()}
        for future in as_completed(futures):
            group_rows, group_failures = future.result()
            rows.extend(group_rows)
            failures.extend(group_failures)
            completed += len(group_rows)
            while completed >= next_report:
                log(f"Spatial OSM audit progress: {next_report}/{len(records)} edges completed")
                next_report += 1

    audit = pd.DataFrame(rows)
    if not audit.empty:
        expected_numeric = [
            "osm_path_km",
            "detour_ratio",
            "route_change_count_ref",
            "route_change_count_name",
            "distance_km",
        ]
        expected_object = [
            "audit_verdict",
            "topology_status",
            "distance_bin",
        ]
        for col in expected_numeric:
            if col not in audit.columns:
                audit[col] = np.nan
            audit[col] = pd.to_numeric(audit[col], errors="coerce")
        for col in expected_object:
            if col not in audit.columns:
                audit[col] = ""
        audit["distance_bin"] = pd.Categorical(audit["distance_bin"], categories=DISTANCE_BIN_ORDER, ordered=True)
        audit = audit.sort_values(["distance_bin", "distance_km", "source", "target"]).reset_index(drop=True)
    summary = (
        audit.groupby(["distance_bin", "audit_verdict"], dropna=False, as_index=False)
        .agg(
            edges=("source", "size"),
            mean_distance_km=("distance_km", "mean"),
            mean_osm_path_km=("osm_path_km", "mean"),
            mean_detour_ratio=("detour_ratio", "mean"),
            mean_route_change_count_ref=("route_change_count_ref", "mean"),
            mean_route_change_count_name=("route_change_count_name", "mean"),
        )
        .sort_values(["distance_bin", "audit_verdict"])
    )
    meta = {
        "sample_size_requested": int(max_edges),
        "sample_size_audited": int(len(audit)),
        "sampling": "stratified_by_distance_bin",
        "grouping": "integer_degree_tile_batched",
        "group_count": int(len(grouped)),
        "workers": int(workers),
        "distance_bins": DISTANCE_BIN_ORDER,
        "failure_examples": failures[:10],
    }
    return audit, summary, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-edges", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    audit, summary, meta = run_audit(max_edges=args.max_edges, seed=args.seed, workers=args.workers)
    audit.to_csv(REPORT_DIR / "spatial_edge_osm_audit.csv", index=False)
    summary.to_csv(REPORT_DIR / "spatial_edge_osm_audit_summary.csv", index=False)
    (REPORT_DIR / "spatial_edge_osm_audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(
        "Spatial OSM audit written to reports/spatial_edge_osm_audit.csv, "
        "reports/spatial_edge_osm_audit_summary.csv, and reports/spatial_edge_osm_audit_meta.json"
    )


if __name__ == "__main__":
    main()
