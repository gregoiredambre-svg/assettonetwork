"""Full OSM-topology validation of the comprehensive graph (~18k edges).

Optimised in two ways:

1. Tile grouping: edges are grouped by 1-degree spatial tile so that one OSM
   download covers many edges (instead of one OSM download per edge).
2. Parallelism: tiles are processed concurrently via a ThreadPoolExecutor
   (OSM downloads are I/O-bound, so threads work well).

For each (source, target) edge, we query OSM (once per tile), find the nearest
road nodes to each endpoint, compute the shortest drive path, and mark the
edge as 'validated' if a path exists AND its length is less than --max-ratio
times the Euclidean distance between the two endpoints.

Designed to be resumable: a partial JSON checkpoint is written after every
tile is processed to Code/graph_data/osm_validation_partial.json.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "graph_data"
CACHE_DIR = ROOT / ".osm_cache"
CACHE_DIR.mkdir(exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.cache_folder = str(CACHE_DIR)
ox.settings.requests_timeout = 90

OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
ox.settings.overpass_url = OVERPASS_MIRRORS[0]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[osm_validation] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def shortest_path_length_m(graph, orig, dest) -> float | None:
    try:
        route = ox.routing.shortest_path(graph, orig, dest, weight="length")
    except Exception:
        return None
    if not route or len(route) < 2:
        return None
    total_m = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            continue
        best = min(edge_data.values(), key=lambda d: float(d.get("length", math.inf)))
        total_m += float(best.get("length", 0.0))
    return total_m if total_m > 0 else None


def tile_key(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[int, int, int, int]:
    """Group key based on the integer-degree tile containing each endpoint."""
    lat_floor_a = int(math.floor(lat1))
    lon_floor_a = int(math.floor(lon1))
    lat_floor_b = int(math.floor(lat2))
    lon_floor_b = int(math.floor(lon2))
    lat_min = min(lat_floor_a, lat_floor_b)
    lat_max = max(lat_floor_a, lat_floor_b)
    lon_min = min(lon_floor_a, lon_floor_b)
    lon_max = max(lon_floor_a, lon_floor_b)
    return (lat_min, lon_min, lat_max, lon_max)


def tile_bbox(records, buffer_km: float = 3.0) -> tuple[float, float, float, float]:
    lats = []
    lons = []
    for rec in records:
        lats.append(rec["lat1"]); lats.append(rec["lat2"])
        lons.append(rec["lon1"]); lons.append(rec["lon2"])
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    deg_per_km_lat = 1.0 / 111.0
    deg_per_km_lon = 1.0 / (111.0 * max(math.cos(math.radians((lat_min + lat_max) / 2)), 0.1))
    north = lat_max + buffer_km * deg_per_km_lat
    south = lat_min - buffer_km * deg_per_km_lat
    east = lon_max + buffer_km * deg_per_km_lon
    west = lon_min - buffer_km * deg_per_km_lon
    return north, south, east, west


def download_graph_with_retry(west, south, east, north, max_attempts: int = 4):
    """Download an OSM bbox graph with exponential backoff and Overpass mirror fallback."""
    delay = 3.0
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        mirror = OVERPASS_MIRRORS[(attempt - 1) % len(OVERPASS_MIRRORS)]
        ox.settings.overpass_url = mirror
        try:
            return ox.graph_from_bbox((west, south, east, north), network_type="drive", simplify=True)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_recoverable = (
                "max retries" in msg
                or "connectionpool" in msg
                or "429" in msg
                or "timeout" in msg
                or "remote disconnected" in msg
                or "connection aborted" in msg
                or "504" in msg
                or "502" in msg
                or "503" in msg
            )
            if attempt == max_attempts or not is_recoverable:
                raise
            time.sleep(delay)
            delay = min(delay * 1.7, 30.0)
    if last_exc is not None:
        raise last_exc
    return None


def process_tile(gkey: tuple, records: list[dict], buffer_km: float, max_ratio: float) -> tuple[tuple, dict, dict]:
    """Download the OSM graph for this tile and validate all its edges.

    Returns (gkey, results_dict, stats_dict) where:
    - results_dict: {edge_key: validation_result}
    - stats_dict: {'validated': int, 'failed': int, 'download_s': float, 'process_s': float}
    """
    results: dict[str, dict] = {}
    north, south, east, west = tile_bbox(records, buffer_km=buffer_km)
    t0 = time.time()
    try:
        graph = download_graph_with_retry(west, south, east, north, max_attempts=5)
        download_s = time.time() - t0
    except Exception as exc:
        for rec in records:
            results[rec["key"]] = {
                "status": "error",
                "euclidean_km": rec["euclid"],
                "osm_km": None,
                "ratio": None,
                "reason": f"osm_download_failed: {str(exc)[:120]}",
            }
        return gkey, results, {"validated": 0, "failed": len(records), "download_s": time.time() - t0, "process_s": 0.0}

    t1 = time.time()
    validated_in_group = 0
    failed_in_group = 0
    for rec in records:
        try:
            orig = ox.distance.nearest_nodes(graph, rec["lon1"], rec["lat1"])
            dest = ox.distance.nearest_nodes(graph, rec["lon2"], rec["lat2"])
            path_m = shortest_path_length_m(graph, orig, dest)
            if path_m is None:
                results[rec["key"]] = {
                    "status": "failed_no_path",
                    "euclidean_km": rec["euclid"],
                    "osm_km": None,
                    "ratio": None,
                    "reason": "no_drive_path",
                }
                failed_in_group += 1
                continue
            osm_km = path_m / 1000.0
            ratio = osm_km / max(rec["euclid"], 1e-6)
            if ratio <= max_ratio:
                results[rec["key"]] = {
                    "status": "validated",
                    "euclidean_km": rec["euclid"],
                    "osm_km": osm_km,
                    "ratio": ratio,
                    "reason": "ok",
                }
                validated_in_group += 1
            else:
                results[rec["key"]] = {
                    "status": "failed_distance",
                    "euclidean_km": rec["euclid"],
                    "osm_km": osm_km,
                    "ratio": ratio,
                    "reason": f"ratio>{max_ratio}",
                }
                failed_in_group += 1
        except Exception as exc:
            results[rec["key"]] = {
                "status": "error",
                "euclidean_km": rec["euclid"],
                "osm_km": None,
                "ratio": None,
                "reason": str(exc)[:160],
            }
            failed_in_group += 1
    process_s = time.time() - t1
    return gkey, results, {
        "validated": validated_in_group,
        "failed": failed_in_group,
        "download_s": download_s,
        "process_s": process_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ratio", type=float, default=2.0)
    parser.add_argument("--buffer-km", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=2, help="parallel tile workers (default 2; keep low to avoid Overpass API rate-limiting)")
    parser.add_argument("--retry-errors", action="store_true", help="Re-process tiles whose previous result was an osm_download_failed error")
    args = parser.parse_args()

    nodes_df = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    edges_df = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    log(f"Loaded {len(nodes_df)} nodes and {len(edges_df)} edges")

    coord_lookup = (
        nodes_df.assign(node_id=nodes_df["node_id"].astype(str))
        .set_index("node_id")[["latitude", "longitude"]]
        .astype(float)
        .to_dict("index")
    )

    partial_path = GRAPH_DIR / "osm_validation_partial.json"
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        log(f"Resuming from partial checkpoint with {len(partial)} processed edges")
        if args.retry_errors:
            before = len(partial)
            partial = {
                k: v for k, v in partial.items()
                if not (v.get("status") == "error" and "osm_download_failed" in str(v.get("reason", "")))
            }
            log(f"--retry-errors: dropped {before - len(partial)} osm_download_failed entries so they will be retried")
    else:
        partial = {}
    partial_lock = threading.Lock()

    log("Building tile groups...")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    skipped_missing_coords = 0
    same_point = 0
    for _, row in edges_df.iterrows():
        key = f"{row['source']}__{row['target']}__{row.get('edge_type', 'unknown')}"
        if key in partial:
            continue
        src = coord_lookup.get(str(row["source"]))
        dst = coord_lookup.get(str(row["target"]))
        if src is None or dst is None:
            partial[key] = {"status": "error", "reason": "missing_coords"}
            skipped_missing_coords += 1
            continue
        lat1, lon1 = float(src["latitude"]), float(src["longitude"])
        lat2, lon2 = float(dst["latitude"]), float(dst["longitude"])
        euclid = haversine_km(lat1, lon1, lat2, lon2)
        if euclid < 0.01:
            partial[key] = {
                "status": "validated",
                "euclidean_km": euclid,
                "osm_km": 0.0,
                "ratio": 1.0,
                "reason": "same_point",
            }
            same_point += 1
            continue
        record = {
            "key": key,
            "lat1": lat1, "lon1": lon1,
            "lat2": lat2, "lon2": lon2,
            "euclid": euclid,
        }
        groups[tile_key(lat1, lon1, lat2, lon2)].append(record)

    log(f"Skipped missing coords: {skipped_missing_coords}")
    log(f"Skipped same-point edges: {same_point}")
    log(f"Total tile groups to process: {len(groups)}")
    sorted_keys = sorted(groups.keys(), key=lambda k: -len(groups[k]))
    if sorted_keys:
        log(f"Largest group has {len(groups[sorted_keys[0]])} edges; median group has {int(np.median([len(v) for v in groups.values()]))} edges")
    log(f"Starting parallel processing with {args.workers} workers")

    # Persist initial skips immediately
    partial_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    completed_groups = 0
    total_groups = len(groups)
    overall_validated = 0
    overall_failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_tile, gkey, groups[gkey], args.buffer_km, args.max_ratio): gkey
            for gkey in sorted_keys
        }
        for future in as_completed(futures):
            gkey = futures[future]
            try:
                _, group_results, stats = future.result()
            except Exception as exc:
                log(f"Group {gkey} FAILED with worker exception: {str(exc)[:160]}")
                continue
            with partial_lock:
                partial.update(group_results)
                partial_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            completed_groups += 1
            overall_validated += stats["validated"]
            overall_failed += stats["failed"]
            log(
                f"[{completed_groups}/{total_groups}] tile={gkey} "
                f"edges={len(group_results)} validated={stats['validated']} failed={stats['failed']} "
                f"download={stats['download_s']:.1f}s process={stats['process_s']:.1f}s "
                f"(running total: validated={overall_validated} failed={overall_failed})"
            )

    # Final summary
    n_total = len(partial)
    n_validated = sum(1 for v in partial.values() if v.get("status") == "validated")
    n_failed_dist = sum(1 for v in partial.values() if v.get("status") == "failed_distance")
    n_failed_path = sum(1 for v in partial.values() if v.get("status") == "failed_no_path")
    n_error = sum(1 for v in partial.values() if v.get("status") == "error")
    ratios = [v.get("ratio") for v in partial.values() if v.get("ratio") is not None]
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    failed_sample = [{"key": k, **v} for k, v in partial.items() if v.get("status") != "validated"][:50]

    summary = {
        "n_total": n_total,
        "n_validated": n_validated,
        "n_failed_distance": n_failed_dist,
        "n_failed_no_path": n_failed_path,
        "n_error": n_error,
        "coverage_pct": float(100.0 * n_validated / max(n_total, 1)),
        "mean_length_ratio": mean_ratio,
        "median_length_ratio": median_ratio,
        "max_ratio_threshold": args.max_ratio,
        "failed_sample_first50": failed_sample,
    }
    (GRAPH_DIR / "osm_validation_full.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n## OSM validation summary")
    print(f"Total edges: {n_total}")
    print(f"Validated  : {n_validated}  ({summary['coverage_pct']:.1f}%)")
    print(f"Failed (distance ratio > {args.max_ratio}): {n_failed_dist}")
    print(f"Failed (no drive path): {n_failed_path}")
    print(f"Errors: {n_error}")
    print(f"Mean length ratio (osm_km / euclidean_km): {mean_ratio:.3f}")
    print(f"Median length ratio: {median_ratio:.3f}")


if __name__ == "__main__":
    main()
