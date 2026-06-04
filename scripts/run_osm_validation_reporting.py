"""Interpret existing OSM validation outputs by graph variant and edge type.

This script does not rerun any OSM calls. It only reads the existing validation
artifacts and produces more interpretable summaries and concrete good/bad edge
examples for dissertation reporting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRAPH_DIR = ROOT / "graph_data"
REPORT_DIR = ROOT / "reports"

GRAPH_VARIANTS = {
    "spatial": {"spatial"},
    "spatial_route": {"spatial", "same_route"},
    "full_refined": {"spatial", "same_route", "same_functional_class"},
}

GRAPH_VARIANT_LABELS = {
    "spatial": "Spatial only",
    "spatial_route": "Spatial + Route",
    "full_refined": "Full refined",
}

EDGE_TYPE_VARIANTS = {
    "spatial": ["spatial", "spatial_route", "full_refined"],
    "same_route": ["spatial_route", "full_refined"],
    "same_functional_class": ["full_refined"],
}

BAD_STATUSES = {"failed_no_path", "failed_distance", "error"}
BAD_TOPOLOGY = {"unreachable", "long_connected"}
GOOD_STATUSES = {"validated"}
GOOD_TOPOLOGY = {"same_osm_edge", "short_connected", "supported_connected"}


def log(message: str) -> None:
    print(f"[osm_reporting] {message}")


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


def maybe_mean(series: pd.Series) -> float | None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def maybe_median(series: pd.Series) -> float | None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.median())


def load_partial_validation() -> pd.DataFrame:
    """Expand the resumable partial validation JSON into a tidy DataFrame."""

    partial_path = GRAPH_DIR / "osm_validation_partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for key, payload in partial.items():
        try:
            source, target, edge_type = key.split("__")
        except ValueError:
            continue
        rows.append(
            {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "status": payload.get("status"),
                "euclidean_km": safe_float(payload.get("euclidean_km")),
                "osm_path_km_partial": safe_float(payload.get("osm_km")),
                "detour_ratio_partial": safe_float(payload.get("ratio")),
                "failure_reason": payload.get("reason"),
            }
        )
    return pd.DataFrame(rows)


def load_edge_comparison_sample() -> pd.DataFrame:
    """Load the smaller topology-rich comparison sample if it exists."""

    sample_path = GRAPH_DIR / "osm_edge_comparison.csv"
    if not sample_path.exists():
        return pd.DataFrame()
    sample = pd.read_csv(sample_path, low_memory=False)
    key_cols = ["source", "target", "edge_type"]
    for col in key_cols:
        sample[col] = sample[col].astype(str)
    return sample


def build_analysis_table() -> tuple[pd.DataFrame, dict[str, int]]:
    """Join partial validation rows to graph edges and node metadata."""

    edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)
    nodes = pd.read_csv(GRAPH_DIR / "nodes.csv", low_memory=False)
    partial = load_partial_validation()
    sample = load_edge_comparison_sample()

    for frame in (edges, nodes, partial):
        for col in [c for c in ["source", "target", "node_id", "edge_type"] if c in frame.columns]:
            frame[col] = frame[col].astype(str)

    counts = {
        "total_edges_all": int(len(edges)),
        "tested_edges_partial": int(len(partial)),
    }

    merged = partial.merge(
        edges,
        on=["source", "target", "edge_type"],
        how="left",
        suffixes=("", "_graph"),
    )

    if not sample.empty:
        sample_keep = [
            col
            for col in [
                "source",
                "target",
                "edge_type",
                "topology_status",
                "topology_level",
                "osm_path_km",
                "detour_ratio",
                "osm_connected",
                "osm_supported",
                "same_osm_edge",
                "distance_bin",
                "component_rank",
                "component_size",
            ]
            if col in sample.columns
        ]
        merged = merged.merge(sample[sample_keep], on=["source", "target", "edge_type"], how="left")

    source_meta = nodes[
        [
            "node_id",
            "STATE_CODE_EXP",
            "route_key",
            "functional_class",
            "latitude",
            "longitude",
        ]
    ].rename(
        columns={
            "node_id": "source",
            "STATE_CODE_EXP": "source_state_name",
            "route_key": "source_route_key",
            "functional_class": "source_functional_class",
            "latitude": "source_latitude",
            "longitude": "source_longitude",
        }
    )
    target_meta = nodes[
        [
            "node_id",
            "STATE_CODE_EXP",
            "route_key",
            "functional_class",
            "latitude",
            "longitude",
        ]
    ].rename(
        columns={
            "node_id": "target",
            "STATE_CODE_EXP": "target_state_name",
            "route_key": "target_route_key",
            "functional_class": "target_functional_class",
            "latitude": "target_latitude",
            "longitude": "target_longitude",
        }
    )
    merged = merged.merge(source_meta, on="source", how="left")
    merged = merged.merge(target_meta, on="target", how="left")

    merged["graph_variants"] = merged["edge_type"].map(
        lambda edge_type: ", ".join(EDGE_TYPE_VARIANTS.get(str(edge_type), []))
    )
    merged["osm_path_km_final"] = merged["osm_path_km"].where(
        pd.to_numeric(merged.get("osm_path_km"), errors="coerce").notna(),
        merged["osm_path_km_partial"],
    )
    merged["detour_ratio_final"] = merged["detour_ratio"].where(
        pd.to_numeric(merged.get("detour_ratio"), errors="coerce").notna(),
        merged["detour_ratio_partial"],
    )
    merged["graph_distance_km"] = pd.to_numeric(merged.get("distance_km"), errors="coerce")
    return merged, counts


def summarise_variant(table: pd.DataFrame, variant: str, all_edges: pd.DataFrame) -> dict[str, object]:
    """Summarise tested OSM outcomes for one graph variant."""

    variant_types = GRAPH_VARIANTS[variant]
    tested = table[table["edge_type"].isin(variant_types)].copy()
    all_variant_edges = all_edges[all_edges["edge_type"].isin(variant_types)].copy()
    n_total_variant_edges = int(len(all_variant_edges))
    n_tested = int(len(tested))

    def count_status(name: str) -> int:
        return int((tested["status"] == name).sum())

    invalid_snap = int((tested["status"].astype(str) == "invalid_snap").sum())
    if "failure_reason" in tested.columns:
        invalid_snap += int(tested["failure_reason"].astype(str).str.contains("snap", case=False, na=False).sum() - invalid_snap)

    return {
        "graph_variant": variant,
        "graph_variant_label": GRAPH_VARIANT_LABELS[variant],
        "total_variant_edges": n_total_variant_edges,
        "tested_edges": n_tested,
        "tested_share_of_variant": float(100.0 * n_tested / max(n_total_variant_edges, 1)),
        "validated_count": count_status("validated"),
        "validated_share": float((tested["status"] == "validated").mean()) if n_tested else np.nan,
        "failed_no_path_count": count_status("failed_no_path"),
        "failed_no_path_share": float((tested["status"] == "failed_no_path").mean()) if n_tested else np.nan,
        "failed_distance_count": count_status("failed_distance"),
        "failed_distance_share": float((tested["status"] == "failed_distance").mean()) if n_tested else np.nan,
        "invalid_snap_count": invalid_snap,
        "invalid_snap_share": float(invalid_snap / max(n_tested, 1)),
        "mean_graph_distance_km": maybe_mean(tested["graph_distance_km"]),
        "median_graph_distance_km": maybe_median(tested["graph_distance_km"]),
        "mean_osm_path_distance_km": maybe_mean(tested["osm_path_km_final"]),
        "median_osm_path_distance_km": maybe_median(tested["osm_path_km_final"]),
        "mean_detour_ratio": maybe_mean(tested["detour_ratio_final"]),
        "median_detour_ratio": maybe_median(tested["detour_ratio_final"]),
    }


def summarise_edge_type(table: pd.DataFrame, edge_type: str, all_edges: pd.DataFrame) -> dict[str, object]:
    """Summarise tested OSM outcomes for one edge type."""

    tested = table[table["edge_type"] == edge_type].copy()
    all_type_edges = all_edges[all_edges["edge_type"] == edge_type].copy()
    n_total = int(len(all_type_edges))
    n_tested = int(len(tested))
    invalid_snap = int((tested["status"].astype(str) == "invalid_snap").sum())
    return {
        "edge_type": edge_type,
        "graph_variants": ", ".join(EDGE_TYPE_VARIANTS.get(edge_type, [])),
        "total_edges": n_total,
        "tested_edges": n_tested,
        "tested_share_of_type": float(100.0 * n_tested / max(n_total, 1)),
        "validated_share": float((tested["status"] == "validated").mean()) if n_tested else np.nan,
        "failed_no_path_share": float((tested["status"] == "failed_no_path").mean()) if n_tested else np.nan,
        "failed_distance_share": float((tested["status"] == "failed_distance").mean()) if n_tested else np.nan,
        "invalid_snap_share": float(invalid_snap / max(n_tested, 1)),
        "mean_graph_distance_km": maybe_mean(tested["graph_distance_km"]),
        "median_graph_distance_km": maybe_median(tested["graph_distance_km"]),
        "mean_osm_path_distance_km": maybe_mean(tested["osm_path_km_final"]),
        "median_osm_path_distance_km": maybe_median(tested["osm_path_km_final"]),
        "mean_detour_ratio": maybe_mean(tested["detour_ratio_final"]),
        "median_detour_ratio": maybe_median(tested["detour_ratio_final"]),
    }


def explain_bad_edge(row: pd.Series) -> str:
    """Return a short plain-English explanation for a weak graph edge."""

    edge_type = str(row.get("edge_type", ""))
    status = str(row.get("status", ""))
    topology = str(row.get("topology_status", ""))
    distance = safe_float(row.get("graph_distance_km"))
    ratio = safe_float(row.get("detour_ratio_final"))

    if status == "failed_no_path" or topology == "unreachable":
        if edge_type == "same_functional_class":
            return "The sections look similar in context, but OSM found no practical local drivable connection, so this behaves as a similarity edge rather than a physical link."
        return "The graph links these sections, but OSM found no local drivable connection between them, so the edge is weak as a physical road link."
    if status == "failed_distance" or topology == "long_connected":
        if edge_type == "same_route":
            return "A drivable path exists, but it is much longer than the straight-line separation, so the link looks more like broad corridor membership than a local neighbour relation."
        if edge_type == "same_functional_class":
            return "A drivable path exists, but it is too indirect for local road adjacency; the edge is better read as similarity/interdependency than topology."
        ratio_text = f" (detour ratio {ratio:.2f})" if ratio is not None else ""
        return f"The sections are close in the graph but the OSM route is disproportionately indirect{ratio_text}, which weakens the physical-road interpretation."
    if edge_type == "same_functional_class" and distance is not None and distance > 20:
        return "This edge is valid as a same-class similarity link, but its role is contextual rather than physically local."
    return "This graph edge is weak as a local road connection under the available OSM validation evidence."


def explain_good_edge(row: pd.Series) -> str:
    """Return a short plain-English explanation for a strong graph edge."""

    edge_type = str(row.get("edge_type", ""))
    topology = str(row.get("topology_status", ""))
    if topology == "same_osm_edge":
        return "Both LTPP sections snap to the same OSM road edge, which is the strongest possible local topological support."
    if edge_type == "same_route":
        return "These sections are on a clearly connected route corridor in OSM, so the same-route interpretation is well supported."
    if edge_type == "same_functional_class":
        return "Even though this edge was added for similarity, OSM still finds a short drivable connection, so it is both contextually similar and locally plausible."
    return "OSM finds a short, practical drivable connection between the two sections, so the graph edge is physically plausible."


def pick_examples(table: pd.DataFrame, edge_type: str, good: bool, limit: int) -> pd.DataFrame:
    """Select representative good or bad examples for one edge type."""

    subset = table[table["edge_type"] == edge_type].copy()
    if subset.empty:
        return pd.DataFrame()

    if good:
        mask = subset["status"].isin(GOOD_STATUSES)
        if "topology_status" in subset.columns:
            mask = mask | subset["topology_status"].isin(GOOD_TOPOLOGY)
        subset = subset[mask].copy()
        subset["validated_rank"] = np.where(subset["status"].isin(GOOD_STATUSES), 0, 1)
        subset["sort_score"] = pd.to_numeric(subset["detour_ratio_final"], errors="coerce").fillna(99.0)
        subset = subset.sort_values(["validated_rank", "sort_score", "graph_distance_km"], ascending=[True, True, True])
    else:
        mask = subset["status"].isin(BAD_STATUSES)
        if "topology_status" in subset.columns:
            mask = mask | subset["topology_status"].isin(BAD_TOPOLOGY)
        subset = subset[mask].copy()
        subset["severity_rank"] = subset["status"].map({"failed_no_path": 0, "failed_distance": 1, "error": 2}).fillna(3)
        subset["sort_score"] = pd.to_numeric(subset["detour_ratio_final"], errors="coerce").fillna(999.0)
        subset = subset.sort_values(["severity_rank", "sort_score", "graph_distance_km"], ascending=[True, False, False])

    subset = subset.drop_duplicates(subset=["source", "target", "edge_type"]).head(limit).copy()
    if subset.empty:
        return subset

    subset["graph_variants"] = subset["edge_type"].map(lambda x: ", ".join(EDGE_TYPE_VARIANTS.get(str(x), [])))
    subset["state_names"] = subset.apply(
        lambda row: " / ".join(sorted({str(v) for v in [row.get("source_state_name"), row.get("target_state_name")] if pd.notna(v)})),
        axis=1,
    )
    subset["route_fields"] = subset.apply(
        lambda row: " | ".join(sorted({str(v) for v in [row.get("source_route_key"), row.get("target_route_key")] if pd.notna(v) and str(v).strip() and str(v) != "nan"})),
        axis=1,
    )
    subset["functional_class_fields"] = subset.apply(
        lambda row: " | ".join(sorted({str(v) for v in [row.get("source_functional_class"), row.get("target_functional_class")] if pd.notna(v) and str(v).strip() and str(v) != "nan"})),
        axis=1,
    )
    subset["plain_english_explanation"] = subset.apply(
        explain_good_edge if good else explain_bad_edge,
        axis=1,
    )
    keep_cols = [
        "source",
        "target",
        "edge_type",
        "graph_variants",
        "state_names",
        "route_fields",
        "functional_class_fields",
        "graph_distance_km",
        "osm_path_km_final",
        "detour_ratio_final",
        "status",
        "topology_status",
        "failure_reason",
        "source_latitude",
        "source_longitude",
        "target_latitude",
        "target_longitude",
        "plain_english_explanation",
    ]
    return subset[keep_cols].rename(
        columns={
            "source": "source_node_id",
            "target": "target_node_id",
            "graph_distance_km": "straight_line_distance_km",
            "osm_path_km_final": "osm_path_distance_km",
            "detour_ratio_final": "detour_ratio",
        }
    )


def write_interpretation_table() -> pd.DataFrame:
    """Create the edge-type interpretation table requested for the dissertation."""

    table = pd.DataFrame(
        [
            {
                "edge_type": "spatial",
                "meaning_in_my_graph": "Local geographic proximity between sections.",
                "should_it_be_osm_routable": "Often yes, especially at short distance.",
                "how_to_interpret_osm_failure": "Straight-line proximity does not necessarily imply a practical drivable connection; the edge may cross barriers or depend on indirect roads.",
            },
            {
                "edge_type": "same_route",
                "meaning_in_my_graph": "Local same-corridor relationship along the same route.",
                "should_it_be_osm_routable": "Usually yes.",
                "how_to_interpret_osm_failure": "More concerning than for spatial edges; inspect whether the pair is actually too far apart or whether the route label is over-broad.",
            },
            {
                "edge_type": "same_functional_class",
                "meaning_in_my_graph": "Same state, same functional class, and similar traffic/climate/pavement context.",
                "should_it_be_osm_routable": "Not necessarily.",
                "how_to_interpret_osm_failure": "Does not invalidate the edge; it means the edge is functioning as a similarity/interdependency link rather than a physical adjacency link.",
            },
        ]
    )
    table.to_csv(REPORT_DIR / "osm_edge_type_interpretation.csv", index=False)
    return table


def build_summary_markdown(
    variant_df: pd.DataFrame,
    edge_type_df: pd.DataFrame,
    bad_examples: pd.DataFrame,
    good_examples: pd.DataFrame,
    coverage_note: str,
) -> str:
    """Create a dissertation-ready markdown interpretation."""

    spatial_row = edge_type_df[edge_type_df["edge_type"] == "spatial"].iloc[0]
    route_row = edge_type_df[edge_type_df["edge_type"] == "same_route"].iloc[0]
    class_row = edge_type_df[edge_type_df["edge_type"] == "same_functional_class"].iloc[0]
    physically_road_like = "spatial_route"
    interdependency_variant = "full_refined"

    def format_example(df: pd.DataFrame, edge_type: str, n: int) -> list[str]:
        rows = df[df["edge_type"] == edge_type].head(n)
        lines = []
        for row in rows.itertuples(index=False):
            ratio = getattr(row, "detour_ratio", None)
            ratio_text = f", detour ratio {ratio:.2f}" if ratio is not None and pd.notna(ratio) else ""
            lines.append(
                f"- `{row.source_node_id}` ↔ `{row.target_node_id}` ({edge_type}, {row.status}): "
                f"{row.plain_english_explanation} Straight-line distance {row.straight_line_distance_km:.2f} km{ratio_text}."
            )
        return lines

    lines = [
        "# OSM Validation Interpretation",
        "",
        "## Coverage and scope",
        "",
        coverage_note,
        "",
        "## Main interpretation",
        "",
        f"- **Most physically road-like graph variant**: `{physically_road_like}` ({GRAPH_VARIANT_LABELS[physically_road_like]}). It keeps the local spatial layer and same-route corridor links without adding the broader same-class similarity layer.",
        f"- **Best interpreted as an interdependency graph**: `{interdependency_variant}` ({GRAPH_VARIANT_LABELS[interdependency_variant]}). Its `same_functional_class` edges are context/similarity links, so they should not be judged only as physical road adjacency.",
        f"- **Are spatial edges mostly valid road connections?** Yes, but not universally. The validated share in the tested OSM checkpoint is {spatial_row['validated_share']:.1%}, which still leaves a meaningful tail of no-path and long-detour cases.",
        f"- **Are same_route edges more strongly validated than spatial edges?** No in the current stored checkpoint. The tested validated share is {route_row['validated_share']:.1%} for `same_route` versus {spatial_row['validated_share']:.1%} for `spatial`, so route links should be interpreted cautiously rather than assumed superior by default.",
        f"- **Are full_refined similarity edges expected to fail OSM validation?** Often yes, and that is not necessarily a problem. `same_functional_class` edges validate at {class_row['validated_share']:.1%}, but they were designed as same-state similarity/interdependency links rather than guaranteed routable adjacency.",
        "",
        "## Concrete bad-edge examples",
        "",
        *format_example(bad_examples, "spatial", 2),
        *format_example(bad_examples, "same_route", 2),
        *format_example(bad_examples, "same_functional_class", 2),
        "",
        "## Concrete good-edge examples",
        "",
        *format_example(good_examples, "spatial", 2),
        *format_example(good_examples, "same_route", 2),
        *format_example(good_examples, "same_functional_class", 2),
        "",
        "## Recommended use in the dissertation",
        "",
        "- **Should the graph be presented as a true road network?** No. It should not be presented as a fully routable national road network.",
        "- **Should it be presented as a section-level interdependency graph?** Yes. That framing is both more accurate and more defensible.",
        "- **Which graph should be used for shortest-path disruption calculations?** Prefer the spatial + same-route view, because those edges are the most interpretable as physical road-like connectivity.",
        "- **Which graph should be used for broader deterioration / treatment-context analysis?** The full refined graph remains appropriate, because similarity and contextual interdependence matter even when the links are not locally routable in OSM.",
        "- **Should OSM validation be treated as full validation?** No. It is best used as a local sanity check that helps distinguish physically road-like edges from broader similarity/interdependency edges.",
    ]
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    merged, counts = build_analysis_table()
    all_edges = pd.read_csv(GRAPH_DIR / "edges.csv", low_memory=False)

    variant_rows = [summarise_variant(merged, variant, all_edges) for variant in GRAPH_VARIANTS]
    variant_df = pd.DataFrame(variant_rows)
    variant_df.to_csv(REPORT_DIR / "osm_validation_by_graph_variant.csv", index=False)

    edge_type_rows = [summarise_edge_type(merged, edge_type, all_edges) for edge_type in ["spatial", "same_route", "same_functional_class"]]
    edge_type_df = pd.DataFrame(edge_type_rows)
    edge_type_df.to_csv(REPORT_DIR / "osm_validation_by_edge_type.csv", index=False)

    bad_parts: list[pd.DataFrame] = []
    good_parts: list[pd.DataFrame] = []
    for edge_type in ["spatial", "same_route", "same_functional_class"]:
        bad = pick_examples(merged, edge_type, good=False, limit=5)
        if not bad.empty:
            bad_parts.append(bad)
        good = pick_examples(merged, edge_type, good=True, limit=5)
        if not good.empty:
            good_parts.append(good)

    bad_examples = pd.concat(bad_parts, ignore_index=True) if bad_parts else pd.DataFrame()
    good_examples = pd.concat(good_parts, ignore_index=True) if good_parts else pd.DataFrame()
    bad_examples.to_csv(REPORT_DIR / "osm_bad_edge_examples.csv", index=False)
    good_examples.to_csv(REPORT_DIR / "osm_good_edge_examples.csv", index=False)

    interpretation_table = write_interpretation_table()

    full_refined_tested_share = variant_df.loc[variant_df["graph_variant"] == "full_refined", "tested_share_of_variant"].iloc[0]
    same_class_tested = edge_type_df.loc[edge_type_df["edge_type"] == "same_functional_class", "tested_edges"].iloc[0]
    coverage_note = (
        f"The existing resumable OSM checkpoint already covers all three edge types and therefore all three graph variants. "
        f"It tests {counts['tested_edges_partial']:,} edges overall, including {same_class_tested:,} `same_functional_class` edges. "
        f"That means the current OSM evidence does reach into `full_refined`, although only {full_refined_tested_share:.1f}% "
        f"of that graph's edges are currently covered by the stored checkpoint."
    )

    route_fail_count = int(
        ((merged["edge_type"] == "same_route") & (merged["status"].isin(["failed_no_path", "failed_distance"]))).sum()
    )
    if route_fail_count == 0:
        route_note = "No failed same_route examples were found in the tested checkpoint; same_route edges were strongly supported in the available OSM sample."
    else:
        route_note = f"The tested checkpoint contains {route_fail_count} failed same_route edges, so corridor links should still be inspected rather than assumed correct by default."

    summary_json = {
        "coverage_note": coverage_note,
        "route_note": route_note,
        "variant_summary": variant_rows,
        "edge_type_summary": edge_type_rows,
        "recommended_conclusions": {
            "present_as_true_road_network": False,
            "present_as_section_level_interdependency_graph": True,
            "best_variant_for_disruption": "spatial_route",
            "best_variant_for_deterioration_context": "full_refined",
            "osm_validation_role": "local sanity check, not full graph validation",
        },
    }
    (REPORT_DIR / "osm_validation_interpretation.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    markdown = build_summary_markdown(variant_df, edge_type_df, bad_examples, good_examples, coverage_note)
    markdown += "\n\n" + route_note + "\n"
    (REPORT_DIR / "osm_validation_interpretation.md").write_text(markdown, encoding="utf-8")

    log(
        f"Variant summary written for {len(variant_df)} graph variants; "
        f"edge-type summary written for {len(edge_type_df)} edge types."
    )
    log(
        f"Bad examples: {len(bad_examples)} rows; good examples: {len(good_examples)} rows."
    )
    print("\nTop-level conclusions")
    print(f"- Physically road-like variant: spatial_route")
    print(f"- Interdependency variant: full_refined")
    print(f"- Spatial validated share: {edge_type_df.loc[edge_type_df['edge_type']=='spatial', 'validated_share'].iloc[0]:.1%}")
    print(f"- Same-route validated share: {edge_type_df.loc[edge_type_df['edge_type']=='same_route', 'validated_share'].iloc[0]:.1%}")
    print(f"- Same-class validated share: {edge_type_df.loc[edge_type_df['edge_type']=='same_functional_class', 'validated_share'].iloc[0]:.1%}")
    print(f"- {route_note}")


if __name__ == "__main__":
    main()
