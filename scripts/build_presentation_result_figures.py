from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "presentation"

COLORS = {
    "baseline": "#4C78A8",
    "graph": "#7A4DA3",
    "refined": "#2CA25F",
    "limit": "#E45756",
    "neutral": "#9E9E9E",
}


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str, *, tight_bbox: bool = True) -> None:
    fig.tight_layout()
    save_kwargs = {"dpi": 220}
    if tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(OUT_DIR / name, **save_kwargs)
    plt.close(fig)


def add_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str, ec: str = "#333333", fontsize: int = 11, weight: str = "normal") -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, wrap=True)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#444444") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color)
    ax.add_patch(arrow)


def plot_graph_construction() -> None:
    df = pd.read_csv(ROOT / "reports" / "part1_rgcn_temporal.csv")
    label_map = {
        "spatial": "Spatial only",
        "spatial_route": "Spatial + route",
        "full_refined": "Full refined",
    }
    df["label"] = df["graph_variant"].map(label_map)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        df["label"],
        df["test_r2"],
        color=[COLORS["baseline"], COLORS["graph"], COLORS["refined"]],
    )
    ax.set_ylabel("Test R²")
    ax.set_title("Graph construction changes graph-model performance")
    ax.set_ylim(0, max(df["test_r2"]) + 0.08)
    style_axes(ax)
    for bar, val in zip(bars, df["test_r2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save(fig, "graph_construction_comparison.png")


def plot_cracking_benchmark() -> None:
    ensemble = load_json("graph_data/ensemble_results.json")
    hpms_sweep = load_json("reports/materials_weight_sweep_hpms16.json")
    mepdg = pd.read_csv(ROOT / "reports" / "mepdg_benchmark.csv")

    hpms = {
        "Local baseline\n(Random Forest)": ensemble["results"]["RF local"]["test"]["r2"],
        "Basic graph model\n(R-GCN)": ensemble["results"]["R-GCN"]["test"]["r2"],
        "Best refined graph\n(R-GCN)": hpms_sweep["best_test_r2"],
        "Stacked ensemble\n(RF + R-GCN)": ensemble["results"]["Stacked MLP"]["test"]["r2"],
    }
    mepdg_map = {row["model"]: row["r2_test"] for _, row in mepdg.iterrows()}
    mepdg_vals = {
        "Local baseline\n(Random Forest)": mepdg_map["RF local"],
        "Basic graph model\n(R-GCN)": mepdg_map["R-GCN baseline"],
        "Best refined graph\n(R-GCN)": mepdg_map["R-GCN best materials sweep (climate_pavement)"],
        "Stacked ensemble\n(RF + R-GCN)": mepdg_map["Stacked MLP ensemble"],
    }

    labels = list(hpms.keys())
    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    hpms_bars = ax.bar(
        [i - width / 2 for i in x],
        list(hpms.values()),
        width=width,
        label="HPMS-style cracking %",
        color="#5B8FF9",
    )
    mepdg_bars = ax.bar(
        [i + width / 2 for i in x],
        list(mepdg_vals.values()),
        width=width,
        label="Mechanistic-empirical cracking %",
        color="#61DDAA",
    )
    ax.set_ylabel("Test R²")
    ax.set_title("Cracking prediction: local baseline vs graph-aware models")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 0.66)
    style_axes(ax)
    for bars in (hpms_bars, mepdg_bars):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.012,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    save(fig, "cracking_prediction_comparison.png")


def plot_distress_targets() -> None:
    preferred = ROOT / "graph_data" / "singletask_better_coverage_targets.json"
    fallback = ROOT / "graph_data" / "singletask_per_distress_results.json"
    data = json.loads(preferred.read_text()) if preferred.exists() else load_json(str(fallback.relative_to(ROOT)))
    rows = []
    label_map = {
        "HPMS16_CRACKING_PERCENT_AC": "HPMS-style cracking %",
        "MEPDG_CRACKING_PERCENT_AC": "Mechanistic-empirical cracking %",
        "ME_PERCENT_WHEEL_PATH_CRACK": "Wheel-path cracking %",
        "LONGIGATOR_CRACKING": "Longigator cracking area",
        "MEPDG_TRANS_CRACK_LENGTH_AC": "Transverse cracking length",
        "PATCH_A": "Patching",
        "POTHOLES_A": "Potholes",
    }
    for row in data["results"]:
        rows.append(
            {
                "label": label_map.get(row["target"], row["target"]),
                "r2_test": row["r2_test"],
            }
        )
    df = pd.DataFrame(rows).sort_values("r2_test", ascending=False)
    colors = [
        COLORS["refined"] if val > 0.3 else COLORS["limit"] if val < 0.05 else COLORS["graph"]
        for val in df["r2_test"]
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(df["label"], df["r2_test"], color=colors)
    ax.set_xlabel("Test R²")
    ax.set_title("The graph performs best on well-covered cracking-style targets")
    ax.axvline(0, color="#666666", linewidth=1)
    style_axes(ax)
    for bar in bars:
        val = bar.get_width()
        ax.text(
            val + (0.012 if val >= 0 else -0.06),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )
    save(fig, "distress_target_performance.png")


def plot_ood_limit() -> None:
    summary = load_json("reports/part1_ood_ensemble_summary.json")
    labels = [
        "Local baseline\n(Random Forest)",
        "Graph model\n(R-GCN)",
        "Stacked ensemble",
    ]
    values = [
        summary["rf_r2"]["mean"],
        summary["rgcn_r2"]["mean"],
        summary["ensemble_r2"]["mean"],
    ]
    colors = [COLORS["baseline"], COLORS["graph"], COLORS["limit"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean held-out-state R²")
    ax.set_title("Unseen-state transfer remains weak")
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_ylim(min(values) - 0.08, max(values) + 0.1)
    style_axes(ax)
    for bar, val in zip(bars, values):
        va = "bottom" if val >= 0 else "top"
        offset = 0.012 if val >= 0 else -0.012
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=10,
        )
    save(fig, "ood_generalisation_limit.png")


def plot_proxy_ranking() -> None:
    df = pd.read_csv(ROOT / "graph_data" / "network_node_impacts_full_refined.csv")
    top = df.nlargest(10, "delta_vht_proxy").copy()
    top["delta_vht_billions"] = top["delta_vht_proxy"] / 1e9
    top = top.sort_values("delta_vht_billions")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.barh(top["node_id"], top["delta_vht_billions"], color=COLORS["graph"])
    ax.set_xlabel("Approximate added travel burden (proxy, billions)")
    ax.set_title("Most disruptive single-section closures by proxy score")
    style_axes(ax)
    for bar in bars:
        val = bar.get_width()
        ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
    save(fig, "proxy_ranking.png")


def plot_full_refined_rgcn_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "How the full-refined relation-aware GCN works", fontsize=18, weight="bold", ha="left")
    ax.text(
        0.03,
        0.91,
        "Prediction target used in the main benchmark: future HPMS-style cracking percentage for each road section",
        fontsize=11,
        ha="left",
        color="#444444",
    )

    # Column 1: inputs
    add_box(
        ax,
        (0.03, 0.63),
        (0.20, 0.18),
        "1. Node features\n\nEach road section has:\ncurrent cracking,\ntraffic,\nclimate,\npavement/materials,\nmaintenance history",
        "#E8F1FB",
        fontsize=11,
        weight="bold",
    )
    add_box(
        ax,
        (0.03, 0.27),
        (0.20, 0.25),
        "2. Three relation types\n\nSpatial\n10,186 links\n\nSame route\n1,779 links\n\nSame functional class\n5,979 scored links",
        "#F3E8FB",
        fontsize=11,
        weight="bold",
    )

    # Column 2: graph construction
    add_box(
        ax,
        (0.30, 0.66),
        (0.22, 0.12),
        "Spatial relation\nNearest local neighbours\nwithin 80 km",
        "#D8E8F8",
        fontsize=11,
    )
    add_box(
        ax,
        (0.30, 0.49),
        (0.22, 0.12),
        "Corridor relation\nConsecutive sections\non the same route",
        "#E4D7F3",
        fontsize=11,
    )
    add_box(
        ax,
        (0.30, 0.26),
        (0.22, 0.18),
        "Refined similarity relation\nWithin functional class,\nscore candidates by:\nspatial + traffic + climate + pavement\nthen keep top 5 local neighbours",
        "#DFF2E5",
        fontsize=10,
    )

    # Column 3: relation matrices / message passing
    add_box(
        ax,
        (0.60, 0.62),
        (0.17, 0.18),
        "3. Build one weighted adjacency\nmatrix per relation type\n\nA_spatial\nA_route\nA_similarity",
        "#FFF3D6",
        fontsize=11,
        weight="bold",
    )
    add_box(
        ax,
        (0.60, 0.30),
        (0.17, 0.20),
        "4. Relation-aware message passing\n\nThe model learns a separate\nlinear transform for each relation\nand combines them with\nself-node information",
        "#FFE1D8",
        fontsize=10,
        weight="bold",
    )

    # Column 4: output
    add_box(
        ax,
        (0.83, 0.46),
        (0.14, 0.20),
        "5. Output\n\nPredicted future cracking\nfor each road section\n\nExample benchmark:\nHPMS16_CRACKING_PERCENT_AC",
        "#E5F6E8",
        fontsize=11,
        weight="bold",
    )

    # Arrows
    add_arrow(ax, (0.23, 0.72), (0.30, 0.72))
    add_arrow(ax, (0.23, 0.40), (0.30, 0.40))
    add_arrow(ax, (0.52, 0.72), (0.60, 0.72))
    add_arrow(ax, (0.52, 0.55), (0.60, 0.70))
    add_arrow(ax, (0.52, 0.35), (0.60, 0.39))
    add_arrow(ax, (0.77, 0.70), (0.83, 0.58))
    add_arrow(ax, (0.77, 0.40), (0.83, 0.54))

    # Legend / note
    ax.text(
        0.03,
        0.07,
        "Key idea: the model does not treat full_refined as one flat network. It keeps three relation channels, "
        "so a road section can learn differently from nearby sections, same-route sections, and scored local similarity links.",
        fontsize=10.5,
        ha="left",
        color="#333333",
        wrap=True,
    )
    ax.text(
        0.03,
        0.03,
        "Important nuance: the refined similarity relation already includes a spatial term, so the relation families overlap rather than being perfectly independent.",
        fontsize=10,
        ha="left",
        color="#666666",
        wrap=True,
    )
    save(fig, "full_refined_rgcn_pipeline.png")


def plot_osm_validation_backup() -> None:
    by_variant = pd.read_csv(ROOT / "reports" / "osm_validation_by_graph_variant.csv")
    by_type = pd.read_csv(ROOT / "reports" / "osm_validation_by_edge_type.csv")
    findings = load_json("graph_data/osm_validation_findings.json")
    interpretation = load_json("reports/osm_validation_interpretation.json")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})

    # Panel 1: tested share and validated share by graph variant
    panel = by_variant.copy()
    x = np.arange(len(panel))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        panel["tested_share_of_variant"],
        width=width,
        color=COLORS["baseline"],
        label="Tested by OSM checkpoint",
    )
    axes[0].bar(
        x + width / 2,
        panel["validated_share"],
        width=width,
        color=COLORS["refined"],
        label="Strictly validated",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(panel["graph_variant_label"])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Share of graph edges")
    axes[0].set_title("The overnight checkpoint covered most graph variants")
    style_axes(axes[0])
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    for idx, row in panel.iterrows():
        axes[0].text(
            idx - width / 2,
            row["tested_share_of_variant"] + 0.03,
            f"{100 * row['tested_share_of_variant']:.1f}%",
            ha="center",
            fontsize=9,
        )
        axes[0].text(
            idx + width / 2,
            row["validated_share"] + 0.03,
            f"{100 * row['validated_share']:.1f}%",
            ha="center",
            fontsize=9,
        )
        axes[0].text(
            idx,
            0.05,
            f"{int(row['tested_edges'])}/{int(row['total_variant_edges'])}",
            ha="center",
            fontsize=8,
            color="#555555",
        )

    # Panel 2: validated share by edge type
    edge_panel = by_type.copy()
    edge_panel["edge_type_label"] = edge_panel["edge_type"].replace(
        {
            "spatial": "Spatial",
            "same_route": "Same-route",
            "same_functional_class": "Similarity",
        }
    )
    x2 = np.arange(len(edge_panel))
    bars = axes[1].bar(
        x2,
        edge_panel["validated_share"],
        color=[COLORS["graph"], COLORS["baseline"], COLORS["refined"]],
        width=0.58,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Strictly validated share")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(edge_panel["edge_type_label"])
    axes[1].set_title("Similarity edges are expected to validate less often")
    style_axes(axes[1])
    for bar, (_, row) in zip(bars, edge_panel.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            row["validated_share"] + 0.03,
            f"{100 * row['validated_share']:.1f}%",
            ha="center",
            fontsize=9,
        )
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            0.04,
            f"tested {int(row['tested_edges'])}/{int(row['total_edges'])}",
            ha="center",
            fontsize=8,
            color="#555555",
        )
    axes[1].text(
        0.02,
        0.98,
        (
            f"Spatial+route headline: {findings['spatial_route_subset']['status_pct']['validated']:.1f}% validated\n"
            f"Full refined tested share: {interpretation['variant_summary'][2]['tested_share_of_variant']:.1f}%"
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f5f6fa", "edgecolor": "#dddddd"},
    )

    fig.suptitle(
        (
            "OpenStreetMap validation from the resumable overnight checkpoint "
            f"(13,267 tested edges; {interpretation['variant_summary'][2]['tested_share_of_variant']:.1f}% of full refined)"
        ),
        fontsize=13,
        y=1.03,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.28)
    save(fig, "osm_validation_backup.png", tight_bbox=False)


def main() -> None:
    ensure_out_dir()
    plot_graph_construction()
    plot_cracking_benchmark()
    plot_distress_targets()
    plot_ood_limit()
    plot_proxy_ranking()
    plot_full_refined_rgcn_pipeline()
    plot_osm_validation_backup()


if __name__ == "__main__":
    main()
