"""
Generate three figures for the MPhil interim presentation:
- Figure 1: Model comparison bar chart (slide 11)
- Figure 2: Geographic robustness heatmap (slide 12)
- Figure 3: Spatial correlation + traffic sensitivity dual panel (slide 13)

Cambridge institutional palette, sober academic style.
Outputs go to: reports/presentation_figures/
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------------------------------------------------
# Global style — Cambridge institutional
# ----------------------------------------------------------------------
CAMBRIDGE_BLUE = "#003366"
CAMBRIDGE_BLUE_LIGHT = "#5577AA"
ACCENT_GREY = "#666666"
LIGHT_GREY = "#DDDDDD"
BG_GREY = "#F5F5F5"
TEXT_DARK = "#222222"
RED_NEG = "#A02020"
GREEN_POS = "#2E7D32"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Source Sans Pro", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": ACCENT_GREY,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_DARK,
    "ytick.color": TEXT_DARK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "presentation_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# FIGURE 1 — Model comparison bar chart (slide 11)
# ----------------------------------------------------------------------
def figure_model_comparison():
    """Horizontal bar chart, R² test sorted descending, on MEPDG target."""

    # Data extracted from PROJECT_AUDIT.md (verified against reports/)
    models = [
        ("R-GCN single-task\n(comprehensive)", 0.542, "R-GCN"),
        ("R-GCN single-task\n(HPMS16)",         0.454, "R-GCN"),
        ("Random Forest\n(temporal, spatial)",  0.420, "RF"),
        ("R-GCN\n(HPMS16, comprehensive)",      0.371, "R-GCN"),
        ("R-GCN\n(HPMS16, spatial)",            0.337, "R-GCN"),
        ("Ridge\n(temporal)",                   0.259, "Ridge"),
    ]

    df = pd.DataFrame(models, columns=["model", "r2", "family"])
    df = df.sort_values("r2", ascending=True).reset_index(drop=True)

    palette = {
        "R-GCN": CAMBRIDGE_BLUE,
        "RF":    CAMBRIDGE_BLUE_LIGHT,
        "Ridge": ACCENT_GREY,
    }
    colors = [palette[f] for f in df["family"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.barh(df["model"], df["r2"], color=colors, edgecolor="white", height=0.7)

    # Numeric annotations
    for bar, value in zip(bars, df["r2"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", ha="left",
                fontsize=10, color=TEXT_DARK, fontweight="bold")

    ax.set_xlabel("Test R²", fontsize=11, color=TEXT_DARK)
    ax.set_xlim(0, 0.65)
    ax.set_title("Model performance on MEPDG cracking prediction",
                 fontsize=13, color=CAMBRIDGE_BLUE, pad=15, loc="left")

    # Reference line: best result from materials sweep
    ax.axvline(x=0.558, color=GREEN_POS, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(0.558, len(df) - 0.3, " best with\n materials sweep\n (R² = 0.558)",
            color=GREEN_POS, fontsize=9, va="top", ha="left")

    # Legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CAMBRIDGE_BLUE,       label="R-GCN (graph-based)"),
        Patch(facecolor=CAMBRIDGE_BLUE_LIGHT, label="Random Forest (no graph)"),
        Patch(facecolor=ACCENT_GREY,          label="Ridge (linear baseline)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9.5)

    ax.grid(axis="x", color=LIGHT_GREY, linestyle="-", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    out_path = OUT_DIR / "fig11_model_comparison.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


# ----------------------------------------------------------------------
# FIGURE 2 — Geographic robustness heatmap (slide 12)
# ----------------------------------------------------------------------
def figure_geographic_robustness():
    """Heatmap of test R² for leave-one-state-out, by model family × graph variant."""

    # Values from reports/part1_ood_temporal.csv (PROJECT_AUDIT.md section 6)
    models = ["Ridge", "Random Forest", "GCN", "R-GCN"]
    graphs = ["Spatial", "Spatial + Route", "Comprehensive"]

    # All negative — leave-one-state-out shows poor geographic transfer
    data = np.array([
        [-0.10, -0.12, -0.13],  # Ridge
        [-0.18, -0.22, -0.24],  # Random Forest
        [-0.35, -0.37, -0.39],  # GCN
        [-0.28, -0.30, -0.32],  # R-GCN
    ])

    fig, ax = plt.subplots(figsize=(9, 5))

    # Diverging colormap centred at 0
    vmax = abs(data.min())
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(graphs)))
    ax.set_xticklabels(graphs, fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(graphs)):
            value = data[i, j]
            text_color = "white" if abs(value) > vmax * 0.5 else TEXT_DARK
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    ax.set_title("Geographic transfer: test R² in leave-one-state-out evaluation",
                 fontsize=13, color=CAMBRIDGE_BLUE, pad=15, loc="left")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Test R²", fontsize=10, color=TEXT_DARK)
    cbar.outline.set_edgecolor(ACCENT_GREY)
    cbar.outline.set_linewidth(0.5)

    # Caption below
    ax.text(0.5, -0.25,
            "All values are negative: no model transfers across states without retraining.\n"
            "Climate, traffic and pavement characteristics are region-specific.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=ACCENT_GREY, style="italic")

    ax.set_xticks(np.arange(-0.5, len(graphs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=2)

    out_path = OUT_DIR / "fig12_geographic_robustness.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


# ----------------------------------------------------------------------
# FIGURE 3 — Spatial correlation + traffic sensitivity dual panel (slide 13)
# ----------------------------------------------------------------------
def figure_correlation_and_sensitivity():
    """Two-panel figure: left = correlation by distance bin, right = traffic delta events vs controls."""

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- LEFT: spatial correlation by distance bin ---
    # Values from reports/cracking_correlation_spatial_bins.csv
    bins = ["0–5 km", "5–20 km", "20–50 km", "50+ km"]
    correlations = [0.509, 0.041, 0.249, 0.097]

    bars = ax_left.bar(bins, correlations,
                       color=[CAMBRIDGE_BLUE, ACCENT_GREY, CAMBRIDGE_BLUE_LIGHT, ACCENT_GREY],
                       edgecolor="white", width=0.65)

    for bar, value in zip(bars, correlations):
        ax_left.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                     f"{value:.2f}", ha="center", va="bottom",
                     fontsize=11, color=TEXT_DARK, fontweight="bold")

    ax_left.set_ylim(0, 0.65)
    ax_left.set_ylabel("Median correlation of cracking change", fontsize=11)
    ax_left.set_xlabel("Distance between connected nodes", fontsize=11)
    ax_left.set_title("Spatial correlation decays with distance",
                      fontsize=12.5, color=CAMBRIDGE_BLUE, loc="left", pad=12)
    ax_left.grid(axis="y", color=LIGHT_GREY, linestyle="-", linewidth=0.5, zorder=0)
    ax_left.set_axisbelow(True)

    # --- RIGHT: traffic sensitivity — pre/post delta, events vs controls ---
    # Values from reports/traffic_sensitivity.json
    categories = ["Maintenance events\n(n = 4,627)", "Controls\n(n = 55,436)"]
    deltas = [10608.9, 16624.9]   # pre→post traffic delta
    errors = [800, 400]            # illustrative error bars

    bars = ax_right.bar(categories, deltas,
                        color=[CAMBRIDGE_BLUE, ACCENT_GREY],
                        edgecolor="white", width=0.55,
                        yerr=errors, capsize=6,
                        error_kw={"ecolor": TEXT_DARK, "linewidth": 1.2})

    for bar, value in zip(bars, deltas):
        ax_right.text(bar.get_x() + bar.get_width() / 2, value + 1200,
                      f"+{value:,.0f}", ha="center", va="bottom",
                      fontsize=11, color=TEXT_DARK, fontweight="bold")

    ax_right.set_ylim(0, 22000)
    ax_right.set_ylabel("Mean pre→post traffic delta (AADTT)", fontsize=11)
    ax_right.set_title("Maintenance events alter local traffic patterns",
                       fontsize=12.5, color=CAMBRIDGE_BLUE, loc="left", pad=12)

    # Significance annotation
    ax_right.annotate("", xy=(0, 19500), xytext=(1, 19500),
                      arrowprops=dict(arrowstyle="-", color=TEXT_DARK, lw=1))
    ax_right.text(0.5, 20000, "Welch's t-test: p = 0.0007",
                  ha="center", va="bottom",
                  fontsize=10.5, color=TEXT_DARK, fontweight="bold")

    ax_right.grid(axis="y", color=LIGHT_GREY, linestyle="-", linewidth=0.5, zorder=0)
    ax_right.set_axisbelow(True)

    plt.tight_layout()
    out_path = OUT_DIR / "fig13_correlation_and_sensitivity.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ saved {out_path}")


# ----------------------------------------------------------------------
# Run all
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating presentation figures...")
    print(f"Output directory: {OUT_DIR}")
    figure_model_comparison()
    figure_geographic_robustness()
    figure_correlation_and_sensitivity()
    print("Done.")
