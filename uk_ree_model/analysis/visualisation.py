"""
visualisation.py
================
Charting module for the UK REE DIO–CGE–ABM model.

All functions return matplotlib Figure objects (or save to file).
Designed for use in Jupyter notebooks or batch export.

Chart catalogue:
  1.  plot_ree_price_path        — REE price over time by scenario
  2.  plot_output_loss_path      — UK total output loss (£bn) over time
  3.  plot_sector_heatmap        — Sector output change heatmap (DIO)
  4.  plot_leontief_multipliers  — Backward linkage bar chart
  5.  plot_ree_dependence        — Direct vs total REE dependence scatter
  6.  plot_ghosh_waterfall       — Waterfall chart of supply shock propagation
  7.  plot_inventory_depletion   — Mean inventory months over time
  8.  plot_substitution_s_curve  — Cumulative substitution adoption curve
  9.  plot_cge_welfare           — EV welfare loss by θ (bar chart)
  10. plot_scenario_comparison   — Multi-KPI radar/spider chart
  11. plot_network_risk          — Supply chain network risk map
  12. plot_mc_uncertainty        — Monte Carlo uncertainty fans
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from typing import Optional

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

SCENARIO_COLOURS = {
    "A: Moderate": "#2196F3",
    "B: Severe": "#FF5722",
    "C: Sustained": "#FF9800",
    "D: Complete": "#9C27B0",
    "E: Net Zero Demand": "#4CAF50",
}
DEFAULT_COLOUR = "#607D8B"


# ---------------------------------------------------------------------------
# 1. REE price path
# ---------------------------------------------------------------------------

def plot_ree_price_path(
    abm_dfs: dict[str, pd.DataFrame],
    title: str = "REE Import Price (Index, Base=1.0)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in abm_dfs.items():
        colour = SCENARIO_COLOURS.get(label, DEFAULT_COLOUR)
        ax.plot(df["step"], df["ree_price"], label=label, color=colour, lw=2)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Base price")
    ax.set_xlabel("Period (months)")
    ax.set_ylabel("REE price index")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Output loss path
# ---------------------------------------------------------------------------

def plot_output_loss_path(
    abm_dfs: dict[str, pd.DataFrame],
    title: str = "UK Manufacturing Output Loss (£bn)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute loss
    ax = axes[0]
    for label, df in abm_dfs.items():
        colour = SCENARIO_COLOURS.get(label, DEFAULT_COLOUR)
        ax.plot(df["step"], df["output_loss_£bn"], label=label, color=colour, lw=2)
    ax.set_xlabel("Period (months)")
    ax.set_ylabel("Output loss (£bn)")
    ax.set_title("Absolute output loss")
    ax.legend(fontsize=9)

    # Right: % loss
    ax = axes[1]
    for label, df in abm_dfs.items():
        colour = SCENARIO_COLOURS.get(label, DEFAULT_COLOUR)
        ax.plot(df["step"], df["output_loss_pct"], label=label, color=colour, lw=2)
    ax.set_xlabel("Period (months)")
    ax.set_ylabel("Output loss (%)")
    ax.set_title("Percentage output loss")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Sector output heatmap (DIO)
# ---------------------------------------------------------------------------

def plot_sector_heatmap(
    multi_theta_df: pd.DataFrame,
    title: str = "Sector Output Change by Shock Severity (%)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    pivot = multi_theta_df.pivot(index="sector", columns="theta", values="delta_x_pct")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=-30, vmax=5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"θ={t:.2f}" for t in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Output change (%)")
    ax.set_title(title)
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 15 else "black")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Leontief multipliers
# ---------------------------------------------------------------------------

def plot_leontief_multipliers(
    key_sectors_df: pd.DataFrame,
    title: str = "UK Sector Backward & Forward Linkages",
    save_path: Optional[str] = None,
) -> plt.Figure:
    df = key_sectors_df.sort_values("backward_linkage", ascending=True)
    colours_map = {
        "Key sector": "#2196F3",
        "Backward-linked": "#FF5722",
        "Forward-linked": "#4CAF50",
        "Weak": "#9E9E9E",
    }
    colours = [colours_map.get(c, DEFAULT_COLOUR) for c in df["classification"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df.index, df["backward_linkage"], color=colours, edgecolor="white", height=0.6)
    ax.axvline(1.0, color="black", lw=0.8, ls="--", label="Average (=1.0)")
    ax.set_xlabel("Backward linkage (normalised)")
    ax.set_title(title)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in colours_map.items()]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. REE dependence scatter
# ---------------------------------------------------------------------------

def plot_ree_dependence(
    dep_df: pd.DataFrame,
    title: str = "Direct vs Total REE Dependence by UK Sector",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        dep_df["direct_ree_dependence"],
        dep_df["total_ree_dependence"],
        s=120, color="#2196F3", alpha=0.8, edgecolors="white", zorder=3,
    )
    for sector, row in dep_df.iterrows():
        ax.annotate(
            sector,
            (row["direct_ree_dependence"], row["total_ree_dependence"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    # 45-degree reference line
    lim = max(dep_df["total_ree_dependence"].max(), dep_df["direct_ree_dependence"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Direct = Total (no indirect)")
    ax.set_xlabel("Direct REE dependence")
    ax.set_ylabel("Total REE dependence (incl. indirect)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Ghosh supply shock waterfall
# ---------------------------------------------------------------------------

def plot_ghosh_waterfall(
    ghosh_df: pd.DataFrame,
    theta: float = 0.75,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    df = ghosh_df[ghosh_df["theta"] == theta].copy()
    if df.empty:
        warnings.warn(f"No data for θ={theta}")
        return plt.figure()

    df = df.sort_values("delta_x_£bn")
    title = title or f"UK Sector Output Change — Ghosh Supply Shock (θ={theta:.2f})"

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#F44336" if v < 0 else "#4CAF50" for v in df["delta_x_£bn"]]
    ax.barh(df["sector"], df["delta_x_£bn"], color=colours, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Output change (£bn)")
    ax.set_title(title)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["delta_x_£bn"] + (0.05 if row["delta_x_£bn"] >= 0 else -0.05),
            i,
            f"{row['delta_x_pct']:.1f}%",
            va="center", ha="left" if row["delta_x_£bn"] >= 0 else "right",
            fontsize=8,
        )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. Inventory depletion
# ---------------------------------------------------------------------------

def plot_inventory_depletion(
    abm_dfs: dict[str, pd.DataFrame],
    reorder_threshold: float = 1.5,
    title: str = "Mean Manufacturer Inventory (months of supply)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in abm_dfs.items():
        colour = SCENARIO_COLOURS.get(label, DEFAULT_COLOUR)
        ax.plot(df["step"], df["mean_inventory_months"], label=label, color=colour, lw=2)
    ax.axhline(reorder_threshold, color="red", lw=1, ls="--", label=f"Reorder threshold ({reorder_threshold}mo)")
    ax.fill_between(
        range(max(df["step"].max() for df in abm_dfs.values()) + 1),
        0, reorder_threshold,
        alpha=0.08, color="red", label="Critical zone",
    )
    ax.set_xlabel("Period (months)")
    ax.set_ylabel("Inventory (months)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 8. Substitution S-curve
# ---------------------------------------------------------------------------

def plot_substitution_scurve(
    abm_dfs: dict[str, pd.DataFrame],
    title: str = "Cumulative REE Substitution Adoption by UK Manufacturers",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in abm_dfs.items():
        colour = SCENARIO_COLOURS.get(label, DEFAULT_COLOUR)
        ax.plot(df["step"], df["pct_firms_substituted"], label=label, color=colour, lw=2)
    ax.set_xlabel("Period (months)")
    ax.set_ylabel("% firms adopted substitute technology")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 9. CGE welfare bar chart
# ---------------------------------------------------------------------------

def plot_cge_welfare(
    cge_results_df: pd.DataFrame,
    title: str = "CGE Welfare Loss (Equivalent Variation) by Shock Severity",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # GDP loss
    ax = axes[0]
    colours = plt.cm.Reds(np.linspace(0.3, 0.9, len(cge_results_df)))
    bars = ax.bar(
        cge_results_df["scenario"],
        -cge_results_df["delta_GDP_£bn"],
        color=colours, edgecolor="white",
    )
    ax.set_ylabel("GDP loss (£bn)")
    ax.set_title("GDP impact (£bn)")
    ax.set_xticklabels(cge_results_df["scenario"], rotation=20, ha="right")
    for bar, val in zip(bars, -cge_results_df["delta_GDP_£bn"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"£{val:.1f}bn", ha="center", fontsize=9)

    # EV welfare
    ax = axes[1]
    ev_vals = -cge_results_df["EV_£bn"]
    colours2 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(cge_results_df)))
    bars2 = ax.bar(cge_results_df["scenario"], ev_vals, color=colours2, edgecolor="white")
    ax.set_ylabel("Equivalent variation (£bn)")
    ax.set_title("Household welfare loss (EV, £bn)")
    ax.set_xticklabels(cge_results_df["scenario"], rotation=20, ha="right")

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 10. Scenario comparison radar
# ---------------------------------------------------------------------------

def plot_scenario_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Scenario Comparison — Key Impact Indicators",
    save_path: Optional[str] = None,
) -> plt.Figure:
    metrics = [
        "peak_output_loss_£bn",
        "peak_ree_price_x_base",
        "peak_employment_loss_kFTE",
        "pct_firms_substituted_final",
    ]
    metrics = [m for m in metrics if m in comparison_df.columns]
    n_metrics = len(metrics)
    if n_metrics < 3:
        warnings.warn("Not enough metrics for radar chart; using bar chart instead.")
        return _fallback_bar_comparison(comparison_df, title, save_path)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for idx, row in comparison_df.iterrows():
        label = str(idx)
        colour = SCENARIO_COLOURS.get(label, DEFAULT_COLOUR)
        # Normalise to [0, 1]
        values = []
        for m in metrics:
            col_max = comparison_df[m].abs().max() + 1e-12
            values.append(abs(row[m]) / col_max)
        values += values[:1]
        ax.plot(angles, values, label=label, color=colour, lw=2)
        ax.fill(angles, values, alpha=0.08, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").replace("£bn", "(£bn)") for m in metrics], fontsize=9)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _fallback_bar_comparison(df, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]][:4]
    df[cols].plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_title(title)
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 11. Network risk map (simplified — requires networkx)
# ---------------------------------------------------------------------------

def plot_network_risk(
    network,
    risk_dict: dict[int, float],
    title: str = "Supply Chain Network: REE Disruption Risk",
    save_path: Optional[str] = None,
) -> plt.Figure:
    try:
        import networkx as nx
    except ImportError:
        warnings.warn("networkx not installed; skipping network plot.")
        return plt.figure()

    G = network.G
    pos = nx.spring_layout(G, seed=42, k=0.8)

    node_colours = [
        plt.cm.RdYlGn_r(risk_dict.get(n, 0.0)) for n in G.nodes()
    ]
    node_sizes = [
        200 + 600 * risk_dict.get(n, 0.0) for n in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_color=node_colours, node_size=node_sizes,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, ax=ax,
                           edge_color="#607D8B", width=0.8, arrowsize=10)
    # Label only high-risk nodes
    high_risk_labels = {
        n: G.nodes[n].get("label", str(n))
        for n in G.nodes()
        if risk_dict.get(n, 0) > 0.3
    }
    nx.draw_networkx_labels(G, pos, labels=high_risk_labels, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Disruption risk", shrink=0.7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 12. Monte Carlo uncertainty fans
# ---------------------------------------------------------------------------

def plot_mc_uncertainty(
    mc_df: pd.DataFrame,
    abm_df_central: pd.DataFrame,
    metric: str = "output_loss_£bn",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    title = title or f"Monte Carlo Uncertainty — {metric}"
    fig, ax = plt.subplots(figsize=(10, 5))

    if metric in abm_df_central.columns:
        ax.plot(abm_df_central["step"], abm_df_central[metric],
                color="#2196F3", lw=2, label="Central run")

    p5 = mc_df["peak_" + metric.replace("mean_", "")].quantile(0.05) if "peak_" + metric.replace("mean_", "") in mc_df.columns else None
    p95 = mc_df["peak_" + metric.replace("mean_", "")].quantile(0.95) if "peak_" + metric.replace("mean_", "") in mc_df.columns else None

    if p5 is not None and p95 is not None:
        ax.axhspan(p5, p95, alpha=0.15, color="#2196F3", label="5–95th percentile (MC)")

    ax.set_xlabel("Period (months)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Composite dashboard
# ---------------------------------------------------------------------------

def save_all_charts(
    output_dir: str,
    abm_dfs: dict[str, pd.DataFrame],
    ghosh_df: pd.DataFrame,
    key_sectors_df: pd.DataFrame,
    dep_df: pd.DataFrame,
    cge_results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
):
    """Export the full chart suite to output_dir."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_ree_price_path(abm_dfs, save_path=f"{output_dir}/01_ree_price.png")
    plot_output_loss_path(abm_dfs, save_path=f"{output_dir}/02_output_loss.png")
    plot_leontief_multipliers(key_sectors_df, save_path=f"{output_dir}/03_multipliers.png")
    plot_ree_dependence(dep_df, save_path=f"{output_dir}/04_ree_dependence.png")
    plot_ghosh_waterfall(ghosh_df, save_path=f"{output_dir}/05_ghosh_waterfall.png")
    plot_inventory_depletion(abm_dfs, save_path=f"{output_dir}/06_inventory.png")
    plot_substitution_scurve(abm_dfs, save_path=f"{output_dir}/07_substitution.png")
    plot_cge_welfare(cge_results_df, save_path=f"{output_dir}/08_cge_welfare.png")
    plot_scenario_comparison(comparison_df, save_path=f"{output_dir}/09_radar.png")

    print(f"All charts saved to {output_dir}/")
