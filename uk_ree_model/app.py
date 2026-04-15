"""
app.py
======
UK REE DIO–CGE–ABM Dashboard
Streamlit interactive dashboard for the UK Rare Earth Elements impact model.

Run:
    cd uk_ree_model
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import warnings
import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.uk_io_synthetic import get_io_data, SECTOR_NAMES
from dio.leontief import StaticLeontief, DynamicLeontief, REEDependenceAnalyser
from dio.ghosh import GhoshModel
from dio.mrio import build_uk_mrio_from_single_region
from cge.sam_builder import SAMBuilder
from cge.equilibrium import CGEModel, run_cge_scenarios
from abm.scheduler import UKREEModel
from integration.scenarios import get_all_scenarios, Scenario

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK REE Impact Model",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent across all charts)
# ──────────────────────────────────────────────────────────────────────────────
SECTOR_COLOURS = px.colors.qualitative.Set3
SCENARIO_COLOURS = {
    "A: Moderate":      "#2196F3",
    "B: Severe":        "#FF5722",
    "C: Sustained":     "#FF9800",
    "D: Complete":      "#9C27B0",
    "E: Net Zero Demand": "#4CAF50",
}
CHART_THEME = "plotly_white"

# ──────────────────────────────────────────────────────────────────────────────
# Cached base IO data
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_base_io():
    return get_io_data()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def apply_io_overrides(base: dict, overrides: dict) -> dict:
    """Return a copy of io_data with user-supplied parameter overrides applied."""
    io = copy.deepcopy(base)
    if "ree_intensity" in overrides:
        io["ree_intensity"] = np.array(overrides["ree_intensity"])
    if "china_import_share" in overrides:
        io["china_import_share"] = np.array(overrides["china_import_share"])
    if "x_scale" in overrides:
        io["x"] = base["x"] * overrides["x_scale"]
    return io


def build_custom_scenario(
    label: str,
    n_pre: int,
    theta_peak: float,
    n_shock: int,
    n_post: int,
    recovery_theta: float,
    policy: dict,
) -> Scenario:
    """Build a custom scenario from sidebar sliders."""
    path = (
        [0.0] * n_pre
        + list(np.linspace(0.0, theta_peak, min(3, n_shock)))
        + [theta_peak] * max(n_shock - 3, 0)
        + list(np.linspace(theta_peak, recovery_theta, n_post))
    )
    return Scenario(label=label, description="Custom scenario", theta_path=path, policy=policy)


# ──────────────────────────────────────────────────────────────────────────────
# Model runners (cached on parameter hash)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_dio(_io_data_hash, io_data_json: str):
    """Run DIO analysis. Uses JSON string for cache key."""
    import json
    io = json.loads(io_data_json)
    io = {k: np.array(v) if isinstance(v, list) else v for k, v in io.items()}
    A = io["A"]
    x = io["x"]
    sectors = SECTOR_NAMES
    lf = StaticLeontief(A, x, sectors)
    analyser = REEDependenceAnalyser(lf, io["ree_intensity"])
    ghosh = GhoshModel(A, x, sectors)
    thetas = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
    ghosh_df = ghosh.multi_theta_analysis([1], thetas)
    dep_df = analyser.summary_table().reset_index()
    key_df = lf.key_sectors().reset_index()
    mrio = build_uk_mrio_from_single_region(A, x, io["y"], sector_names=sectors)
    mrio_df = mrio.multi_scenario_summary(thetas)
    return dep_df, key_df, ghosh_df, mrio_df


@st.cache_data(show_spinner=False)
def run_cge(_io_hash, io_data_json: str, sigma_int: float, sigma_a: float):
    import json
    io = json.loads(io_data_json)
    io = {k: np.array(v) if isinstance(v, list) else v for k, v in io.items()}
    sam = SAMBuilder(io)
    thetas = [0.0, 0.10, 0.25, 0.30, 0.50, 0.75, 1.00]
    records = []
    for theta in thetas:
        model = CGEModel(sam, io, theta=theta)
        for sp in model.sector_params:
            sp.sigma_INT = sigma_int
            sp.sigma_A = sigma_a
        res = model.solve(verbose=False)
        records.append({
            "theta": theta,
            "delta_GDP_pct": res["delta_GDP_pct"],
            "EV_£bn": res["EV_£bn"],
            "CPI_pct": (res["CPI"] - 1) * 100,
            "delta_employment_kFTE": res["delta_employment_kFTE"],
            "ree_price": res["p_ree_imp"],
            "sector_delta_pct": res["sector_delta_pct"].to_dict(),
        })
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def run_abm(
    _io_hash,
    io_data_json: str,
    theta_path_tuple: tuple,
    n_manufacturers: int,
    lambda_expect: float,
    sub_threshold: float,
    s_reorder: float,
    stockpile_months: float,
    release_trigger: float,
    seed: int,
):
    import json
    io = json.loads(io_data_json)
    io = {k: np.array(v) if isinstance(v, list) else v for k, v in io.items()}

    theta_path = list(theta_path_tuple)
    model = UKREEModel(
        io_data=io,
        theta_path=theta_path,
        n_manufacturers=n_manufacturers,
        seed=seed,
    )
    # Override agent parameters
    for m in model.manufacturers:
        m.lambda_expect = lambda_expect
        m.sub_price_threshold = sub_threshold
        m.s_reorder = s_reorder
    # Override government
    model.government.stockpile = stockpile_months * io["x"][1] * 0.10 / 12
    model.government.release_trigger = release_trigger

    df = model.run()
    return df


def io_to_json(io: dict) -> str:
    import json
    serialisable = {}
    for k, v in io.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, list):
            serialisable[k] = v
        elif isinstance(v, (int, float, str)):
            serialisable[k] = v
    return json.dumps(serialisable, sort_keys=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar(base_io: dict) -> tuple[dict, dict, dict, dict]:
    """Render all sidebar controls and return (io_data, abm_params, cge_params, scenario)."""
    st.sidebar.header("UK REE Impact Model")
    st.sidebar.markdown("*Adjust parameters and re-run the simulation.*")

    # ── Scenario ──────────────────────────────────────────────────────────────
    st.sidebar.subheader("Scenario")
    use_preset = st.sidebar.radio(
        "Mode", ["Preset scenario", "Custom scenario"], horizontal=True
    )

    all_scenarios = get_all_scenarios()
    if use_preset == "Preset scenario":
        sc_key = st.sidebar.selectbox(
            "Select scenario",
            list(all_scenarios.keys()),
            format_func=lambda k: f"{k}: {all_scenarios[k].label}",
            index=1,
        )
        scenario = all_scenarios[sc_key]
        st.sidebar.info(f"**{scenario.label}**: {scenario.description[:120]}...")
    else:
        theta_peak = st.sidebar.slider("Peak disruption θ", 0.0, 1.0, 0.50, 0.05)
        n_pre      = st.sidebar.slider("Pre-shock periods (mo)", 0, 6, 3)
        n_shock    = st.sidebar.slider("Shock duration (mo)", 1, 24, 6)
        n_post     = st.sidebar.slider("Recovery periods (mo)", 1, 24, 12)
        recovery   = st.sidebar.slider("Recovery θ floor", 0.0, 0.5, 0.05, 0.05)
        scenario = build_custom_scenario(
            label="Custom",
            n_pre=n_pre, theta_peak=theta_peak,
            n_shock=n_shock, n_post=n_post,
            recovery_theta=recovery,
            policy={},
        )

    # ── ABM parameters ────────────────────────────────────────────────────────
    st.sidebar.subheader("ABM Parameters")
    n_mfr = st.sidebar.slider("Number of manufacturer agents", 20, 150, 60, 10)
    lambda_expect = st.sidebar.slider(
        "Price expectation learning rate (λ)",
        0.05, 0.95, 0.30, 0.05,
        help="How quickly firms update their REE price expectations. Higher = more reactive.",
    )
    sub_threshold = st.sidebar.slider(
        "Substitution price threshold (×base)",
        1.0, 6.0, 2.5, 0.25,
        help="REE price multiple at which firms start adopting substitute technology.",
    )
    s_reorder = st.sidebar.slider(
        "Reorder point (months of supply)",
        0.5, 4.0, 1.5, 0.25,
        help="Firms reorder when inventory falls below this level.",
    )
    abm_seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)

    # ── Government policy ─────────────────────────────────────────────────────
    st.sidebar.subheader("Government Policy")
    stockpile_months = st.sidebar.slider(
        "Strategic stockpile (months)", 0.0, 12.0, 3.0, 0.5,
        help="Size of UK strategic REE reserve, in months of consumption.",
    )
    release_trigger = st.sidebar.slider(
        "Stockpile release trigger (×price)", 1.0, 5.0, 2.5, 0.25,
        help="Government releases stockpile when REE price exceeds this multiple.",
    )

    # ── CGE elasticities ──────────────────────────────────────────────────────
    st.sidebar.subheader("CGE Elasticities")
    sigma_int = st.sidebar.slider(
        "REE substitution elasticity (σ_INT)",
        0.05, 0.80, 0.20, 0.05,
        help="How easily firms substitute away from REE inputs. Lower = more locked-in.",
    )
    sigma_a = st.sidebar.slider(
        "Armington elasticity (σ_A)",
        0.05, 1.00, 0.30, 0.05,
        help="How easily UK can switch between domestic and imported REE. Lower = more dependent on China.",
    )

    # ── IO / Sector parameters ────────────────────────────────────────────────
    st.sidebar.subheader("REE Intensity Overrides")
    with st.sidebar.expander("Adjust sector REE intensities", expanded=False):
        ree_overrides = []
        high_intensity_sectors = [5, 6, 7, 9]  # Electronics, Auto, Aerospace, Wind
        base_ree = base_io["ree_intensity"]
        for i, s in enumerate(SECTOR_NAMES):
            default_val = float(base_ree[i])
            if i == 1:  # REE sector itself — skip
                ree_overrides.append(1.0)
                continue
            new_val = st.slider(
                s, 0.0, 0.10, default_val, 0.001,
                key=f"ree_int_{i}",
                format="%.3f",
            )
            ree_overrides.append(new_val)

    st.sidebar.subheader("China Import Share Overrides")
    with st.sidebar.expander("Adjust China supply dependency", expanded=False):
        china_overrides = []
        base_china = base_io["china_import_share"]
        for i, s in enumerate(SECTOR_NAMES):
            new_val = st.slider(
                s, 0.0, 1.0, float(base_china[i]), 0.05,
                key=f"china_{i}",
            )
            china_overrides.append(new_val)

    # Build modified IO data
    io_data = apply_io_overrides(base_io, {
        "ree_intensity": ree_overrides,
        "china_import_share": china_overrides,
    })

    abm_params = dict(
        n_manufacturers=n_mfr,
        lambda_expect=lambda_expect,
        sub_threshold=sub_threshold,
        s_reorder=s_reorder,
        stockpile_months=stockpile_months,
        release_trigger=release_trigger,
        seed=int(abm_seed),
    )
    cge_params = dict(sigma_int=sigma_int, sigma_a=sigma_a)
    return io_data, abm_params, cge_params, scenario


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ──────────────────────────────────────────────────────────────────────────────
def tab_overview(scenario: Scenario, abm_df: pd.DataFrame, cge_df: pd.DataFrame):
    st.header("Overview")
    st.markdown(f"**Scenario:** {scenario.label}  \n{scenario.description}")

    if abm_df.empty:
        st.warning("Run the model to see results.")
        return

    # KPI cards
    peak_loss   = abm_df["output_loss_£bn"].max()
    peak_pct    = abm_df["output_loss_pct"].max()
    peak_price  = abm_df["ree_price"].max()
    peak_emp    = abm_df["employment_loss_kFTE"].max()
    min_inv     = abm_df["mean_inventory_months"].min()
    pct_sub     = abm_df["pct_firms_substituted"].iloc[-1]
    cge_gdp     = cge_df["delta_GDP_pct"].min() if not cge_df.empty else float("nan")
    cge_ev      = cge_df["EV_£bn"].min() if not cge_df.empty else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak output loss", f"£{peak_loss:.1f}bn/mo", f"{peak_pct:.1f}%")
    c2.metric("Peak REE price", f"{peak_price:.2f}×",
              delta=f"+{(peak_price-1)*100:.0f}%", delta_color="inverse")
    c3.metric("CGE GDP impact", f"{cge_gdp:.1f}%", delta_color="inverse")
    c4.metric("Employment at risk", f"{peak_emp:.0f}k FTE", delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Min inventory", f"{min_inv:.1f} months",
              delta="⚠ Critical" if min_inv < 1.0 else "OK",
              delta_color="inverse" if min_inv < 1.0 else "normal")
    c6.metric("Firms substituted", f"{pct_sub:.0f}%")
    c7.metric("CGE welfare loss (EV)", f"£{abs(cge_ev):.1f}bn", delta_color="inverse")
    c8.metric("Simulation periods", f"{len(abm_df)} months")

    st.divider()

    # Theta path
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(scenario.theta_path))),
            y=scenario.theta_path,
            mode="lines+markers",
            name="θ (supply restriction)",
            line=dict(color="#FF5722", width=2),
            fill="tozeroy", fillcolor="rgba(255,87,34,0.1)",
        ))
        fig.update_layout(
            title="REE Supply Restriction Path (θ)",
            xaxis_title="Period (months)",
            yaxis_title="θ",
            template=CHART_THEME,
            yaxis=dict(range=[0, 1.05]),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=abm_df["step"], y=abm_df["ree_price"],
            mode="lines", name="REE price",
            line=dict(color="#9C27B0", width=2),
        ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray",
                       annotation_text="Base price")
        fig2.add_hline(y=2.5, line_dash="dot", line_color="#FF5722",
                       annotation_text="Stockpile release trigger")
        fig2.update_layout(
            title="REE Import Price Index",
            xaxis_title="Period (months)", yaxis_title="Price (×base)",
            template=CHART_THEME, height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Output loss + employment
    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=abm_df["step"], y=abm_df["output_loss_£bn"],
            name="Monthly output loss",
            marker_color="#2196F3",
        ))
        fig3.update_layout(
            title="UK Manufacturing Output Loss (£bn/month)",
            xaxis_title="Period (months)", yaxis_title="Loss (£bn/mo)",
            template=CHART_THEME, height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=abm_df["step"], y=abm_df["mean_inventory_months"],
            mode="lines", name="Mean inventory",
            line=dict(color="#4CAF50", width=2), fill="tozeroy",
            fillcolor="rgba(76,175,80,0.1)",
        ))
        fig4.add_hrect(y0=0, y1=1.5, fillcolor="rgba(255,0,0,0.05)",
                       line_width=0, annotation_text="Critical zone", annotation_position="top right")
        fig4.update_layout(
            title="Mean Firm Inventory (months of REE supply)",
            xaxis_title="Period (months)", yaxis_title="Months",
            template=CHART_THEME, height=300,
        )
        st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — DIO Analysis
# ──────────────────────────────────────────────────────────────────────────────
def tab_dio(dep_df: pd.DataFrame, key_df: pd.DataFrame, ghosh_df: pd.DataFrame,
            mrio_df: pd.DataFrame):
    st.header("Dynamic Input–Output Analysis")

    sub = st.tabs(["REE Dependence", "Leontief Multipliers", "Ghosh Supply Shock", "MRIO Impact"])

    # ── REE Dependence ────────────────────────────────────────────────────────
    with sub[0]:
        st.subheader("Direct vs Total REE Dependence by Sector")
        st.markdown(
            "Total dependence (via Leontief inverse) captures hidden upstream REE "
            "requirements not visible in direct sourcing data. "
            "The further a sector sits above the 45° line, the larger its indirect exposure."
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            for i, row in dep_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["direct_ree_dependence"]],
                    y=[row["total_ree_dependence"]],
                    mode="markers+text",
                    name=row["sector"],
                    text=[row["sector"]],
                    textposition="top right",
                    marker=dict(size=14, color=SECTOR_COLOURS[i % len(SECTOR_COLOURS)]),
                ))
            # 45° reference
            max_val = max(dep_df["total_ree_dependence"].max(), 0.01) * 1.1
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Direct = Total (no indirect)",
                line=dict(color="gray", dash="dash"), showlegend=True,
            ))
            fig.update_layout(
                xaxis_title="Direct REE dependence",
                yaxis_title="Total REE dependence (incl. upstream)",
                template=CHART_THEME, height=480,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(
                dep_df[["sector", "direct_ree_dependence", "total_ree_dependence", "multiplier_ratio"]]
                .rename(columns={
                    "direct_ree_dependence": "Direct",
                    "total_ree_dependence": "Total",
                    "multiplier_ratio": "Multiplier",
                })
                .round(4)
                .set_index("sector"),
                height=460,
            )

    # ── Leontief Multipliers ──────────────────────────────────────────────────
    with sub[1]:
        st.subheader("Sector Linkage Classification")
        st.markdown(
            "**Key sectors** (both backward and forward linkage > 1) drive the most "
            "economy-wide activity. REE supply shocks propagate most through these sectors."
        )
        colour_map = {
            "Key sector": "#2196F3",
            "Backward-linked": "#FF5722",
            "Forward-linked": "#4CAF50",
            "Weak": "#9E9E9E",
        }
        key_df_sorted = key_df.sort_values("backward_linkage", ascending=True)
        fig = go.Figure()
        for _, row in key_df_sorted.iterrows():
            fig.add_trace(go.Bar(
                y=[row["sector"]], x=[row["backward_linkage"]],
                orientation="h",
                name=row["classification"],
                marker_color=colour_map.get(row["classification"], "#607D8B"),
                legendgroup=row["classification"],
                showlegend=True,
            ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="black",
                      annotation_text="Average linkage")
        fig.update_layout(
            xaxis_title="Backward linkage (normalised)",
            template=CHART_THEME, height=420,
            barmode="overlay",
            legend=dict(title="Classification"),
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.scatter(
                key_df, x="backward_linkage", y="forward_linkage",
                text="sector", color="classification",
                color_discrete_map=colour_map,
                labels={"backward_linkage": "Backward linkage", "forward_linkage": "Forward linkage"},
                template=CHART_THEME, height=380,
                title="Backward vs Forward Linkages",
            )
            fig2.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig2.add_vline(x=1.0, line_dash="dash", line_color="gray")
            fig2.update_traces(textposition="top right", marker_size=10)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.dataframe(
                key_df[["sector", "backward_linkage", "forward_linkage", "classification"]]
                .round(3).set_index("sector"),
                height=380,
            )

    # ── Ghosh Supply Shock ────────────────────────────────────────────────────
    with sub[2]:
        st.subheader("Ghosh Supply-Side Shock: Sector Output Impact")
        theta_sel = st.select_slider(
            "Select shock severity θ",
            options=[0.0, 0.10, 0.25, 0.50, 0.75, 1.00],
            value=0.75,
        )
        ghosh_slice = ghosh_df[ghosh_df["theta"] == theta_sel].copy()

        col1, col2 = st.columns([2, 1])
        with col1:
            ghosh_slice_sorted = ghosh_slice.sort_values("delta_x_£bn")
            colours = ["#F44336" if v < 0 else "#4CAF50" for v in ghosh_slice_sorted["delta_x_£bn"]]
            fig = go.Figure(go.Bar(
                y=ghosh_slice_sorted["sector"],
                x=ghosh_slice_sorted["delta_x_£bn"],
                orientation="h",
                marker_color=colours,
                text=[f"{p:.1f}%" for p in ghosh_slice_sorted["delta_x_pct"]],
                textposition="outside",
            ))
            fig.add_vline(x=0, line_color="black", line_width=1)
            fig.update_layout(
                title=f"Output change at θ={theta_sel:.2f} (£bn)",
                xaxis_title="Output change (£bn)",
                template=CHART_THEME, height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            total_loss = ghosh_slice["delta_x_£bn"].sum()
            st.metric("Total UK output change", f"£{total_loss:.2f}bn")
            st.metric("Worst affected", ghosh_slice.loc[ghosh_slice["delta_x_£bn"].idxmin(), "sector"])
            st.dataframe(
                ghosh_slice[["sector", "delta_x_£bn", "delta_x_pct"]]
                .rename(columns={"delta_x_£bn": "Change (£bn)", "delta_x_pct": "Change (%)"})
                .round(2).set_index("sector"),
                height=340,
            )

        # Heatmap across all theta values
        st.subheader("Output Impact Heatmap — All Shock Levels")
        pivot = ghosh_df.pivot_table(index="sector", columns="theta", values="delta_x_pct")
        fig2 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"θ={t:.2f}" for t in pivot.columns],
            y=list(pivot.index),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(title="% change"),
        ))
        fig2.update_layout(
            title="Sector Output % Change by Shock Severity",
            template=CHART_THEME, height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── MRIO Impact ───────────────────────────────────────────────────────────
    with sub[3]:
        st.subheader("Multi-Region IO: UK Impact via China–UK Trade Linkages")
        theta_mrio = st.select_slider(
            "MRIO shock severity", [0.0, 0.10, 0.25, 0.50, 0.75, 1.00], value=0.75, key="mrio_theta"
        )
        mrio_slice = mrio_df[mrio_df["theta"] == theta_mrio]

        fig = px.bar(
            mrio_slice.sort_values("delta_pct"),
            x="delta_pct", y="sector",
            orientation="h",
            color="delta_pct",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={"delta_pct": "UK output change (%)"},
            title=f"UK Sector Output Change via MRIO (θ={theta_mrio:.2f})",
            template=CHART_THEME, height=420,
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — CGE Analysis
# ──────────────────────────────────────────────────────────────────────────────
def tab_cge(cge_df: pd.DataFrame, sigma_int: float, sigma_a: float):
    st.header("Computable General Equilibrium Analysis")
    st.markdown(
        f"CES substitution elasticities: **σ_INT = {sigma_int:.2f}** "
        f"(REE-intermediate substitution) · **σ_A = {sigma_a:.2f}** (Armington trade elasticity)"
    )

    if cge_df.empty:
        st.warning("CGE results not available.")
        return

    sub = st.tabs(["GDP & Welfare", "Employment & Prices", "Sector Impact", "Elasticity Sensitivity"])

    # ── GDP & Welfare ─────────────────────────────────────────────────────────
    with sub[0]:
        col1, col2 = st.columns(2)
        with col1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=cge_df["theta"], y=-cge_df["delta_GDP_pct"],
                name="GDP loss (%)", marker_color="#2196F3",
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=cge_df["theta"], y=-cge_df["EV_£bn"],
                name="EV welfare loss (£bn)", mode="lines+markers",
                line=dict(color="#FF5722", width=2),
            ), secondary_y=True)
            fig.update_xaxes(title_text="Supply shock severity (θ)")
            fig.update_yaxes(title_text="GDP loss (%)", secondary_y=False)
            fig.update_yaxes(title_text="EV welfare loss (£bn)", secondary_y=True)
            fig.update_layout(title="GDP Impact & Welfare Loss vs Shock Severity",
                               template=CHART_THEME, height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure(go.Scatter(
                x=cge_df["theta"], y=cge_df["ree_price"],
                mode="lines+markers",
                line=dict(color="#9C27B0", width=2),
                fill="tozeroy", fillcolor="rgba(156,39,176,0.08)",
            ))
            fig2.update_layout(
                title="REE Import Price vs Supply Restriction",
                xaxis_title="θ", yaxis_title="REE price (×base)",
                template=CHART_THEME, height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            cge_df[["theta", "delta_GDP_pct", "EV_£bn", "CPI_pct", "delta_employment_kFTE", "ree_price"]]
            .rename(columns={
                "theta": "θ", "delta_GDP_pct": "ΔGDPp (%)",
                "EV_£bn": "EV (£bn)", "CPI_pct": "ΔCPI (%)",
                "delta_employment_kFTE": "Δemp (kFTE)", "ree_price": "REE price ×",
            })
            .round(3),
            use_container_width=True,
        )

    # ── Employment & Prices ───────────────────────────────────────────────────
    with sub[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=cge_df["theta"], y=-cge_df["delta_employment_kFTE"],
                marker_color=[f"rgba(255,{int(87+120*t)},34,0.8)" for t in cge_df["theta"]],
                text=[f"{v:.0f}k" for v in -cge_df["delta_employment_kFTE"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Employment Loss by Shock Severity",
                xaxis_title="θ", yaxis_title="Jobs at risk (k FTE)",
                template=CHART_THEME, height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure(go.Scatter(
                x=cge_df["theta"], y=cge_df["CPI_pct"],
                mode="lines+markers",
                line=dict(color="#FF9800", width=2),
                fill="tozeroy", fillcolor="rgba(255,152,0,0.1)",
            ))
            fig2.update_layout(
                title="Consumer Price Index Change",
                xaxis_title="θ", yaxis_title="CPI change (%)",
                template=CHART_THEME, height=360,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Sector Impact ─────────────────────────────────────────────────────────
    with sub[2]:
        theta_sel = st.select_slider(
            "Select θ for sector breakdown",
            options=sorted(cge_df["theta"].unique()), value=0.75, key="cge_sec_theta",
        )
        row = cge_df[cge_df["theta"] == theta_sel].iloc[0]
        sec_dict = row.get("sector_delta_pct", {})
        if sec_dict:
            sec_df = pd.DataFrame({"sector": list(sec_dict.keys()),
                                   "delta_pct": list(sec_dict.values())})
            fig = px.bar(
                sec_df.sort_values("delta_pct"),
                x="delta_pct", y="sector", orientation="h",
                color="delta_pct",
                color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                title=f"Sector Output Change at θ={theta_sel:.2f}",
                labels={"delta_pct": "Output change (%)"},
                template=CHART_THEME, height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sector breakdown not available for this θ.")

    # ── Elasticity Sensitivity ────────────────────────────────────────────────
    with sub[3]:
        st.subheader("How sensitive are results to substitution assumptions?")
        st.markdown(
            "Lower σ_INT → firms cannot replace REE inputs → larger GDP impact.  \n"
            "Higher σ_A → UK can more easily switch to non-Chinese REE → smaller impact."
        )
        sigma_int_vals = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
        sigma_a_vals   = [0.10, 0.20, 0.30, 0.50, 0.80]
        theta_fixed    = st.select_slider(
            "Shock severity for elasticity grid", [0.30, 0.50, 0.75], value=0.75, key="elast_theta"
        )
        st.info("Running elasticity grid… (cached after first run)")
        records = []
        io_base = load_base_io()
        sam = SAMBuilder(io_base)
        for si in sigma_int_vals:
            for sa in sigma_a_vals:
                m = CGEModel(sam, io_base, theta=theta_fixed)
                for sp in m.sector_params:
                    sp.sigma_INT = si
                    sp.sigma_A   = sa
                res = m.solve(verbose=False)
                records.append({"sigma_INT": si, "sigma_A": sa, "delta_GDP_pct": res["delta_GDP_pct"]})
        elast_df = pd.DataFrame(records)
        pivot = elast_df.pivot(index="sigma_INT", columns="sigma_A", values="delta_GDP_pct")
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"σ_A={v}" for v in pivot.columns],
            y=[f"σ_INT={v}" for v in pivot.index],
            colorscale="RdYlGn",
            zmid=pivot.values.mean(),
            text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(title="GDP change (%)"),
        ))
        fig.update_layout(
            title=f"GDP % Change — Elasticity Sensitivity Grid (θ={theta_fixed})",
            template=CHART_THEME, height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 — ABM Simulation
# ──────────────────────────────────────────────────────────────────────────────
def tab_abm(abm_df: pd.DataFrame, abm_params: dict):
    st.header("Agent-Based Model Simulation")
    st.markdown(
        f"**{abm_params['n_manufacturers']} manufacturer agents** · "
        f"λ = {abm_params['lambda_expect']:.2f} · "
        f"Sub threshold = {abm_params['sub_threshold']:.1f}× · "
        f"Reorder = {abm_params['s_reorder']:.1f}mo · "
        f"Stockpile = {abm_params['stockpile_months']:.1f}mo"
    )

    if abm_df.empty:
        st.warning("Run the model to see results.")
        return

    sub = st.tabs(["REE Market", "Output & Employment", "Inventory & Substitution",
                   "Government Stockpile", "Data Table"])

    # ── REE Market ────────────────────────────────────────────────────────────
    with sub[0]:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("REE Price (×base)", "Supply vs Demand (£bn/mo)",
                            "Supply Deficit", "Inventory Health"),
        )
        fig.add_trace(go.Scatter(
            x=abm_df["step"], y=abm_df["ree_price"],
            name="REE price", line=dict(color="#9C27B0", width=2),
        ), row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=abm_df["step"], y=abm_df["ree_supply_£bn"],
            name="Supply", line=dict(color="#4CAF50", width=2),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=abm_df["step"], y=abm_df["ree_demand_£bn"],
            name="Demand", line=dict(color="#FF5722", width=2, dash="dash"),
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=abm_df["step"], y=abm_df["ree_supply_deficit"],
            name="Supply deficit", marker_color="#FF5722",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=abm_df["step"], y=abm_df["mean_inventory_months"],
            name="Inventory (mo)", line=dict(color="#2196F3", width=2),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
        ), row=2, col=2)
        fig.add_hrect(y0=0, y1=1.5, fillcolor="rgba(255,0,0,0.06)",
                      line_width=0, row=2, col=2)

        fig.update_layout(template=CHART_THEME, height=520, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # ── Output & Employment ───────────────────────────────────────────────────
    with sub[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=abm_df["step"], y=abm_df["total_output_base_£bn"],
                name="Baseline", line=dict(color="gray", dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=abm_df["step"], y=abm_df["total_output_£bn"],
                name="Actual", line=dict(color="#2196F3", width=2),
                fill="tonexty", fillcolor="rgba(255,87,34,0.1)",
            ))
            fig.update_layout(
                title="Total Output: Baseline vs Actual",
                xaxis_title="Period (months)", yaxis_title="Output (£bn/mo)",
                template=CHART_THEME, height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(
                x=abm_df["step"], y=abm_df["output_loss_£bn"],
                name="Output loss (£bn)", marker_color="#FF5722", opacity=0.7,
            ), secondary_y=False)
            fig2.add_trace(go.Scatter(
                x=abm_df["step"], y=abm_df["employment_loss_kFTE"],
                name="Employment loss (k FTE)", mode="lines",
                line=dict(color="#FF9800", width=2),
            ), secondary_y=True)
            fig2.update_yaxes(title_text="Output loss (£bn/mo)", secondary_y=False)
            fig2.update_yaxes(title_text="Employment loss (k FTE)", secondary_y=True)
            fig2.update_layout(
                title="Output & Employment Loss",
                xaxis_title="Period (months)",
                template=CHART_THEME, height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Inventory & Substitution ───────────────────────────────────────────────
    with sub[2]:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=abm_df["step"], y=abm_df["mean_inventory_months"],
                mode="lines", name="Mean inventory",
                line=dict(color="#4CAF50", width=2),
                fill="tozeroy", fillcolor="rgba(76,175,80,0.1)",
            ))
            fig.add_trace(go.Scatter(
                x=abm_df["step"], y=abm_df["pct_firms_below_reorder"],
                mode="lines", name="Firms below reorder (%)",
                line=dict(color="#FF5722", dash="dot", width=2),
                yaxis="y2",
            ))
            fig.update_layout(
                title="Inventory Depletion",
                xaxis_title="Period (months)",
                yaxis=dict(title="Inventory (months)"),
                yaxis2=dict(title="% firms below reorder", overlaying="y", side="right"),
                template=CHART_THEME, height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # S-curve adoption
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=abm_df["step"], y=abm_df["pct_firms_substituted"],
                mode="lines", name="Firms substituted (%)",
                line=dict(color="#9C27B0", width=2),
                fill="tozeroy", fillcolor="rgba(156,39,176,0.1)",
            ))
            fig2.update_layout(
                title="Cumulative REE Substitution Adoption (S-curve)",
                xaxis_title="Period (months)", yaxis_title="% firms adopted",
                yaxis=dict(range=[0, 105]),
                template=CHART_THEME, height=360,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Government Stockpile ───────────────────────────────────────────────────
    with sub[3]:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Stockpile Level (£bn)", "Stockpile Releases (£bn/mo)"))
        fig.add_trace(go.Scatter(
            x=abm_df["step"], y=abm_df["gov_stockpile_£bn"],
            line=dict(color="#2196F3", width=2), name="Stockpile",
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=abm_df["step"], y=abm_df["gov_release_£bn"],
            marker_color="#4CAF50", name="Release",
        ), row=1, col=2)
        fig.update_layout(template=CHART_THEME, height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Data Table ─────────────────────────────────────────────────────────────
    with sub[4]:
        st.dataframe(
            abm_df[[
                "step", "theta", "ree_price", "ree_supply_£bn", "ree_demand_£bn",
                "total_output_£bn", "output_loss_£bn", "output_loss_pct",
                "employment_kFTE", "employment_loss_kFTE",
                "mean_inventory_months", "pct_firms_below_reorder",
                "pct_firms_substituted", "gov_stockpile_£bn",
            ]].round(3),
            use_container_width=True,
        )
        csv = abm_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download ABM results (CSV)", csv, "abm_results.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 5 — Scenario Comparison
# ──────────────────────────────────────────────────────────────────────────────
def tab_comparison(io_data: dict, abm_params: dict, cge_params: dict):
    st.header("Multi-Scenario Comparison")
    st.markdown("Compare all five pre-defined scenarios side-by-side using current model parameters.")

    run_btn = st.button("Run all 5 scenarios", type="primary")
    if not run_btn and "comparison_results" not in st.session_state:
        st.info("Click **Run all 5 scenarios** to generate the comparison (takes ~30–60 seconds).")
        return

    if run_btn:
        io_json = io_to_json(io_data)
        scenarios = get_all_scenarios()
        results = {}
        prog = st.progress(0, text="Running scenarios...")
        for i, (k, sc) in enumerate(scenarios.items()):
            prog.progress((i) / 5, text=f"Running scenario {k}: {sc.label}…")
            abm_df_sc = run_abm(
                hash(io_json),
                io_json,
                tuple(sc.theta_path),
                abm_params["n_manufacturers"],
                abm_params["lambda_expect"],
                abm_params["sub_threshold"],
                abm_params["s_reorder"],
                abm_params["stockpile_months"],
                abm_params["release_trigger"],
                abm_params["seed"],
            )
            results[sc.label] = {"abm": abm_df_sc, "scenario": sc}
        prog.progress(1.0, text="Done.")
        st.session_state["comparison_results"] = results

    results = st.session_state.get("comparison_results", {})
    if not results:
        return

    # ── REE price paths ───────────────────────────────────────────────────────
    fig = go.Figure()
    for label, res in results.items():
        df = res["abm"]
        colour = SCENARIO_COLOURS.get(label, "#607D8B")
        fig.add_trace(go.Scatter(
            x=df["step"], y=df["ree_price"],
            name=label, line=dict(color=colour, width=2),
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="REE Price Path — All Scenarios",
        xaxis_title="Period (months)", yaxis_title="REE price (×base)",
        template=CHART_THEME, height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        for label, res in results.items():
            df = res["abm"]
            colour = SCENARIO_COLOURS.get(label, "#607D8B")
            fig2.add_trace(go.Scatter(
                x=df["step"], y=df["output_loss_pct"],
                name=label, line=dict(color=colour, width=2),
            ))
        fig2.update_layout(
            title="Output Loss % — All Scenarios",
            xaxis_title="Period (months)", yaxis_title="Output loss (%)",
            template=CHART_THEME, height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        for label, res in results.items():
            df = res["abm"]
            colour = SCENARIO_COLOURS.get(label, "#607D8B")
            fig3.add_trace(go.Scatter(
                x=df["step"], y=df["pct_firms_substituted"],
                name=label, line=dict(color=colour, width=2),
            ))
        fig3.update_layout(
            title="Substitution Adoption — All Scenarios",
            xaxis_title="Period (months)", yaxis_title="% firms substituted",
            template=CHART_THEME, height=350,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Peak Impact Summary")
    rows = []
    for label, res in results.items():
        df = res["abm"]
        rows.append({
            "Scenario": label,
            "Peak θ": f"{max(res['scenario'].theta_path):.2f}",
            "Duration (mo)": res["scenario"].n_periods,
            "Peak output loss (£bn/mo)": round(df["output_loss_£bn"].max(), 2),
            "Peak output loss (%)": round(df["output_loss_pct"].max(), 1),
            "Peak REE price (×)": round(df["ree_price"].max(), 2),
            "Peak emp loss (k FTE)": round(df["employment_loss_kFTE"].max(), 0),
            "Min inventory (mo)": round(df["mean_inventory_months"].min(), 2),
            "% firms substituted": round(df["pct_firms_substituted"].iloc[-1], 1),
        })
    comp_df = pd.DataFrame(rows).set_index("Scenario")
    st.dataframe(comp_df, use_container_width=True)

    csv = comp_df.to_csv().encode("utf-8")
    st.download_button("Download comparison table (CSV)", csv, "scenario_comparison.csv", "text/csv")

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("Scenario Radar (normalised)")
    metrics = ["Peak output loss (%)", "Peak REE price (×)", "Peak emp loss (k FTE)",
                "% firms substituted"]
    fig4 = go.Figure()
    for label, row in comp_df.iterrows():
        values = []
        for m in metrics:
            col_max = comp_df[m].max() + 1e-9
            values.append(abs(row[m]) / col_max * 100)
        values += values[:1]
        colour = SCENARIO_COLOURS.get(label, "#607D8B")
        fig4.add_trace(go.Scatterpolar(
            r=values, theta=metrics + [metrics[0]],
            fill="toself", name=label,
            line=dict(color=colour),
            fillcolor=colour.replace(")", ",0.1)").replace("rgb", "rgba"),
        ))
    fig4.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template=CHART_THEME, height=450,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 6 — Sensitivity Analysis
# ──────────────────────────────────────────────────────────────────────────────
def tab_sensitivity(io_data: dict, abm_params: dict):
    st.header("Sensitivity Analysis")
    sub = st.tabs(["ABM Parameter Sweep", "REE Intensity Sensitivity", "China Dependency"])

    # ── ABM Parameter Sweep ───────────────────────────────────────────────────
    with sub[0]:
        st.subheader("How does peak output loss vary with ABM agent parameters?")
        st.markdown("Select a parameter to sweep while holding others at sidebar values.")

        sweep_param = st.selectbox(
            "Parameter to sweep",
            ["lambda_expect (learning rate)", "sub_threshold (substitution trigger)",
             "s_reorder (reorder point)", "stockpile_months"],
        )
        io_json = io_to_json(io_data)
        all_sc = get_all_scenarios()
        sc_b   = all_sc["B"]

        if sweep_param.startswith("lambda"):
            sweep_vals = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
            param_key  = "lambda_expect"
        elif sweep_param.startswith("sub_threshold"):
            sweep_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
            param_key  = "sub_threshold"
        elif sweep_param.startswith("s_reorder"):
            sweep_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
            param_key  = "s_reorder"
        else:
            sweep_vals = [0.0, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0]
            param_key  = "stockpile_months"

        sweep_records = []
        prog = st.progress(0, text=f"Sweeping {sweep_param}…")
        for i, val in enumerate(sweep_vals):
            prog.progress(i / len(sweep_vals), text=f"  {param_key} = {val}")
            kwargs = {**abm_params, param_key: val}
            df_s = run_abm(
                hash(io_json + str(val)),
                io_json, tuple(sc_b.theta_path),
                kwargs["n_manufacturers"], kwargs["lambda_expect"],
                kwargs["sub_threshold"], kwargs["s_reorder"],
                kwargs["stockpile_months"], kwargs["release_trigger"],
                kwargs["seed"],
            )
            sweep_records.append({
                param_key: val,
                "peak_loss_pct": df_s["output_loss_pct"].max(),
                "peak_price": df_s["ree_price"].max(),
                "final_substituted": df_s["pct_firms_substituted"].iloc[-1],
                "min_inventory": df_s["mean_inventory_months"].min(),
            })
        prog.progress(1.0, text="Done.")
        sweep_df = pd.DataFrame(sweep_records)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                sweep_df, x=param_key, y="peak_loss_pct", markers=True,
                labels={param_key: sweep_param, "peak_loss_pct": "Peak output loss (%)"},
                title=f"Peak Output Loss vs {sweep_param}",
                template=CHART_THEME, height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(
                sweep_df, x=param_key, y=["peak_price", "min_inventory"],
                markers=True,
                labels={param_key: sweep_param, "value": "Value"},
                title=f"REE Price & Inventory vs {sweep_param}",
                template=CHART_THEME, height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── REE Intensity Sensitivity ─────────────────────────────────────────────
    with sub[1]:
        st.subheader("Impact of Sector REE Intensity on UK Output Loss")
        st.markdown(
            "Test how UK economy-wide output loss changes as the REE intensity "
            "of a selected sector varies."
        )
        target_sector = st.selectbox(
            "Sector to vary", SECTOR_NAMES, index=6
        )
        sec_idx = SECTOR_NAMES.index(target_sector)
        base_val = float(io_data["ree_intensity"][sec_idx])
        intensity_range = st.slider(
            "Intensity range (min, max)",
            0.0, 0.15, (max(0.0, base_val - 0.02), min(0.15, base_val + 0.04)), 0.002,
        )
        intensity_vals = list(np.linspace(intensity_range[0], intensity_range[1], 12))

        int_records = []
        io_base = load_base_io()
        for iv in intensity_vals:
            io_mod = copy.deepcopy(io_data)
            io_mod["ree_intensity"][sec_idx] = iv
            io_json_mod = io_to_json(io_mod)
            df_int = run_abm(
                hash(io_json_mod),
                io_json_mod, tuple(all_sc["B"].theta_path),
                abm_params["n_manufacturers"], abm_params["lambda_expect"],
                abm_params["sub_threshold"], abm_params["s_reorder"],
                abm_params["stockpile_months"], abm_params["release_trigger"],
                abm_params["seed"],
            )
            int_records.append({
                "ree_intensity": round(iv, 4),
                "peak_loss_pct": df_int["output_loss_pct"].max(),
                "peak_price": df_int["ree_price"].max(),
            })

        int_df = pd.DataFrame(int_records)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Peak Output Loss (%)", "Peak REE Price (×)"))
        fig.add_trace(go.Scatter(
            x=int_df["ree_intensity"], y=int_df["peak_loss_pct"],
            mode="lines+markers", line=dict(color="#FF5722", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=int_df["ree_intensity"], y=int_df["peak_price"],
            mode="lines+markers", line=dict(color="#9C27B0", width=2),
        ), row=1, col=2)
        fig.add_vline(x=base_val, line_dash="dash", line_color="blue",
                      annotation_text="Current", row=1, col=1)
        fig.add_vline(x=base_val, line_dash="dash", line_color="blue", row=1, col=2)
        fig.update_xaxes(title_text="REE intensity")
        fig.update_layout(template=CHART_THEME, height=360, showlegend=False,
                           title=f"Sensitivity to {target_sector} REE Intensity (Scenario B)")
        st.plotly_chart(fig, use_container_width=True)

    # ── China Dependency ──────────────────────────────────────────────────────
    with sub[2]:
        st.subheader("Impact of China Import Dependency on UK Vulnerability")
        china_share_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        china_records = []
        for cs in china_share_vals:
            io_mod = copy.deepcopy(io_data)
            # Scale all China import shares uniformly
            io_mod["china_import_share"] = np.minimum(
                io_data["china_import_share"] * (cs / 0.70), 0.99
            )
            io_json_c = io_to_json(io_mod)
            df_c = run_abm(
                hash(io_json_c + f"china{cs}"),
                io_json_c, tuple(all_sc["B"].theta_path),
                abm_params["n_manufacturers"], abm_params["lambda_expect"],
                abm_params["sub_threshold"], abm_params["s_reorder"],
                abm_params["stockpile_months"], abm_params["release_trigger"],
                abm_params["seed"],
            )
            china_records.append({
                "china_share_scale": cs,
                "peak_loss_pct": df_c["output_loss_pct"].max(),
                "peak_price": df_c["ree_price"].max(),
                "peak_emp_loss": df_c["employment_loss_kFTE"].max(),
            })

        china_df = pd.DataFrame(china_records)
        fig = px.line(
            china_df, x="china_share_scale",
            y=["peak_loss_pct", "peak_price"],
            markers=True,
            labels={"china_share_scale": "China supply share (scaled)", "value": "Value"},
            title="UK Vulnerability vs China Import Dependency (Scenario B, θ=0.75)",
            template=CHART_THEME, height=380,
        )
        fig.add_vline(x=0.70, line_dash="dash", line_color="blue",
                      annotation_text="Current baseline (70%)")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("UK Rare Earth Elements Impact Model")
    st.markdown(
        "**Dynamic IO · Computable General Equilibrium · Agent-Based Model**  \n"
        "Analyse the economic impact of REE supply disruptions on the UK economy "
        "across five supply shock scenarios."
    )

    base_io = load_base_io()
    io_data, abm_params, cge_params, scenario = render_sidebar(base_io)

    # ── Run models ────────────────────────────────────────────────────────────
    io_json = io_to_json(io_data)
    io_hash = hash(io_json)

    with st.spinner("Running DIO analysis…"):
        dep_df, key_df, ghosh_df, mrio_df = run_dio(io_hash, io_json)

    with st.spinner("Running CGE analysis…"):
        cge_df = run_cge(io_hash, io_json, cge_params["sigma_int"], cge_params["sigma_a"])

    with st.spinner(f"Running ABM ({scenario.label}, {scenario.n_periods} periods)…"):
        abm_df = run_abm(
            io_hash, io_json,
            tuple(scenario.theta_path),
            abm_params["n_manufacturers"],
            abm_params["lambda_expect"],
            abm_params["sub_threshold"],
            abm_params["s_reorder"],
            abm_params["stockpile_months"],
            abm_params["release_trigger"],
            abm_params["seed"],
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Overview",
        "DIO Analysis",
        "CGE Analysis",
        "ABM Simulation",
        "Scenario Comparison",
        "Sensitivity",
    ])

    with tabs[0]:
        tab_overview(scenario, abm_df, cge_df)
    with tabs[1]:
        tab_dio(dep_df, key_df, ghosh_df, mrio_df)
    with tabs[2]:
        tab_cge(cge_df, cge_params["sigma_int"], cge_params["sigma_a"])
    with tabs[3]:
        tab_abm(abm_df, abm_params)
    with tabs[4]:
        tab_comparison(io_data, abm_params, cge_params)
    with tabs[5]:
        tab_sensitivity(io_data, abm_params)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "UK REE DIO–CGE–ABM Model · Data: ONS, BGS CMIC, HMRC, OECD ICIO (synthetic calibration) · "
        "April 2026"
    )


if __name__ == "__main__":
    main()
