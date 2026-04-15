"""
sensitivity.py
==============
Sensitivity and uncertainty analysis for the UK REE DIO–CGE–ABM model.

Analyses:
  1. One-at-a-time (OAT) sensitivity of DIO output multipliers to A matrix
  2. Monte Carlo uncertainty quantification over ABM agent parameters
  3. CGE elasticity sensitivity (σ_INT, σ_A)
  4. Scenario comparison table across all five scenarios

Output: DataFrames and summary statistics ready for visualisation.py.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# 1. DIO sensitivity: output multipliers to A matrix perturbations
# ---------------------------------------------------------------------------

def dio_oat_sensitivity(
    leontief,
    ree_intensity: np.ndarray,
    perturbation: float = 0.10,
) -> pd.DataFrame:
    """
    One-at-a-time sensitivity of total REE requirements to each A matrix entry.

    Parameters
    ----------
    leontief      : StaticLeontief  Fitted Leontief model.
    ree_intensity : np.ndarray      REE intensity vector.
    perturbation  : float           Fractional change to each a_ij (default 10%).

    Returns
    -------
    DataFrame with sensitivity indices (d(TRR) / d(a_ij)) for each entry.
    """
    from dio.leontief import StaticLeontief, REEDependenceAnalyser

    n = leontief.n
    base_trd = REEDependenceAnalyser(leontief, ree_intensity).total_dependence()
    base_norm = np.linalg.norm(base_trd)

    records = []
    for i in range(n):
        for j in range(n):
            if leontief.A[i, j] < 1e-6:
                continue
            A_perturbed = leontief.A.copy()
            A_perturbed[i, j] *= (1 + perturbation)
            try:
                lf_perturbed = StaticLeontief(A_perturbed, leontief.x, leontief.sector_names)
                trd_p = REEDependenceAnalyser(lf_perturbed, ree_intensity).total_dependence()
                delta_norm = np.linalg.norm(trd_p) - base_norm
                records.append({
                    "input_sector": leontief.sector_names[i],
                    "using_sector": leontief.sector_names[j],
                    "a_ij_base": leontief.A[i, j],
                    "sensitivity": delta_norm / (perturbation * base_norm + 1e-12),
                })
            except Exception:
                pass

    df = pd.DataFrame(records).sort_values("sensitivity", ascending=False)
    return df


# ---------------------------------------------------------------------------
# 2. Monte Carlo uncertainty over ABM parameters
# ---------------------------------------------------------------------------

def _run_single_mc(
    io_data: dict,
    scenario,
    seed: int,
    n_manufacturers: int,
    lambda_mean: float,
    sub_threshold_mean: float,
    s_reorder_mean: float,
) -> dict:
    """Single Monte Carlo run with randomised agent parameters."""
    from abm.scheduler import UKREEModel

    rng = np.random.default_rng(seed)

    # Perturb global parameters via monkey-patching the rng seed in the model
    model = UKREEModel(
        io_data=io_data,
        cge_equilibrium=None,
        theta_path=scenario.theta_path,
        n_manufacturers=n_manufacturers,
        seed=seed,
    )

    # Re-draw agent parameters around means
    for m in model.manufacturers:
        m.lambda_expect = float(np.clip(rng.normal(lambda_mean, 0.08), 0.05, 0.95))
        m.sub_price_threshold = float(np.clip(rng.normal(sub_threshold_mean, 0.5), 1.0, 6.0))
        m.s_reorder = float(np.clip(rng.normal(s_reorder_mean, 0.3), 0.5, 4.0))

    results_df = model.run()
    peak = results_df.loc[results_df["output_loss_£bn"].idxmax()]

    return {
        "seed": seed,
        "peak_output_loss_£bn": float(peak["output_loss_£bn"]),
        "peak_ree_price": float(peak["ree_price"]),
        "peak_employment_loss_kFTE": float(peak["employment_loss_kFTE"]),
        "final_pct_substituted": float(results_df["pct_firms_substituted"].iloc[-1]),
        "mean_inventory_months": float(results_df["mean_inventory_months"].mean()),
    }


def monte_carlo_abm(
    io_data: dict,
    scenario,
    n_runs: int = 50,
    n_manufacturers: int = 60,
    n_jobs: int = -1,
    lambda_mean: float = 0.30,
    sub_threshold_mean: float = 2.5,
    s_reorder_mean: float = 1.5,
) -> pd.DataFrame:
    """
    Monte Carlo simulation: vary ABM agent parameters across n_runs.

    Parameters
    ----------
    io_data          : dict      UK IO data.
    scenario         : Scenario  Scenario to test.
    n_runs           : int       Number of MC samples.
    n_manufacturers  : int       Agents per run (fewer = faster).
    n_jobs           : int       Parallel jobs (-1 = all cores).

    Returns
    -------
    DataFrame with per-run results; use .describe() for uncertainty bands.
    """
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_run_single_mc)(
            io_data, scenario, seed=i,
            n_manufacturers=n_manufacturers,
            lambda_mean=lambda_mean,
            sub_threshold_mean=sub_threshold_mean,
            s_reorder_mean=s_reorder_mean,
        )
        for i in range(n_runs)
    )
    df = pd.DataFrame(results)
    return df


def mc_summary(mc_df: pd.DataFrame) -> pd.DataFrame:
    """Return 5th, 50th, 95th percentiles for key MC outputs."""
    cols = [c for c in mc_df.columns if c != "seed"]
    rows = []
    for col in cols:
        rows.append({
            "metric": col,
            "p5": mc_df[col].quantile(0.05),
            "p50": mc_df[col].quantile(0.50),
            "mean": mc_df[col].mean(),
            "p95": mc_df[col].quantile(0.95),
            "std": mc_df[col].std(),
        })
    return pd.DataFrame(rows).set_index("metric")


# ---------------------------------------------------------------------------
# 3. CGE elasticity sensitivity
# ---------------------------------------------------------------------------

def cge_elasticity_sensitivity(
    sam,
    io_data: dict,
    theta: float,
    sigma_int_values: Optional[list] = None,
    sigma_a_values: Optional[list] = None,
) -> pd.DataFrame:
    """
    Sensitivity of CGE GDP impact to key substitution elasticities.

    Parameters
    ----------
    sam             : SAMBuilder
    io_data         : dict
    theta           : float
    sigma_int_values: list  Values of σ_INT to test (REE–NREE substitution).
    sigma_a_values  : list  Values of σ_A to test (Armington).

    Returns
    -------
    DataFrame with GDP loss at each elasticity combination.
    """
    from cge.equilibrium import CGEModel

    if sigma_int_values is None:
        sigma_int_values = [0.05, 0.15, 0.25, 0.40, 0.60]
    if sigma_a_values is None:
        sigma_a_values = [0.10, 0.20, 0.30, 0.50, 0.80]

    records = []
    for sig_int in sigma_int_values:
        for sig_a in sigma_a_values:
            # Temporarily override elasticities
            mod_io = io_data.copy()
            model = CGEModel(sam, mod_io, theta=theta)
            for sp in model.sector_params:
                sp.sigma_INT = sig_int
                sp.sigma_A = sig_a
            try:
                result = model.solve(verbose=False)
                records.append({
                    "sigma_INT": sig_int,
                    "sigma_A": sig_a,
                    "delta_GDP_£bn": result["delta_GDP_£bn"],
                    "delta_GDP_pct": result["delta_GDP_pct"],
                    "EV_£bn": result["EV_£bn"],
                    "delta_employment_kFTE": result["delta_employment_kFTE"],
                })
            except Exception as e:
                records.append({
                    "sigma_INT": sig_int,
                    "sigma_A": sig_a,
                    "delta_GDP_£bn": np.nan,
                    "delta_GDP_pct": np.nan,
                    "EV_£bn": np.nan,
                    "delta_employment_kFTE": np.nan,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Cross-scenario comparison
# ---------------------------------------------------------------------------

def scenario_comparison_table(
    coupled_results: dict[str, dict],
) -> pd.DataFrame:
    """
    Build a summary comparison table across all scenarios.

    Parameters
    ----------
    coupled_results : dict  label → output of CoupledModel.run()

    Returns
    -------
    DataFrame with key KPIs per scenario.
    """
    rows = []
    for label, res in coupled_results.items():
        abm_df = res.get("abm_df", pd.DataFrame())
        cge_df = res.get("cge_summary", pd.DataFrame())
        mrio_df = res.get("mrio_summary", pd.DataFrame())

        peak_output_loss = abm_df["output_loss_£bn"].max() if not abm_df.empty else np.nan
        peak_ree_price = abm_df["ree_price"].max() if not abm_df.empty else np.nan
        peak_emp_loss = abm_df["employment_loss_kFTE"].max() if not abm_df.empty else np.nan
        final_substituted = abm_df["pct_firms_substituted"].iloc[-1] if not abm_df.empty else np.nan
        min_inventory = abm_df["mean_inventory_months"].min() if not abm_df.empty else np.nan

        cge_gdp_pct = cge_df["delta_GDP_pct"].min() if not cge_df.empty else np.nan
        cge_ev = cge_df["EV_£bn"].min() if not cge_df.empty else np.nan

        mrio_loss = mrio_df["delta_£bn"].sum() if not mrio_df.empty else np.nan

        rows.append({
            "scenario": label,
            "peak_output_loss_£bn": peak_output_loss,
            "peak_ree_price_x_base": peak_ree_price,
            "peak_employment_loss_kFTE": peak_emp_loss,
            "cge_delta_GDP_pct": cge_gdp_pct,
            "cge_EV_£bn": cge_ev,
            "mrio_uk_loss_£bn": mrio_loss,
            "pct_firms_substituted_final": final_substituted,
            "min_inventory_months": min_inventory,
        })

    return pd.DataFrame(rows).set_index("scenario")
