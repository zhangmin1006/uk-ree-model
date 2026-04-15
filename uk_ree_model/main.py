"""
main.py
=======
UK REE DIO–CGE–ABM Model — Main Runner

Usage:
  python main.py                          # run all 5 scenarios
  python main.py --scenario B             # run single scenario
  python main.py --scenario B --mc        # with Monte Carlo
  python main.py --quick                  # fast test run (fewer agents, fewer periods)

Outputs written to: uk_ree_model/output/
"""

from __future__ import annotations

import argparse
import io as _io
import os
import sys
import warnings

# Force UTF-8 stdout/stderr on Windows (avoids £/θ/Δ encoding errors)
if sys.platform == "win32":
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure uk_ree_model package is importable from this directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.uk_io_synthetic import get_io_data, as_dataframes

from dio.leontief import StaticLeontief, DynamicLeontief, REEDependenceAnalyser
from dio.ghosh import GhoshModel
from dio.mrio import build_uk_mrio_from_single_region

from cge.sam_builder import SAMBuilder
from cge.equilibrium import CGEModel, run_cge_scenarios

from abm.scheduler import UKREEModel

from integration.scenarios import get_all_scenarios, cge_scenario_thetas
from integration.coupling import CoupledModel

from analysis.sensitivity import (
    dio_oat_sensitivity, monte_carlo_abm, mc_summary,
    cge_elasticity_sensitivity, scenario_comparison_table,
)
from analysis.visualisation import save_all_charts


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def banner(text: str):
    width = 65
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def section(text: str):
    print(f"\n--- {text} ---")


# ---------------------------------------------------------------------------
# DIO standalone analysis
# ---------------------------------------------------------------------------

def run_dio_analysis(io_data: dict) -> dict:
    banner("LAYER 1: Dynamic Input–Output Analysis")

    A = io_data["A"]
    B = io_data["B"]
    x = io_data["x"]
    y = io_data["y"]
    ree = io_data["ree_intensity"]
    sectors = io_data["sector_names"]

    # Static Leontief
    section("Static Leontief Model")
    lf = StaticLeontief(A, x, sectors)
    x_hat = lf.final_demand_to_output(y)
    key_df = lf.key_sectors()
    print(key_df.to_string())

    # REE dependence
    section("REE Dependence Analysis")
    analyser = REEDependenceAnalyser(lf, ree)
    dep_df = analyser.summary_table()
    print(dep_df.round(4).to_string())

    # Ghosh: supply shock across θ values
    section("Ghosh Supply-Side Shock Propagation")
    ghosh = GhoshModel(A, x, sectors)
    thetas = [0.0, 0.25, 0.50, 0.75, 1.00]
    ghosh_df = ghosh.multi_theta_analysis([1], thetas)
    peak = ghosh_df[ghosh_df["theta"] == 0.75].copy()
    print(f"\nOutput losses at θ=0.75 (£bn):")
    print(peak[["sector", "delta_x_£bn", "delta_x_pct"]].set_index("sector").round(3).to_string())

    # MRIO
    section("Multi-Region IO: UK–China–RoW")
    mrio = build_uk_mrio_from_single_region(A, x, y, sector_names=sectors)
    mrio_impact = mrio.multi_scenario_summary([0.0, 0.30, 0.50, 0.75], ree_rows=[1])
    uk_peak = mrio_impact[mrio_impact["theta"] == 0.75].groupby("sector")["delta_£bn"].sum()
    print(f"Total UK output loss at θ=0.75 (MRIO): £{uk_peak.sum():.2f}bn")

    # Dynamic: simulate 24-period output path under scenario B theta profile
    section("Dynamic IO: 24-period output path (Scenario B)")
    sc_b_theta = [0.0] * 3 + [0.0 + 0.75 * (t / 2) for t in range(2)] + [0.75] * 4 + \
                 [0.75 - 0.065 * t for t in range(1, 13)] + [0.1, 0.1, 0.05, 0.05]
    sc_b_theta = sc_b_theta[:24]
    c_path = np.tile(y * 0.85, (24, 1))  # non-investment demand
    dio = DynamicLeontief(A, B, x, sectors)
    x_base, x_shocked = dio.simulate_shock(
        c_path=c_path,
        shock_vector=-ree * x * 0.75,
        shock_start=3,
        shock_end=9,
    )
    va_coeff = io_data.get("va_coeff", np.full(len(sectors), 0.45))
    gdp_base = dio.gdp_path(x_base, va_coeff)
    gdp_shocked = dio.gdp_path(x_shocked, va_coeff)
    print(f"Peak GDP loss (Dynamic IO, scenario B): £{(gdp_shocked - gdp_base).min():.1f}bn")

    return {
        "leontief": lf,
        "key_sectors": key_df,
        "dep_df": dep_df,
        "ghosh": ghosh,
        "ghosh_df": ghosh_df,
        "mrio": mrio,
        "dio": dio,
    }


# ---------------------------------------------------------------------------
# CGE standalone analysis
# ---------------------------------------------------------------------------

def run_cge_analysis(io_data: dict) -> dict:
    banner("LAYER 2: Computable General Equilibrium Analysis")

    sam = SAMBuilder(io_data)
    sam.print_summary()

    section("SAM balance check")
    balance_df = sam.check_balance()
    print(balance_df[["imbalance_pct", "balanced"]].to_string())

    section("CGE: Baseline and scenario θ results")
    thetas = [0.0, 0.25, 0.50, 0.75, 1.00]
    cge_df = run_cge_scenarios(sam, io_data, thetas, verbose=True)
    print("\nCGE scenario summary:")
    print(cge_df.to_string(index=False))

    section("CGE elasticity sensitivity (σ_INT × σ_A grid)")
    elast_df = cge_elasticity_sensitivity(sam, io_data, theta=0.75)
    pivot = elast_df.pivot(index="sigma_INT", columns="sigma_A", values="delta_GDP_pct")
    print("\nGDP % change by (σ_INT, σ_A):")
    print(pivot.round(2).to_string())

    return {
        "sam": sam,
        "cge_df": cge_df,
        "elast_df": elast_df,
    }


# ---------------------------------------------------------------------------
# ABM standalone analysis
# ---------------------------------------------------------------------------

def run_abm_analysis(io_data: dict, quick: bool = False) -> dict:
    banner("LAYER 3: Agent-Based Model")

    from integration.scenarios import scenario_b
    sc = scenario_b()
    n_mfr = 30 if quick else 80

    section(f"Running ABM — {sc.label} ({sc.n_periods} periods, {n_mfr} agents)")
    t0 = time.time()

    sam = SAMBuilder(io_data)
    cge = CGEModel(sam, io_data, theta=0.0)
    cge_eq = cge.solve(verbose=False)

    model = UKREEModel(
        io_data=io_data,
        cge_equilibrium=cge_eq,
        theta_path=sc.theta_path,
        n_manufacturers=n_mfr,
        seed=42,
    )
    abm_df = model.run()
    elapsed = time.time() - t0

    print(f"ABM completed in {elapsed:.1f}s")
    print(f"\nABM summary (scenario B):")
    print(model.metrics.summary().round(3).to_string())

    peak = model.metrics.peak_disruption()
    print(f"\nPeak disruption period: {peak.get('step', '?')}")
    print(f"  Peak output loss:   £{peak.get('output_loss_£bn', 0):.2f}bn")
    print(f"  Peak REE price:     {peak.get('ree_price', 0):.2f}×")
    print(f"  Employment loss:    {peak.get('employment_loss_kFTE', 0):.1f}k FTE")

    return {
        "model": model,
        "abm_df": abm_df,
        "network": model.network,
    }


# ---------------------------------------------------------------------------
# Full coupled run
# ---------------------------------------------------------------------------

def run_coupled(
    io_data: dict,
    scenario_keys: list[str],
    quick: bool = False,
    run_mc: bool = False,
) -> dict:
    banner("FULL COUPLED DIO–CGE–ABM SIMULATION")

    all_scenarios = get_all_scenarios()
    n_mfr = 30 if quick else 80
    coupled_results = {}
    abm_dfs = {}

    for key in scenario_keys:
        sc = all_scenarios[key]
        section(f"Running coupled model — {sc.label}")
        t0 = time.time()

        model = CoupledModel(
            io_data=io_data,
            scenario=sc,
            n_manufacturers=n_mfr,
            cge_freq=4 if quick else 3,
            verbose=True,
            seed=42,
        )
        result = model.run()
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        coupled_results[sc.label] = result
        abm_dfs[sc.label] = result["abm_df"]

    # Cross-scenario comparison
    section("Cross-scenario comparison table")
    comp_df = scenario_comparison_table(coupled_results)
    print(comp_df.round(2).to_string())

    # Monte Carlo
    mc_df = None
    if run_mc and "B" in scenario_keys:
        section("Monte Carlo uncertainty (Scenario B, 30 runs)")
        mc_df = monte_carlo_abm(
            io_data=io_data,
            scenario=all_scenarios["B"],
            n_runs=30 if quick else 50,
            n_manufacturers=30,
            n_jobs=1,
        )
        print("\nMonte Carlo uncertainty bands:")
        print(mc_summary(mc_df).round(3).to_string())

    return {
        "coupled_results": coupled_results,
        "abm_dfs": abm_dfs,
        "comp_df": comp_df,
        "mc_df": mc_df,
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(results: dict, dio_results: dict, cge_results: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    section("Saving outputs")

    # Tables
    if "comp_df" in results and results["comp_df"] is not None:
        results["comp_df"].to_csv(f"{OUTPUT_DIR}/scenario_comparison.csv")
        print(f"  Saved scenario_comparison.csv")

    if "cge_df" in cge_results:
        cge_results["cge_df"].to_csv(f"{OUTPUT_DIR}/cge_results.csv", index=False)
        print(f"  Saved cge_results.csv")

    if "ghosh_df" in dio_results:
        dio_results["ghosh_df"].to_csv(f"{OUTPUT_DIR}/ghosh_results.csv", index=False)
        print(f"  Saved ghosh_results.csv")

    if "dep_df" in dio_results:
        dio_results["dep_df"].to_csv(f"{OUTPUT_DIR}/ree_dependence.csv")
        print(f"  Saved ree_dependence.csv")

    # Charts
    if results.get("abm_dfs") and dio_results.get("key_sectors") is not None:
        try:
            save_all_charts(
                output_dir=f"{OUTPUT_DIR}/charts",
                abm_dfs=results["abm_dfs"],
                ghosh_df=dio_results.get("ghosh_df", pd.DataFrame()),
                key_sectors_df=dio_results["key_sectors"],
                dep_df=dio_results["dep_df"],
                cge_results_df=cge_results.get("cge_df", pd.DataFrame()),
                comparison_df=results.get("comp_df", pd.DataFrame()),
            )
        except Exception as e:
            print(f"  Chart export warning: {e}")

    print(f"\nAll outputs written to: {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UK REE DIO–CGE–ABM Impact Model"
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=["A", "B", "C", "D", "E", "all"],
        default="all",
        help="Scenario(s) to run (default: all)",
    )
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Fast test run (fewer agents, shorter horizon)")
    parser.add_argument("--mc", action="store_true",
                        help="Run Monte Carlo uncertainty analysis")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving outputs to disk")
    args = parser.parse_args()

    # Load IO data
    io_data = get_io_data()
    dfs = as_dataframes(io_data)
    print(f"Loaded UK IO data: {io_data['n_sectors']} sectors")

    # Determine scenarios
    if args.scenario == "all":
        scenario_keys = ["A", "B", "C", "D", "E"]
        if args.quick:
            scenario_keys = ["B", "C"]   # most informative for testing
    else:
        scenario_keys = [args.scenario]

    # --- Layer 1: DIO ---
    dio_results = run_dio_analysis(io_data)

    # --- Layer 2: CGE ---
    cge_results = run_cge_analysis(io_data)

    # --- Layer 3: ABM (standalone quick test) ---
    abm_results = run_abm_analysis(io_data, quick=args.quick)

    # --- Coupled simulation ---
    coupled = run_coupled(
        io_data,
        scenario_keys=scenario_keys,
        quick=args.quick,
        run_mc=args.mc,
    )

    # --- Save ---
    if not args.no_save:
        save_outputs(coupled, dio_results, cge_results)

    banner("SIMULATION COMPLETE")
    print(f"\nScenarios run: {', '.join(scenario_keys)}")
    if coupled.get("comp_df") is not None:
        print("\nFinal scenario comparison:")
        print(coupled["comp_df"].round(2).to_string())


if __name__ == "__main__":
    main()
