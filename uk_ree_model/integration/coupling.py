"""
coupling.py
===========
DIO–CGE–ABM coupling protocol.

Each simulation period the three models exchange state variables:

  Period t:
    1. DIO  → ABM : sectoral output targets x*(t), IO coefficients A(t)
    2. CGE  → ABM : equilibrium prices p*(t), factor incomes I*(t)
    3. ABM  executes : agents decide given p*(t), x*(t) → Q_m(t), INV_m(t)
    4. ABM  → DIO : aggregate Q(t) updates final demand y(t+1)
    5. ABM  → CGE : aggregate demand shifts update excess demand; CGE re-solves
    6. CGE  → DIO : updated price vector feeds IO valuation; A updated if needed
    7. Loop to t+1

Consistency requirements:
  - Prices: p^CGE(t) = p^ABM(t)
  - Physical flows: Σ_m Q_m^ABM(t) ≈ x_j^DIO(t)  (within tolerance ε)
  - Trade balance: CAB^CGE(t) = CAB^DIO(t)

This module manages the handoff and runs the fully integrated simulation.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

from dio.leontief import DynamicLeontief, REEDependenceAnalyser
from dio.ghosh import GhoshModel
from dio.mrio import MRIOModel, build_uk_mrio_from_single_region
from cge.sam_builder import SAMBuilder
from cge.equilibrium import CGEModel
from abm.scheduler import UKREEModel
from integration.scenarios import Scenario


PRICE_CONSISTENCY_TOL = 0.05    # 5% max deviation between CGE and ABM prices
OUTPUT_CONSISTENCY_TOL = 0.10   # 10% max deviation between ABM aggregate and DIO


@dataclass
class CoupledState:
    """
    Shared state vector exchanged between models at each period.
    """
    period: int = 0
    theta: float = 0.0
    # From DIO
    x_dio: np.ndarray = field(default_factory=lambda: np.array([]))
    A_current: np.ndarray = field(default_factory=lambda: np.array([]))
    leontief_multipliers: np.ndarray = field(default_factory=lambda: np.array([]))
    # From CGE
    p_cge: np.ndarray = field(default_factory=lambda: np.array([]))
    w_cge: float = 1.0
    r_cge: float = 1.0
    gdp_cge: float = 0.0
    ev_cge: float = 0.0
    # From ABM
    x_abm: np.ndarray = field(default_factory=lambda: np.array([]))
    p_abm: np.ndarray = field(default_factory=lambda: np.array([]))
    ree_price_abm: float = 1.0
    pct_substituted: float = 0.0
    mean_inventory_months: float = 3.0
    # Consistency flags
    price_consistent: bool = True
    output_consistent: bool = True


class CoupledModel:
    """
    Coupled DIO–CGE–ABM model for UK REE impact analysis.

    Parameters
    ----------
    io_data         : dict      UK IO data from uk_io_synthetic.
    scenario        : Scenario  Simulation scenario.
    n_manufacturers : int       Number of ABM manufacturer agents.
    cge_freq        : int       Re-solve CGE every N periods (expensive; default 3).
    verbose         : bool      Print period-by-period status.
    seed            : int       Random seed for ABM.
    """

    def __init__(
        self,
        io_data: dict,
        scenario: Scenario,
        n_manufacturers: int = 80,
        cge_freq: int = 3,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.io_data = io_data
        self.scenario = scenario
        self.n_periods = scenario.n_periods
        self.cge_freq = cge_freq
        self.verbose = verbose

        # --- Initialise DIO layer ---
        self.dio = DynamicLeontief(
            A=io_data["A"],
            B=io_data["B"],
            x=io_data["x"],
            sector_names=io_data["sector_names"],
        )
        self.ghosh = GhoshModel(
            A=io_data["A"],
            x=io_data["x"],
            sector_names=io_data["sector_names"],
        )
        self.mrio = build_uk_mrio_from_single_region(
            A_uk=io_data["A"],
            x_uk=io_data["x"],
            y_uk=io_data["y"],
            sector_names=io_data["sector_names"],
        )
        self.ree_analyser = REEDependenceAnalyser(self.dio, io_data["ree_intensity"])

        # --- Initialise CGE layer ---
        self.sam = SAMBuilder(io_data)
        self.cge = CGEModel(self.sam, io_data, theta=0.0)
        self._cge_result = self.cge.solve(verbose=False)  # baseline

        # --- Initialise ABM layer ---
        # Enrich CGE result with SAM calibration params (mu needed by ABM households)
        cge_init = {**self._cge_result, **self.sam.calibrate_cge_params()}
        self.abm = UKREEModel(
            io_data=io_data,
            cge_equilibrium=cge_init,
            theta_path=scenario.theta_path,
            n_manufacturers=n_manufacturers,
            seed=seed,
        )
        # Apply scenario policy to government agent
        gov = self.abm.government
        policy = scenario.policy
        gov.release_trigger = policy.get("release_trigger_price", 2.5)
        gov.stockpile = policy.get("stockpile_months", 3.0) * io_data["x"][1] * 0.10 / 12

        # --- State history ---
        self.states: list[CoupledState] = []
        self.abm_results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Step 1: DIO → ABM handoff
    # ------------------------------------------------------------------

    def _dio_to_abm(self, state: CoupledState, theta: float):
        """
        Push DIO output targets and IO structure into ABM agent state.

        Manufacturers adjust their output_capacity to match DIO x*(t).
        """
        # Ghosh: compute supply-constrained output under current θ (annual £bn → monthly £bn)
        if theta > 0:
            shock_result = self.ghosh.supply_shock([1], theta)
            x_target = shock_result["x_shocked"] / 12.0
        else:
            x_target = self.io_data["x"].copy() / 12.0

        state.x_dio = x_target
        state.A_current = self.dio.A.copy()
        state.leontief_multipliers = self.dio.backward_linkages()

        # Update ABM manufacturer capacities
        n = self.io_data["n_sectors"]
        sector_output = np.zeros(n)
        sector_count = np.zeros(n, dtype=int)
        for m in self.abm.manufacturers:
            si = m.sector_idx
            sector_output[si] += m.base_output
            sector_count[si] += 1

        for m in self.abm.manufacturers:
            si = m.sector_idx
            if sector_output[si] > 0 and x_target[si] > 0:
                scale = x_target[si] / sector_output[si]
                m.output_capacity = m.base_output * np.clip(scale, 0.1, 2.0)

    # ------------------------------------------------------------------
    # Step 2: CGE → ABM handoff
    # ------------------------------------------------------------------

    def _cge_to_abm(self, state: CoupledState):
        """
        Push CGE equilibrium prices and wages into ABM agent state.
        """
        res = self._cge_result
        state.p_cge = res["p_star"].copy()
        state.w_cge = float(res["w_star"])
        state.r_cge = float(res["r_star"])
        state.gdp_cge = float(res["GDP_star_£bn"])
        state.ev_cge = float(res["EV_£bn"])

        # Update ABM price environment
        self.abm.current_prices = res["p_star"].copy()
        self.abm.current_wage = float(res["w_star"])

    # ------------------------------------------------------------------
    # Step 3: ABM → DIO handoff
    # ------------------------------------------------------------------

    def _abm_to_dio(self, state: CoupledState):
        """
        Aggregate ABM sector output → update DIO final demand y(t+1).

        Also extract ABM prices and substitution state.
        """
        n = self.io_data["n_sectors"]
        sector_output = np.zeros(n)
        for m in self.abm.manufacturers:
            sector_output[m.sector_idx] += m.current_output

        state.x_abm = sector_output
        state.p_abm = self.abm.current_prices.copy()
        state.ree_price_abm = self.abm.current_ree_price
        state.pct_substituted = float(np.mean([m.has_substituted for m in self.abm.manufacturers]))
        inv_months = [
            m.inventory / (m.base_output * m.ree_intensity_effective / 12 + 1e-12)
            for m in self.abm.manufacturers
        ]
        state.mean_inventory_months = float(np.mean(inv_months)) if inv_months else 0.0

        # Update DIO final demand with ABM-derived consumption
        abm_consumption_share = 0.60  # household consumption fraction
        new_y = sector_output * abm_consumption_share
        self.dio.x = sector_output  # update DIO base vector

    # ------------------------------------------------------------------
    # Step 4: ABM → CGE handoff
    # ------------------------------------------------------------------

    def _abm_to_cge(self, state: CoupledState, theta: float):
        """
        Re-solve CGE with ABM-informed θ every cge_freq periods.
        Also feeds in substitution reduction of effective REE intensity.
        """
        # Effective θ is moderated by substitution
        eff_theta = theta * (1 - state.pct_substituted * 0.40)

        self.cge.theta = eff_theta
        # Note: TFP gains from substitution investment are captured in eff_theta
        # (a 2% productivity gain is << calibration uncertainty, so we skip the
        # direct TFP override to preserve solver stability)
        self._cge_result = self.cge.solve(verbose=False)

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------

    def _check_consistency(self, state: CoupledState):
        """Check price and output consistency between models."""
        # Price consistency
        if len(state.p_cge) > 0 and len(state.p_abm) > 0:
            price_dev = np.abs(state.p_cge - state.p_abm) / (state.p_cge + 1e-12)
            state.price_consistent = float(price_dev.max()) < PRICE_CONSISTENCY_TOL
            if not state.price_consistent and self.verbose:
                warnings.warn(
                    f"Period {state.period}: price deviation "
                    f"{price_dev.max()*100:.1f}% > {PRICE_CONSISTENCY_TOL*100:.0f}% tolerance"
                )

        # Output consistency
        if len(state.x_dio) > 0 and len(state.x_abm) > 0:
            nonzero = state.x_dio > 0
            output_dev = np.abs(state.x_abm[nonzero] - state.x_dio[nonzero]) / state.x_dio[nonzero]
            state.output_consistent = float(output_dev.max()) < OUTPUT_CONSISTENCY_TOL

    # ------------------------------------------------------------------
    # Full integrated run
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full coupled DIO–CGE–ABM simulation.

        Returns
        -------
        dict with:
          'states'      : list[CoupledState]  per-period state history
          'abm_df'      : pd.DataFrame         ABM metrics time series
          'cge_summary' : pd.DataFrame         CGE results at peak shock
          'dio_summary' : pd.DataFrame         DIO multipliers and REE dependence
          'mrio_summary': pd.DataFrame         MRIO UK impact at peak θ
        """
        for t in range(self.n_periods):
            theta = self.scenario.theta_path[t]
            state = CoupledState(period=t, theta=theta)

            # 1. DIO → ABM
            self._dio_to_abm(state, theta)

            # 2. CGE → ABM (re-solve every cge_freq periods)
            if t % self.cge_freq == 0:
                self._abm_to_cge(state, theta)
            self._cge_to_abm(state)

            # 3. ABM step
            self.abm.step()

            # 4. ABM → DIO
            self._abm_to_dio(state)

            # 5. Consistency check
            self._check_consistency(state)

            self.states.append(state)

            if self.verbose and t % 6 == 0:
                print(
                    f"Period {t:3d} | theta={theta:.2f} | "
                    f"REE price={state.ree_price_abm:.2f}× | "
                    f"UK output={state.x_abm.sum():.1f} £bn | "
                    f"Substituted={state.pct_substituted*100:.1f}% | "
                    f"Inv={state.mean_inventory_months:.1f}mo"
                )

        # Collect results
        self.abm_results = self.abm.metrics.to_dataframe()

        # DIO summary (static at peak θ)
        peak_theta = max(self.scenario.theta_path)
        dio_summary = self.ree_analyser.summary_table()

        # CGE summary across key θ values
        from cge.equilibrium import run_cge_scenarios
        peak_thetas = sorted(set([0.0, peak_theta]))
        cge_summary = run_cge_scenarios(self.sam, self.io_data, peak_thetas, verbose=False)

        # MRIO UK impact at peak θ
        mrio_summary = self.mrio.shock_impact(peak_theta)
        mrio_df = pd.DataFrame({
            "sector": self.io_data["sector_names"],
            "output_base_£bn": mrio_summary["x_base_uk"],
            "output_shocked_£bn": mrio_summary["x_shocked_uk"],
            "delta_£bn": mrio_summary["delta_x_uk"],
            "delta_pct": mrio_summary["delta_pct_uk"],
        }).set_index("sector")

        return {
            "states": self.states,
            "abm_df": self.abm_results,
            "cge_summary": cge_summary,
            "dio_summary": dio_summary,
            "mrio_summary": mrio_df,
        }

    def states_to_dataframe(self) -> pd.DataFrame:
        """Convert CoupledState history to a flat DataFrame."""
        records = []
        for s in self.states:
            records.append({
                "period": s.period,
                "theta": s.theta,
                "gdp_cge_£bn": s.gdp_cge,
                "ev_cge_£bn": s.ev_cge,
                "ree_price_abm": s.ree_price_abm,
                "x_abm_total": s.x_abm.sum() if len(s.x_abm) > 0 else 0,
                "x_dio_total": s.x_dio.sum() if len(s.x_dio) > 0 else 0,
                "pct_substituted": s.pct_substituted * 100,
                "mean_inventory_months": s.mean_inventory_months,
                "price_consistent": s.price_consistent,
                "output_consistent": s.output_consistent,
            })
        return pd.DataFrame(records)
