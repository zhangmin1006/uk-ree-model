"""
equilibrium.py
==============
CGE equilibrium solver for the UK REE impact model.

Structure (supply-side + mobile labour):
  - Zero-profit:   p_j  = c_j(w, r0, p_ree)          CES cost
  - Output supply: Y_j* = Y0_j * (c_j0 / c_j*)^eta   supply response
  - Labour market: Σ L_j(Y*) = L̄                     1-equation, 1-unknown (w)
  - Capital fixed: sector-specific (short-run putty-clay)

Calibration guarantee (replication check):
  At theta=0  →  p_ree=1, c_j=1 (TFP-calibrated), Y*=Y0, L_demand=L̄, EV=0.

Solver: scalar root-finder (brentq) on the labour-market residual.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from typing import Optional

from .production import NestedCESProduction, SectorParams, build_sector_params
from .sam_builder import SAMBuilder
from .trade import ArmingtonTrade, build_trade_params

# Supply elasticity: how much output falls per 1% unit-cost increase
SUPPLY_ELASTICITY = 1.5


class CGEModel:
    """
    Computable General Equilibrium model for UK REE analysis.

    Solves for the equilibrium wage w* that clears the labour market given
    an exogenous REE import price shock theta.  Capital is treated as
    sector-specific (short-run), so the rental rate r0=1 is fixed.

    All prices, quantities and welfare are derived from w*.

    Parameters
    ----------
    sam      : SAMBuilder
    io_data  : dict   (from uk_io_synthetic.get_io_data)
    theta    : float  REE supply shock (0 = baseline, 1 = full disruption)
    """

    def __init__(self, sam: SAMBuilder, io_data: dict, theta: float = 0.0):
        self.sam = sam
        self.io_data = io_data
        self.theta = theta
        self.n = sam.n
        self.sectors = sam.sectors

        self.cge_params = sam.calibrate_cge_params()
        self.sector_params = build_sector_params(
            io_data["sector_names"],
            io_data["ree_intensity"],
            io_data["china_import_share"],
        )
        self.producers = [NestedCESProduction(sp) for sp in self.sector_params]

        self.trade = build_trade_params(
            io_data["sector_names"],
            io_data["ree_intensity"],
            io_data["china_import_share"],
            sam.imports,
            sam.y_exp,
        )

        # Baseline prices (fixed normalisations)
        self.w0 = 1.0
        self.r0 = 1.0
        self.p_ree_imp0 = 1.0
        self.p_ree_dom0 = 1.0
        self.p_nree0 = 1.0
        self.p0 = np.ones(self.n)
        self.Y0 = io_data["x"].copy()

        self._saverate = 0.08

    # ------------------------------------------------------------------
    # Calibration (run at start of every solve so sigma overrides apply)
    # ------------------------------------------------------------------

    def _calibrate(self):
        """
        Calibrate TFP and endowments given current sigma/alpha values.

        After this call:
          unit_cost_j(w0, r0, p_ree_imp0) == 1.0   for every j     ← replication
          L_bar  = Σ_j labour_coeff_j * Y0_j                       ← IO-consistent
          K_bar  = Σ_j capital_coeff_j * Y0_j
          mu_j   = y_hh_j / I_H0_disp                              ← demand shares
        """
        w0, r0 = self.w0, self.r0

        # ① TFP: unit_cost at baseline prices = 1
        # Reset TFP to 1 first so unit_cost returns the raw CES cost (not c/TFP_old)
        for prod in self.producers:
            prod.p.TFP = 1.0
            c_base = prod.unit_cost(w0, r0, self.p_ree_dom0, self.p_ree_imp0, self.p_nree0)
            prod.p.TFP = max(c_base, 1e-9)

        # ② Factor endowments from IO labour/capital coefficients
        self.L_bar = (self.sam.labour_coeff * self.Y0).sum()
        self.K_bar = (self.sam.capital_coeff * self.Y0).sum()

        # ③ Household income at baseline
        self.I_H0 = w0 * self.L_bar + r0 * self.K_bar * 0.30 + self.sam.gov_transfers_to_hh
        self.I_H0_disp = self.I_H0 * (1.0 - self._saverate)

        # ④ Expenditure shares: mu_j so that C_j = y_hh_j at baseline (p=1)
        self.mu = self.sam.y_hh / (self.I_H0_disp + 1e-12)
        self.mu_norm = self.mu / (self.mu.sum() + 1e-12)   # sum→1 for CPI

    # ------------------------------------------------------------------
    # Prices and output
    # ------------------------------------------------------------------

    def _ree_import_price(self, theta: float) -> float:
        """
        REE import price schedule.

        theta=0   →  1.00×  (baseline)
        theta=0.5 →  ~1.74×
        theta=0.75 → ~3.03×  (consistent with 2010/2023 REE episodes)
        theta=1.0 → capped at 50×
        """
        if theta <= 0.0:
            return 1.0
        return min(1.0 / max(1.0 - theta, 1e-4) ** 0.8, 50.0)

    def _unit_costs(self, w: float, p_ree_imp: float) -> np.ndarray:
        """Unit cost vector c_j(w, r0, p_ree)  — r is kept at r0=1."""
        return np.array([
            max(prod.unit_cost(w, self.r0, self.p_ree_dom0, p_ree_imp, self.p_nree0), 1e-9)
            for prod in self.producers
        ])

    def _output(self, costs: np.ndarray) -> np.ndarray:
        """
        Supply-response output.

        Y_j* = Y0_j * (1 / c_j*)^eta    [since c_j0 = 1 after TFP calibration]

        Supply contracts when c_j* > 1 (costs rise above baseline).
        Supply expands when c_j* < 1 (e.g. wages fall, costs below baseline).
        Clipped to [10%, 300%] of baseline.
        """
        return self.Y0 * np.clip(costs ** (-SUPPLY_ELASTICITY), 0.10, 3.0)

    # ------------------------------------------------------------------
    # Labour-market residual (scalar root problem)
    # ------------------------------------------------------------------

    def _labour_residual(self, w: float, p_ree_imp: float) -> float:
        """
        Labour market excess demand (normalised).

        ED_L(w) = [ Σ_j labour_coeff_j * Y_j*(w) - L̄ ] / L̄
        """
        costs = self._unit_costs(w, p_ree_imp)
        Y_star = self._output(costs)
        L_demand = (self.sam.labour_coeff * Y_star).sum()
        return (L_demand - self.L_bar) / (self.L_bar + 1e-12)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = False) -> dict:
        """
        Solve for equilibrium wage w* and derive all other quantities.

        Returns
        -------
        dict with: p_star, w_star, r_star, p_ree_imp, Y_star, Y_base,
                   delta_GDP_pct, EV_£bn, CPI, delta_employment_kFTE, …
        """
        # Re-calibrate (picks up any sigma overrides set after __init__)
        self._calibrate()

        p_ree_imp = self._ree_import_price(self.theta)

        # ── Baseline check at theta=0 ──────────────────────────────────────
        if self.theta <= 0.0:
            w_star = self.w0
            converged = True
        else:
            # Brent's method on the 1-D labour market residual
            # Bracket: w in [0.05, 5.0]
            try:
                f_lo = self._labour_residual(0.05, p_ree_imp)
                f_hi = self._labour_residual(5.00, p_ree_imp)

                if f_lo * f_hi > 0:
                    # No sign change — fall back to Newton start from w0
                    w_star = self.w0
                    converged = False
                else:
                    w_star = brentq(
                        self._labour_residual,
                        0.05, 5.00,
                        args=(p_ree_imp,),
                        xtol=1e-8,
                        maxiter=200,
                    )
                    converged = True
            except Exception as e:
                warnings.warn(f"CGE solver: {e}")
                w_star = self.w0
                converged = False

        r_star = self.r0   # capital rental fixed (sector-specific capital)

        # ── Equilibrium quantities ─────────────────────────────────────────
        costs_star = self._unit_costs(w_star, p_ree_imp)
        p_star = costs_star                           # zero-profit: p = c
        Y_star = self._output(costs_star)

        # ── GDP via factor incomes (putty-clay capital) ────────────────────
        # Labour income: w* × L_bar (mobile labour, wage adjusts)
        # Capital income: sector-specific utilisation, proportional to Y*/Y0
        labour_GDP_star = w_star * self.L_bar
        labour_GDP_base = self.w0 * self.L_bar
        capital_GDP_star = (self.sam.capital_coeff * Y_star).sum()
        capital_GDP_base = (self.sam.capital_coeff * self.Y0).sum()
        GDP_star = labour_GDP_star + capital_GDP_star
        GDP_base = labour_GDP_base + capital_GDP_base
        delta_GDP = GDP_star - GDP_base

        # ── Employment ─────────────────────────────────────────────────────
        emp_star = Y_star * self.io_data["employment_coeff"]
        emp_base = self.Y0 * self.io_data["employment_coeff"]
        delta_emp = (emp_star - emp_base).sum()

        # ── CPI (expenditure-share weighted price index) ───────────────────
        CPI = float((self.mu_norm * p_star).sum())   # = 1.0 at baseline

        # ── Equivalent Variation ───────────────────────────────────────────
        # EV = real disposable income change (base-year £bn)
        #    = I_H*_disp / CPI  -  I_H0_disp
        I_H_star = w_star * self.L_bar + r_star * self.K_bar * 0.30 + self.sam.gov_transfers_to_hh
        I_H_star_disp = I_H_star * (1.0 - self._saverate)
        EV = I_H_star_disp / (CPI + 1e-12) - self.I_H0_disp

        if verbose:
            print(
                f"theta={self.theta:.2f}  "
                f"dGDP={delta_GDP / (GDP_base + 1e-12) * 100:+.2f}%  "
                f"EV={EV:+.1f}bn  "
                f"CPI={(CPI-1)*100:+.2f}%  "
                f"w*={w_star:.4f}  "
                f"p_ree={p_ree_imp:.2f}x  "
                f"conv={converged}"
            )

        return {
            "theta": self.theta,
            "converged": converged,
            "p_star": p_star,
            "w_star": w_star,
            "r_star": r_star,
            "p_ree_imp": p_ree_imp,
            "Y_star": Y_star,
            "Y_base": self.Y0,
            "delta_Y": Y_star - self.Y0,
            "delta_Y_pct": (Y_star - self.Y0) / (self.Y0 + 1e-12) * 100,
            "GDP_base_£bn": GDP_base,
            "GDP_star_£bn": GDP_star,
            "delta_GDP_£bn": delta_GDP,
            "delta_GDP_pct": delta_GDP / (GDP_base + 1e-12) * 100,
            "EV_£bn": EV,
            "CPI": CPI,
            "employment_base_kFTE": emp_base.sum(),
            "employment_star_kFTE": emp_star.sum(),
            "delta_employment_kFTE": delta_emp,
            "sector_output": pd.Series(Y_star, index=self.sectors),
            "sector_delta_£bn": pd.Series(Y_star - self.Y0, index=self.sectors),
            "sector_delta_pct": pd.Series(
                (Y_star - self.Y0) / (self.Y0 + 1e-12) * 100,
                index=self.sectors,
            ),
        }

    def welfare_decomposition(self, result: dict) -> pd.DataFrame:
        """Decompose welfare change by sector price contribution."""
        p_star = result["p_star"]
        price_change = p_star - 1.0
        return pd.DataFrame({
            "expenditure_share": self.mu_norm,
            "price_change_pct": price_change * 100,
            "welfare_contribution_£bn": -self.mu_norm * price_change * self.I_H0_disp,
        }, index=self.sectors)


def run_cge_scenarios(
    sam: SAMBuilder,
    io_data: dict,
    thetas: list[float],
    verbose: bool = True,
) -> pd.DataFrame:
    """Run CGE model across a list of shock scenarios and return a summary DataFrame."""
    records = []
    for theta in thetas:
        model = CGEModel(sam, io_data, theta=theta)
        result = model.solve(verbose=verbose)
        records.append({
            "theta": theta,
            "scenario": f"theta={theta:.2f}",
            "delta_GDP_£bn": result["delta_GDP_£bn"],
            "delta_GDP_pct": result["delta_GDP_pct"],
            "EV_£bn": result["EV_£bn"],
            "CPI_change_pct": (result["CPI"] - 1) * 100,
            "delta_employment_kFTE": result["delta_employment_kFTE"],
            "ree_price_multiplier": result["p_ree_imp"],
            "converged": result["converged"],
        })
    return pd.DataFrame(records)
