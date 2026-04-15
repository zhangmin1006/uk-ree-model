"""
metrics.py
==========
Metrics collector for the UK REE ABM.

Aggregates agent-level decisions each period into macro statistics
that can be compared against DIO/CGE outputs and real UK data.

Collected each period:
  - REE price, supply, demand
  - UK aggregate output (£bn) by sector
  - Employment (k FTE)
  - Inventory depletion ratio
  - Fraction of firms that have substituted
  - Network resilience
  - Government stockpile level
  - Consumer welfare proxy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scheduler import UKREEModel


class MetricsCollector:
    """
    Collects and stores per-period aggregate statistics from the ABM.

    Parameters
    ----------
    model : UKREEModel
    """

    def __init__(self, model: "UKREEModel"):
        self.model = model
        self.records: list[dict] = []

    def collect(self, step: int, theta: float):
        """Gather all metrics for the current time step."""
        model = self.model
        mfrs = model.manufacturers
        suppliers = model.suppliers
        gov = model.government
        io = model.io_data

        # --- REE market ---
        ree_supply = sum(s.current_supply for s in suppliers) + model.government_supply_release
        ree_demand = sum(m.ree_demand_target for m in mfrs)
        ree_price = model.current_ree_price

        # --- Output by sector ---
        n = io["n_sectors"]
        sector_output = np.zeros(n)
        sector_output_base = np.zeros(n)
        for m in mfrs:
            si = m.sector_idx
            sector_output[si] += m.current_output
            sector_output_base[si] += m.base_output

        total_output = sector_output.sum()
        total_output_base = sector_output_base.sum()
        output_loss = total_output_base - total_output

        # --- Employment ---
        # emp_coeff is k FTE per £bn annual output; sector_output is monthly £bn
        # → annualise output before applying coefficient
        emp_coeff = io["employment_coeff"]
        employment = (sector_output * 12 * emp_coeff).sum()
        employment_base = (sector_output_base * 12 * emp_coeff).sum()
        employment_loss = employment_base - employment

        # --- Inventory health ---
        monthly_uses = [
            m.base_output * m.ree_intensity_effective for m in mfrs
        ]
        inv_months = [
            m.inventory / (mu + 1e-12) for m, mu in zip(mfrs, monthly_uses)
        ]
        mean_inv_months = np.mean(inv_months) if inv_months else 0.0
        pct_below_reorder = np.mean([im < m.s_reorder for im, m in zip(inv_months, mfrs)])

        # --- Substitution ---
        pct_substituted = np.mean([m.has_substituted for m in mfrs]) if mfrs else 0.0
        n_substituted = sum(m.has_substituted for m in mfrs)

        # --- Government stockpile ---
        stockpile = gov.stockpile
        stockpile_release = gov.releases[-1] if gov.releases else 0.0

        # --- Welfare proxy (average utility) ---
        hh_utilities = []
        for hh in model.households:
            if hh.utility_history:
                hh_utilities.append(hh.utility_history[-1])
        mean_utility = np.mean(hh_utilities) if hh_utilities else 1.0

        # --- Sector-level output for DIO coupling ---
        sector_record = {
            f"output_{io['sector_names'][i].replace(' ', '_').replace('&', 'n')}": sector_output[i]
            for i in range(n)
        }

        record = {
            "step": step,
            "theta": theta,
            # REE market
            "ree_price": ree_price,
            "ree_supply_£bn": ree_supply,
            "ree_demand_£bn": ree_demand,
            "ree_supply_deficit": max(ree_demand - ree_supply, 0),
            # Output
            "total_output_£bn": total_output,
            "total_output_base_£bn": total_output_base,
            "output_loss_£bn": output_loss,
            "output_loss_pct": output_loss / (total_output_base + 1e-12) * 100,
            # Employment
            "employment_kFTE": employment,
            "employment_loss_kFTE": employment_loss,
            # Inventory
            "mean_inventory_months": mean_inv_months,
            "pct_firms_below_reorder": pct_below_reorder * 100,
            # Substitution
            "pct_firms_substituted": pct_substituted * 100,
            "n_firms_substituted": n_substituted,
            # Government
            "gov_stockpile_£bn": stockpile,
            "gov_release_£bn": stockpile_release,
            # Welfare
            "mean_household_utility": mean_utility,
            **sector_record,
        }
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Return full time-series of collected metrics."""
        return pd.DataFrame(self.records)

    def summary(self) -> pd.DataFrame:
        """Return a summary of key stats across the full simulation run."""
        df = self.to_dataframe()
        if df.empty:
            return df

        shocked = df[df["theta"] > 0]
        baseline = df[df["theta"] == 0]

        rows = []
        for col in ["ree_price", "total_output_£bn", "employment_kFTE",
                    "mean_inventory_months", "pct_firms_substituted"]:
            rows.append({
                "metric": col,
                "baseline_mean": baseline[col].mean() if not baseline.empty else np.nan,
                "shocked_mean": shocked[col].mean() if not shocked.empty else np.nan,
                "shocked_min": shocked[col].min() if not shocked.empty else np.nan,
                "shocked_max": shocked[col].max() if not shocked.empty else np.nan,
            })
        return pd.DataFrame(rows).set_index("metric")

    def sector_output_timeseries(self) -> pd.DataFrame:
        """Return sector-level output over time."""
        df = self.to_dataframe()
        io = self.model.io_data
        sector_cols = [
            f"output_{s.replace(' ', '_').replace('&', 'n')}"
            for s in io["sector_names"]
        ]
        available = [c for c in sector_cols if c in df.columns]
        return df[["step", "theta"] + available]

    def peak_disruption(self) -> dict:
        """Return the single worst period statistics."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        worst_idx = df["output_loss_£bn"].idxmax()
        return df.iloc[worst_idx].to_dict()
