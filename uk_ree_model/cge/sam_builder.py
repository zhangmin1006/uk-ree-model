"""
sam_builder.py
==============
Social Accounting Matrix (SAM) builder for the UK CGE model.

A SAM is a square matrix recording all monetary flows between accounts.
Rows = receipts, Columns = expenditures.

UK SAM account structure:
  Production accounts (n sectors)
  Factor accounts:     Labour, Capital
  Household account
  Government account
  Rest of World (RoW)
  Capital/Savings account

Primary data sources (real implementation):
  - ONS Supply & Use Tables (intermediate flows, final demand)
  - ONS Sector Accounts (factor incomes)
  - HMRC Trade Statistics (imports/exports)
  - OBR Economic Fiscal Outlook (government budget)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class SAMBuilder:
    """
    Builds and validates a Social Accounting Matrix for CGE calibration.

    Parameters
    ----------
    io_data : dict  Output from data.uk_io_synthetic.get_io_data().
    """

    def __init__(self, io_data: dict):
        self.n = io_data["n_sectors"]
        self.sectors = io_data["sector_names"]
        self.A = io_data["A"]
        self.x = io_data["x"]
        self.y = io_data["y"]
        self.emp_coeff = io_data["employment_coeff"]
        self._build_sam()

    def _build_sam(self):
        """Construct the full SAM from IO data."""
        n = self.n
        A = self.A
        x = self.x
        y = self.y

        # --- Intermediate transactions Z (n x n) ---
        Z = A * x[np.newaxis, :]   # Z_ij = a_ij * x_j

        # --- Final demand decomposition ---
        # Negative y sectors (e.g. REE, Steel) are net-import sectors: their
        # domestic output is entirely absorbed as intermediate inputs and any
        # net final demand is met via imports.  Household/government/investment
        # demand cannot be negative, so we use max(y, 0) for expenditure
        # decomposition.  The Leontief identity is preserved in the IO layer
        # (y exact), while the SAM expenditure accounts stay economically valid.
        y_pos = np.maximum(y, 0.0)
        y_hh  = y_pos * 0.60
        y_gov = y_pos * 0.15
        y_inv = y_pos * 0.10
        y_exp = y_pos * 0.15

        # --- Value added (factor payments) ---
        total_intermediate = Z.sum(axis=0)  # sum of intermediate inputs per sector
        VA = x - total_intermediate          # value added = output - intermediate cost
        VA = np.maximum(VA, x * 0.30)        # floor: at least 30% VA (data consistency)

        # Split VA: 65% labour, 35% capital (ONS factor shares)
        labour_income = VA * 0.65
        capital_income = VA * 0.35

        # --- Imports ---
        import_share = 0.18   # UK imports ≈ 18% of GDP
        imports = x * import_share * 0.3   # per sector, approximate

        # --- Government receipts ---
        vat_rate = 0.20
        income_tax_rate = 0.22
        corp_tax_rate = 0.25
        gov_revenue = (
            vat_rate * y_hh.sum()
            + income_tax_rate * labour_income.sum()
            + corp_tax_rate * capital_income.sum()
        )
        gov_transfers_to_hh = gov_revenue * 0.40  # 40% of tax revenue as transfers

        # --- Household income ---
        hh_income = (
            labour_income.sum()
            + capital_income.sum() * 0.30  # households receive 30% of capital income
            + gov_transfers_to_hh
        )
        hh_savings_rate = 0.08
        hh_savings = hh_income * hh_savings_rate
        hh_consumption = hh_income - hh_savings

        # --- Store key SAM aggregates ---
        self.Z = Z
        self.VA = VA
        self.labour_income = labour_income
        self.capital_income = capital_income
        self.y_hh = y_hh
        self.y_gov = y_gov
        self.y_inv = y_inv
        self.y_exp = y_exp
        self.imports = imports
        self.hh_income = hh_income
        self.hh_savings = hh_savings
        self.gov_revenue = gov_revenue
        self.gov_transfers_to_hh = gov_transfers_to_hh

        # --- Value-added coefficients (for CGE calibration) ---
        self.va_coeff = VA / (x + 1e-12)
        self.labour_coeff = labour_income / (x + 1e-12)
        self.capital_coeff = capital_income / (x + 1e-12)

    def calibrate_cge_params(self) -> dict:
        """
        Extract CGE calibration parameters from the SAM.

        Returns
        -------
        dict with factor prices, shares, and base-year quantities.
        """
        # Normalise factor prices to 1 at base year
        w0 = 1.0       # base wage (normalised)
        r0 = 1.0       # base capital rental rate
        p0 = np.ones(self.n)   # base output prices

        # Factor market totals
        L_total = self.labour_income.sum() / w0    # aggregate labour (physical units)
        K_total = self.capital_income.sum() / r0   # aggregate capital

        # Sector labour and capital quantities
        L_sector = self.labour_income / w0
        K_sector = self.capital_income / r0

        # Household expenditure shares (Cobb-Douglas)
        mu = self.y_hh / (self.y_hh.sum() + 1e-12)   # shares sum to 1

        return {
            "w0": w0,
            "r0": r0,
            "p0": p0,
            "L_total": L_total,
            "K_total": K_total,
            "L_sector": L_sector,
            "K_sector": K_sector,
            "mu": mu,                        # household expenditure shares
            "va_coeff": self.va_coeff,
            "labour_coeff": self.labour_coeff,
            "capital_coeff": self.capital_coeff,
            "y_hh": self.y_hh,
            "y_gov": self.y_gov,
            "y_exp": self.y_exp,
            "imports": self.imports,
            "gov_revenue": self.gov_revenue,
        }

    def check_balance(self, tol: float = 0.05) -> pd.DataFrame:
        """
        Check SAM row-column balance (receipts = expenditures for each account).

        For a balanced SAM, each account's row sum = column sum.
        """
        # Simplified balance check for production accounts
        x_demand_side = (
            self.Z.sum(axis=1)       # sales to intermediate use
            + self.y_hh              # household final demand
            + self.y_gov             # government demand
            + self.y_inv             # investment
            + self.y_exp             # exports
        )

        x_supply_side = self.x + self.imports

        balance = x_demand_side - x_supply_side
        rel_balance = balance / (x_supply_side + 1e-12)

        df = pd.DataFrame({
            "sector": self.sectors,
            "receipts (supply)": x_supply_side,
            "expenditures (demand)": x_demand_side,
            "imbalance_£bn": balance,
            "imbalance_pct": rel_balance * 100,
            "balanced": np.abs(rel_balance) < tol,
        }).set_index("sector")

        n_unbalanced = (~df["balanced"]).sum()
        if n_unbalanced > 0:
            import warnings
            warnings.warn(
                f"{n_unbalanced} SAM accounts are out of balance by >{tol*100:.0f}%. "
                "Consider RAS adjustment before CGE calibration.",
                UserWarning,
            )
        return df

    def ras_adjustment(self, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
        """
        RAS bi-proportional adjustment to balance the intermediate transaction matrix Z.

        Iteratively scales rows and columns of Z until row sums match supply totals
        and column sums match demand totals.
        """
        Z = self.Z.copy()
        row_target = self.x - self.VA          # intermediate output supply by sector
        col_target = self.x - self.VA          # intermediate input demand by sector
        row_target = np.maximum(row_target, 0)
        col_target = np.maximum(col_target, 0)

        for iteration in range(max_iter):
            # Row scaling
            row_sums = Z.sum(axis=1)
            r = np.where(row_sums > 0, row_target / row_sums, 1.0)
            Z = Z * r[:, np.newaxis]

            # Column scaling
            col_sums = Z.sum(axis=0)
            s = np.where(col_sums > 0, col_target / col_sums, 1.0)
            Z = Z * s[np.newaxis, :]

            # Check convergence
            err = max(
                np.abs(Z.sum(axis=1) - row_target).max(),
                np.abs(Z.sum(axis=0) - col_target).max(),
            )
            if err < tol:
                break

        self.Z_balanced = Z
        return Z

    def to_dataframe(self) -> pd.DataFrame:
        """Return intermediate transaction matrix as labelled DataFrame."""
        return pd.DataFrame(self.Z, index=self.sectors, columns=self.sectors)

    def print_summary(self):
        """Print key SAM aggregates."""
        print("=== UK SAM Summary (£bn) ===")
        print(f"Total GDP (value added):     {self.VA.sum():.1f}")
        print(f"  Labour income:             {self.labour_income.sum():.1f}")
        print(f"  Capital income:            {self.capital_income.sum():.1f}")
        print(f"Total household consumption: {self.y_hh.sum():.1f}")
        print(f"Government revenue:          {self.gov_revenue:.1f}")
        print(f"Exports:                     {self.y_exp.sum():.1f}")
        print(f"Imports:                     {self.imports.sum():.1f}")
        print(f"Household savings:           {self.hh_savings:.1f}")
