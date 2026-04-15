"""
mrio.py
=======
Multi-Regional Input-Output (MRIO) model for UK–China–RoW REE analysis.

Block structure:
  Regions: UK (U), China (C), Rest of World (R)
  Sectors: shared n-sector classification

Full MRIO system:
  ┌ x^U ┐   ┌ I - A^UU  -A^UC  -A^UR ┐^{-1}  ┌ y^U ┐
  │ x^C │ = │  -A^CU  I-A^CC  -A^CR │       * │ y^C │
  └ x^R ┘   └  -A^RU   -A^RC  I-A^RR┘         └ y^R ┘

The Chinese export restriction shock modifies A^UC (UK intermediate
imports from China) by scaling the REE-related rows by (1 - θ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import inv
from typing import Optional

REGIONS = ["UK", "China", "RoW"]


class MRIOModel:
    """
    Three-region IO model: UK, China, Rest of World.

    Parameters
    ----------
    A_blocks : dict  Keys are ('UK','UK'), ('UK','China'), etc.
               Each value is an (n x n) direct input coefficient matrix
               A^{rs}_{ij} = intermediate imports by sector i in r from sector j in s.
    x_blocks : dict  Keys 'UK', 'China', 'RoW' — total output vectors (n,).
    y_blocks : dict  Keys 'UK', 'China', 'RoW' — final demand vectors (n,).
    sector_names : list[str]
    """

    def __init__(
        self,
        A_blocks: dict,
        x_blocks: dict,
        y_blocks: dict,
        sector_names: Optional[list] = None,
    ):
        self.n = list(x_blocks.values())[0].shape[0]
        self.n_total = self.n * len(REGIONS)
        self.sector_names = sector_names or [f"S{i}" for i in range(self.n)]
        self.A_blocks = {k: v.copy() for k, v in A_blocks.items()}
        self.x_blocks = {k: v.copy() for k, v in x_blocks.items()}
        self.y_blocks = {k: v.copy() for k, v in y_blocks.items()}
        self._build_full_system()

    def _build_full_system(self):
        """Assemble the full (3n x 3n) MRIO coefficient matrix."""
        n = self.n
        A_full = np.zeros((self.n_total, self.n_total))

        region_idx = {r: i * n for i, r in enumerate(REGIONS)}

        for (r_from, r_to), A_block in self.A_blocks.items():
            ri = region_idx[r_to]    # column region (using sector)
            rj = region_idx[r_from]  # row region (supplying sector)
            A_full[rj : rj + n, ri : ri + n] = A_block

        self.A_full = A_full
        self.I_minus_A_full = np.eye(self.n_total) - A_full
        self.L_full = inv(self.I_minus_A_full)

        # Assemble full output and demand vectors
        self.x_full = np.concatenate([self.x_blocks[r] for r in REGIONS])
        self.y_full = np.concatenate([self.y_blocks[r] for r in REGIONS])

    # ------------------------------------------------------------------
    # REE shock: modify A^{China→UK} block
    # ------------------------------------------------------------------

    def apply_chinese_export_restriction(
        self,
        theta: float,
        ree_rows: Optional[list[int]] = None,
    ) -> "MRIOModel":
        """
        Create a new MRIOModel with Chinese REE exports to UK reduced by θ.

        The restriction reduces A^{China→UK} entries for REE-related rows.

        Parameters
        ----------
        theta    : float        Restriction severity (0 = no restriction, 1 = full).
        ree_rows : list[int]    Indices of REE sectors in the sector list.
                                Defaults to [1] (REE & Critical Minerals sector).

        Returns
        -------
        MRIOModel  A new model instance with shocked coefficients.
        """
        if ree_rows is None:
            ree_rows = [1]

        shocked_blocks = {k: v.copy() for k, v in self.A_blocks.items()}

        # Scale down China→UK intermediate imports for REE sectors
        key = ("China", "UK")
        if key in shocked_blocks:
            for row in ree_rows:
                shocked_blocks[key][row, :] *= (1 - theta)
        else:
            # Build a zero block and apply shock
            shocked_blocks[key] = np.zeros((self.n, self.n))

        return MRIOModel(
            A_blocks=shocked_blocks,
            x_blocks=self.x_blocks,
            y_blocks=self.y_blocks,
            sector_names=self.sector_names,
        )

    def solve(self, y_full: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve x = L * y for given (or base) final demand."""
        y = y_full if y_full is not None else self.y_full
        return self.L_full @ y

    def shock_impact(
        self,
        theta: float,
        ree_rows: Optional[list[int]] = None,
    ) -> dict:
        """
        Compare baseline and shocked output for UK sectors.

        Returns
        -------
        dict with:
          'x_base_uk'    : np.ndarray (n,)  UK output at baseline
          'x_shocked_uk' : np.ndarray (n,)  UK output post-shock
          'delta_x_uk'   : np.ndarray (n,)  Change in UK output (£bn)
          'delta_pct_uk' : np.ndarray (n,)  % change in UK output
          'total_loss'   : float             Sum of UK output losses (£bn)
        """
        x_base = self.solve()
        uk_base = x_base[: self.n]

        shocked_model = self.apply_chinese_export_restriction(theta, ree_rows)
        x_shocked = shocked_model.solve()
        uk_shocked = x_shocked[: self.n]

        delta = uk_shocked - uk_base
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_pct = np.where(uk_base > 0, delta / uk_base * 100, 0.0)

        return {
            "x_base_uk": uk_base,
            "x_shocked_uk": uk_shocked,
            "delta_x_uk": delta,
            "delta_pct_uk": delta_pct,
            "total_loss": delta.sum(),
            "theta": theta,
        }

    def gvc_exposure(self) -> pd.DataFrame:
        """
        UK Global Value Chain (GVC) exposure via China for each sector.

        Measures the share of UK output that depends on Chinese intermediates.
        """
        n = self.n
        # Extract China→UK block from the Leontief inverse
        # L_full block (UK rows, China cols)
        uk_rows = slice(0, n)
        cn_cols = slice(n, 2 * n)
        L_uk_cn = self.L_full[uk_rows, cn_cols]

        # GVC exposure = row sum of L^{UK,China} weighted by Chinese output share
        gvc = L_uk_cn.sum(axis=1)

        return pd.DataFrame({
            "sector": self.sector_names,
            "gvc_china_exposure": gvc,
            "uk_output_£bn": self.x_blocks["UK"],
            "exposure_£bn": gvc * self.x_blocks["UK"],
        }).set_index("sector")

    def multi_scenario_summary(
        self,
        thetas: list[float],
        ree_rows: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Run and tabulate impact results across multiple θ scenarios."""
        records = []
        for theta in thetas:
            result = self.shock_impact(theta, ree_rows)
            for i, s in enumerate(self.sector_names):
                records.append({
                    "theta": theta,
                    "sector": s,
                    "output_base": result["x_base_uk"][i],
                    "output_shocked": result["x_shocked_uk"][i],
                    "delta_£bn": result["delta_x_uk"][i],
                    "delta_pct": result["delta_pct_uk"][i],
                })
        return pd.DataFrame(records)


def build_uk_mrio_from_single_region(
    A_uk: np.ndarray,
    x_uk: np.ndarray,
    y_uk: np.ndarray,
    china_ree_share: float = 0.70,
    sector_names: Optional[list] = None,
) -> MRIOModel:
    """
    Construct a stylised 3-region MRIO from a single-region UK IO table.

    China and RoW are approximated as scaled versions of the UK matrix,
    with China's REE sector share reflecting USGS/BGS data.

    Parameters
    ----------
    A_uk             : np.ndarray (n x n)  UK domestic IO coefficients.
    x_uk             : np.ndarray (n,)     UK total output.
    y_uk             : np.ndarray (n,)     UK final demand.
    china_ree_share  : float               China's share of global REE supply (0–1).

    Returns
    -------
    MRIOModel
    """
    n = A_uk.shape[0]

    # China: REE sector is much larger; other sectors scaled to reflect size
    scale_china = 7.0   # China GDP roughly 7× UK
    x_china = x_uk * scale_china
    x_china[1] *= (china_ree_share / 0.05)  # REE sector ~69% global = 14× UK share
    y_china = y_uk * scale_china

    # RoW: remaining global output
    scale_row = 25.0
    x_row = x_uk * scale_row
    y_row = y_uk * scale_row

    # Off-diagonal A blocks: UK imports from China
    # REE-intensive sectors import significant share from China
    A_china_to_uk = np.zeros((n, n))
    # REE sector (row 1): China supplies ~70% of UK REE needs
    A_china_to_uk[1, :] = A_uk[1, :] * 0.70  # China supplies 70% of REE inputs
    A_china_to_uk[4, :] = A_uk[4, :] * 0.05  # Some steel/metals from China
    A_china_to_uk[5, :] = A_uk[5, :] * 0.08  # Electronics components

    # UK exports to China (small)
    A_uk_to_china = A_uk * 0.02

    # RoW trade (small cross flows)
    A_row_to_uk = np.zeros((n, n))
    A_row_to_uk[1, :] = A_uk[1, :] * 0.20   # 20% of REE from RoW (Australia, US)
    A_uk_to_row = A_uk * 0.03

    A_china_to_row = np.zeros((n, n))
    A_row_to_china = np.zeros((n, n))

    # Domestic blocks
    A_uk_dom = A_uk * 0.75       # 75% of inputs sourced domestically
    A_china_dom = A_uk * 0.85
    A_row_dom = A_uk * 0.80

    A_blocks = {
        ("UK", "UK"):       A_uk_dom,
        ("China", "UK"):    A_china_to_uk,
        ("RoW", "UK"):      A_row_to_uk,
        ("UK", "China"):    A_uk_to_china,
        ("China", "China"): A_china_dom,
        ("RoW", "China"):   A_row_to_china,
        ("UK", "RoW"):      A_uk_to_row,
        ("China", "RoW"):   A_china_to_row,
        ("RoW", "RoW"):     A_row_dom,
    }

    x_blocks = {"UK": x_uk, "China": x_china, "RoW": x_row}
    y_blocks = {"UK": y_uk, "China": y_china, "RoW": y_row}

    return MRIOModel(A_blocks, x_blocks, y_blocks, sector_names)
