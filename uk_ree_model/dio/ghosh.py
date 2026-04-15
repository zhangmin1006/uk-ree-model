"""
ghosh.py
========
Supply-side (Ghosh) Input-Output model for REE supply shock propagation.

The Ghosh (1958) output-allocation model is the supply-side dual of Leontief:
    x' = v' (I - G)^{-1}

Where G_{ij} = z_{ij} / x_i  (output allocation coefficients).

Because REE disruptions originate on the supply side (export controls,
production quotas), the Ghosh propagation is the primary tool for
estimating downstream output losses.

Hybrid approach (Leontief + Ghosh):
  - Leontief captures final-demand multipliers
  - Ghosh captures supply-constrained propagation from REE bottleneck
  Both are run together in scenario analysis for cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import inv
from typing import Optional


class GhoshModel:
    """
    Ghosh supply-driven IO model.

    Parameters
    ----------
    A            : np.ndarray (n x n)  Direct input coefficient matrix (Leontief).
    x            : np.ndarray (n,)     Base-year total output vector.
    sector_names : list[str]
    """

    def __init__(
        self,
        A: np.ndarray,
        x: np.ndarray,
        sector_names: Optional[list] = None,
    ):
        self.A = A.copy()
        self.x = x.copy()
        self.n = A.shape[0]
        self.sector_names = sector_names or [f"S{i}" for i in range(self.n)]
        self._build_ghosh_matrix()

    def _build_ghosh_matrix(self):
        """
        Build output allocation matrix G from A and x.

        z_{ij} = a_{ij} * x_j   (transaction value, intermediate flow)
        g_{ij} = z_{ij} / x_i   (fraction of i's output going to j)

        Note: in a single-region IO table the A matrix mixes domestic and import
        flows, so for net-import sectors (e.g. REE) the row sum of G can exceed 1
        and v[i] = x[i] - Z[i,:].sum() can be negative.  Supply shocks are
        therefore applied as output-based shocks (see supply_shock) rather than
        primary-input shocks, avoiding sign errors from negative v[i].
        """
        # Reconstruct transaction matrix Z from A
        self.Z = self.A * self.x[np.newaxis, :]   # Z_ij = A_ij * x_j

        # Output allocation matrix: G_ij = Z_ij / x_i
        with np.errstate(divide="ignore", invalid="ignore"):
            self.G = np.where(
                self.x[:, np.newaxis] > 0,
                self.Z / self.x[:, np.newaxis],
                0.0,
            )

        # Primary inputs (value added) — may be negative for net-import sectors
        self.v = self.x - self.Z.sum(axis=1)  # v_i = x_i - sum_j z_{ij}

        # Ghosh inverse
        I = np.eye(self.n)
        self.I_minus_G = I - self.G
        self.H = inv(self.I_minus_G)   # Ghosh inverse = (I - G)^{-1}

        # Baseline Ghosh output: x_ghosh = v @ H
        # For sectors with negative v, x_ghosh may differ from self.x.
        # All supply shocks compare against x_ghosh (not self.x) for consistency.
        self.x_ghosh = self.v @ self.H

    # ------------------------------------------------------------------
    # Supply shock propagation
    # ------------------------------------------------------------------

    def supply_shock(
        self,
        shocked_sector_indices: list[int],
        theta: float,
    ) -> dict:
        """
        Propagate a supply shock (θ reduction in primary inputs) through Ghosh.

        The shock reduces primary inputs (value added / supply capacity) of
        sector k by fraction θ, triggering downstream output losses.

        Parameters
        ----------
        shocked_sector_indices : list[int]  Indices of sectors receiving shock.
        theta                  : float      Shock magnitude (0 = no shock, 1 = complete).

        Returns
        -------
        dict with keys:
          'x_shocked'      : np.ndarray (n,)  Post-shock output
          'delta_x'        : np.ndarray (n,)  Output change (negative)
          'delta_x_pct'    : np.ndarray (n,)  % output change
          'gdp_loss_£bn'   : float             Total output loss (£bn)
          'va_coeff'       : np.ndarray (n,)  Value-added coefficients
        """
        # Output-based shock: set x_shocked[k] = (1-theta) * x[k]
        #
        # For each shocked sector k we derive Δv_k so that the Ghosh solution
        # gives the desired output x'[k] = (1-theta)*x[k], regardless of the
        # sign of v[k].  From x = v H (row vector form):
        #   x'[k] = x[k] + Δv_k * H[k, k]
        #   Δv_k  = -theta * x[k] / H[k, k]
        #
        # This is sign-correct even when v[k] < 0 (net-import sectors like REE).
        v_shocked = self.v.copy()
        for k in shocked_sector_indices:
            h_kk = self.H[k, k]
            if abs(h_kk) > 1e-12:
                v_shocked[k] = self.v[k] - theta * self.x[k] / h_kk
            else:
                v_shocked[k] = self.v[k] * (1 - theta)   # fallback

        # Ghosh: x' = v' H  (compare against Ghosh baseline x_ghosh, not self.x)
        x_shocked = v_shocked @ self.H
        delta_x = x_shocked - self.x_ghosh
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_x_pct = np.where(self.x > 0, delta_x / self.x * 100, 0.0)

        return {
            "x_shocked": self.x_ghosh + delta_x,   # absolute post-shock output
            "delta_x": delta_x,
            "delta_x_pct": delta_x_pct,
            "gdp_loss_£bn": delta_x.sum(),
            "v_shocked": v_shocked,
        }

    def multi_theta_analysis(
        self,
        shocked_sector_indices: list[int],
        thetas: list[float],
    ) -> pd.DataFrame:
        """
        Run supply shock analysis across multiple θ values.

        Returns a DataFrame with output losses by sector and θ scenario.
        """
        records = []
        for theta in thetas:
            result = self.supply_shock(shocked_sector_indices, theta)
            for i, s in enumerate(self.sector_names):
                records.append({
                    "theta": theta,
                    "sector": s,
                    "output_base_£bn": self.x[i],
                    "output_shocked_£bn": result["x_shocked"][i],
                    "delta_x_£bn": result["delta_x"][i],
                    "delta_x_pct": result["delta_x_pct"][i],
                })
        return pd.DataFrame(records)

    def output_multipliers(self) -> np.ndarray:
        """Row sums of Ghosh inverse: forward supply multipliers."""
        return self.H.sum(axis=1)

    def sensitivity_of_supply(self) -> np.ndarray:
        """Column sums of Ghosh inverse: sensitivity of total supply to sector j."""
        return self.H.sum(axis=0)

    # ------------------------------------------------------------------
    # REE-specific methods
    # ------------------------------------------------------------------

    def ree_supply_bottleneck(
        self,
        ree_intensity: np.ndarray,
        theta: float,
        ree_sector_idx: int = 1,
        china_import_share: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute sector-level output loss from REE supply restriction.

        Combines:
          1. Ghosh propagation from REE sector shock
          2. Sector-level REE intensity weighting
          3. (Optional) China import share to scale effective shock

        Parameters
        ----------
        ree_intensity      : np.ndarray (n,)  REE input per £ output.
        theta              : float            Export restriction severity.
        ree_sector_idx     : int              IO index of REE sector.
        china_import_share : np.ndarray (n,)  Share of REE imports from China.
        """
        # Effective shock accounts for China's share of UK REE supply
        if china_import_share is not None:
            effective_theta = theta * china_import_share
        else:
            effective_theta = np.full(self.n, theta)

        # Direct output loss via Ghosh from REE sector supply cut
        ghosh_result = self.supply_shock([ree_sector_idx], theta)

        # Weight by REE intensity (sectors with high REE use suffer more)
        intensity_weight = ree_intensity / (ree_intensity.sum() + 1e-12)
        total_ghosh_loss = ghosh_result["delta_x"].sum()
        sector_loss = intensity_weight * np.abs(total_ghosh_loss) * effective_theta

        # Price-impact component: sectors pay more for scarce REE
        # (REE price elasticity calibrated from 2025 observed spikes)
        price_elasticity = 0.15   # % output reduction per 100% price rise
        ree_price_multiplier = 1 / (1 - theta + 1e-6) - 1   # implied price rise
        price_impact = ree_intensity * price_elasticity * ree_price_multiplier * self.x

        total_sector_loss = sector_loss + price_impact

        df = pd.DataFrame({
            "sector": self.sector_names,
            "ree_intensity": ree_intensity,
            "china_import_share": china_import_share if china_import_share is not None else effective_theta,
            "effective_theta": effective_theta,
            "ghosh_output_loss_£bn": ghosh_result["delta_x"],
            "ree_intensity_loss_£bn": -sector_loss,
            "price_impact_loss_£bn": -price_impact,
            "total_loss_£bn": -(sector_loss + price_impact),
            "output_base_£bn": self.x,
            "loss_pct": -(sector_loss + price_impact) / (self.x + 1e-12) * 100,
        }).set_index("sector")

        return df

    def hhi_concentration(self, import_shares_by_country: dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute Herfindahl-Hirschman Index for REE import concentration by sector.

        Parameters
        ----------
        import_shares_by_country : dict  country → np.ndarray(n,) of import shares.

        Returns
        -------
        hhi : np.ndarray (n,)  HHI value per sector (0–10000 scale).
        """
        hhi = np.zeros(self.n)
        for country, shares in import_shares_by_country.items():
            hhi += (shares * 100) ** 2
        return hhi

    def summary(self, ree_intensity: np.ndarray, theta: float = 0.75) -> pd.DataFrame:
        """Print a readable supply-shock summary."""
        result = self.supply_shock([1], theta)  # default: shock REE sector
        df = pd.DataFrame({
            "output_base_£bn": self.x,
            "output_shocked_£bn": result["x_shocked"],
            "delta_x_£bn": result["delta_x"],
            "pct_change": result["delta_x_pct"],
            "forward_multiplier": self.output_multipliers(),
        }, index=self.sector_names)
        return df
