"""
trade.py
========
Armington trade module for the UK CGE model.

Armington (1969) assumption: domestically produced and imported goods are
imperfect substitutes, differentiated by country of origin.

Import demand:   M_i = (alpha_m / alpha_d)^σ * (p_d / p_m)^σ * D_i
Export supply:   E_i = E_i^0 * (p_x / p_x^0)^η_E

REE-specific:
  Very low Armington elasticity (σ_A ≈ 0.3) because Chinese and non-Chinese
  REE are NOT freely substitutable in the short run — supply chains are
  locked in, processing technology is specialised.

  Calibration from 2025 price spike data:
    Observed: +598% price rise (yttrium) with ~30% quantity adjustment
    Implies σ_A ≈ 0.3 / 1.99 ≈ 0.15 for most-affected REEs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeParams:
    """Armington trade parameters for one sector."""
    sector: str
    sigma_m: float      # Armington substitution elasticity (imports vs domestic)
    sigma_e: float      # Export supply elasticity
    alpha_m: float      # Import share parameter
    alpha_d: float      # Domestic share parameter (= 1 - alpha_m at base)
    tariff_rate: float = 0.0   # Ad valorem tariff on imports
    export_subsidy: float = 0.0


class ArmingtonTrade:
    """
    Armington trade module for a multi-sector UK economy.

    Tracks import demand, export supply, and trade balance by sector.
    """

    def __init__(
        self,
        sector_names: list[str],
        sigma_m: np.ndarray,
        sigma_e: np.ndarray,
        import_shares_base: np.ndarray,
        p_dom_base: Optional[np.ndarray] = None,
        p_imp_base: Optional[np.ndarray] = None,
        p_exp_base: Optional[np.ndarray] = None,
        imports_base: Optional[np.ndarray] = None,
        exports_base: Optional[np.ndarray] = None,
    ):
        self.sectors = sector_names
        self.n = len(sector_names)
        self.sigma_m = sigma_m
        self.sigma_e = sigma_e
        self.import_shares_base = import_shares_base

        # Base prices (normalised to 1)
        self.p_dom0 = p_dom_base if p_dom_base is not None else np.ones(self.n)
        self.p_imp0 = p_imp_base if p_imp_base is not None else np.ones(self.n)
        self.p_exp0 = p_exp_base if p_exp_base is not None else np.ones(self.n)

        # Base quantities
        self.M0 = imports_base if imports_base is not None else np.zeros(self.n)
        self.E0 = exports_base if exports_base is not None else np.zeros(self.n)

        # Armington share parameters calibrated from base-year data
        self.alpha_m = import_shares_base
        self.alpha_d = 1.0 - import_shares_base

    # ------------------------------------------------------------------
    # Armington composite price
    # ------------------------------------------------------------------

    def composite_price(
        self,
        p_dom: np.ndarray,
        p_imp: np.ndarray,
        tariff: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Armington composite (Armington aggregator) price index.

        P_i = [α_d^σ p_d^{1-σ} + α_m^σ (p_m(1+t))^{1-σ}]^{1/(1-σ)}
        """
        if tariff is None:
            tariff = np.zeros(self.n)
        p_m_tariff = p_imp * (1 + tariff)
        sigma = self.sigma_m
        P = np.where(
            np.abs(sigma - 1.0) < 1e-6,
            # Cobb-Douglas limit
            p_dom ** self.alpha_d * p_m_tariff ** self.alpha_m,
            # CES
            (
                self.alpha_d ** sigma * p_dom ** (1 - sigma)
                + self.alpha_m ** sigma * p_m_tariff ** (1 - sigma)
            ) ** (1.0 / (1 - sigma)),
        )
        return P

    # ------------------------------------------------------------------
    # Import demand
    # ------------------------------------------------------------------

    def import_demand(
        self,
        p_dom: np.ndarray,
        p_imp: np.ndarray,
        Q_composite: np.ndarray,
        tariff: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Import demand from Armington cost minimisation:

        M_i = [α_m / (1-α_m)]^σ * (p_d / p_m)^σ * D_i

        Parameters
        ----------
        p_dom        : domestic prices
        p_imp        : import prices (before tariff)
        Q_composite  : total Armington demand (domestic + imported composite)
        tariff       : import tariff rates

        Returns
        -------
        M : np.ndarray (n,) import quantities
        """
        if tariff is None:
            tariff = np.zeros(self.n)
        p_m = p_imp * (1 + tariff)
        P = self.composite_price(p_dom, p_imp, tariff)
        sigma = self.sigma_m
        M = (self.alpha_m ** sigma) * (P / p_m) ** sigma * Q_composite
        return M

    def domestic_demand(
        self,
        p_dom: np.ndarray,
        p_imp: np.ndarray,
        Q_composite: np.ndarray,
        tariff: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Domestic demand from Armington cost minimisation."""
        if tariff is None:
            tariff = np.zeros(self.n)
        p_m = p_imp * (1 + tariff)
        P = self.composite_price(p_dom, p_imp, tariff)
        sigma = self.sigma_m
        D = (self.alpha_d ** sigma) * (P / p_dom) ** sigma * Q_composite
        return D

    # ------------------------------------------------------------------
    # Export supply
    # ------------------------------------------------------------------

    def export_supply(
        self,
        p_exp: np.ndarray,
        export_subsidy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Export supply from profit maximisation:

        E_i = E_i^0 * (p_x (1+sub) / p_x^0)^η_E
        """
        if export_subsidy is None:
            export_subsidy = np.zeros(self.n)
        p_x_eff = p_exp * (1 + export_subsidy)
        return self.E0 * (p_x_eff / self.p_exp0) ** self.sigma_e

    # ------------------------------------------------------------------
    # REE-specific supply shock
    # ------------------------------------------------------------------

    def ree_import_price_shock(
        self,
        theta: float,
        ree_sector_idx: int = 1,
        price_elasticity: float = -2.0,
    ) -> np.ndarray:
        """
        Compute REE import price increase under supply restriction θ.

        Price rise implied by supply reduction:
            Δ%P = -Δ%Q / ε_s  where ε_s = supply price elasticity (~2)

        Calibrated against 2025 observed spikes:
          θ=0.75 → observed Dy +168%, Y +598%, Sm +6000%
          Use moderate multiplier for aggregate REE price basket.

        Parameters
        ----------
        theta              : float  Supply restriction (0–1)
        ree_sector_idx     : int    Index of REE sector
        price_elasticity   : float  Supply price elasticity (negative sign)

        Returns
        -------
        p_imp_shocked : np.ndarray (n,)  Import prices post-shock
        """
        p_imp_shocked = self.p_imp0.copy()

        # REE price spike: supply falls by θ → price rises
        pct_quantity_fall = theta * 100
        pct_price_rise = pct_quantity_fall / abs(price_elasticity)  # % price rise

        # Apply price spike to REE sector and downstream intensive sectors
        ree_price_multiplier = 1 + pct_price_rise / 100
        p_imp_shocked[ree_sector_idx] *= ree_price_multiplier

        # Downstream sectors face higher REE embedded costs
        # Electronics, Automotive, Aerospace, Wind: partial price transmission
        downstream_sectors = [5, 6, 7, 9]   # from our 12-sector classification
        for idx in downstream_sectors:
            p_imp_shocked[idx] *= (1 + 0.20 * theta)  # 20% cost pass-through per unit θ

        return p_imp_shocked

    def trade_balance(
        self,
        p_imp: np.ndarray,
        p_exp: np.ndarray,
        M: np.ndarray,
        E: np.ndarray,
    ) -> float:
        """Compute nominal trade balance (exports minus imports, £bn)."""
        return (p_exp * E).sum() - (p_imp * M).sum()

    def summary(
        self,
        p_dom: np.ndarray,
        p_imp: np.ndarray,
        Q: np.ndarray,
    ) -> pd.DataFrame:
        """Trade summary table."""
        M = self.import_demand(p_dom, p_imp, Q)
        D = self.domestic_demand(p_dom, p_imp, Q)
        P = self.composite_price(p_dom, p_imp)
        return pd.DataFrame({
            "composite_price": P,
            "domestic_price": p_dom,
            "import_price": p_imp,
            "import_demand": M,
            "domestic_demand": D,
            "import_share": M / (M + D + 1e-12),
        }, index=self.sectors)


def build_trade_params(
    sector_names: list[str],
    ree_intensity: np.ndarray,
    import_shares: np.ndarray,
    imports_base: np.ndarray,
    exports_base: np.ndarray,
) -> ArmingtonTrade:
    """
    Build ArmingtonTrade object calibrated for UK REE analysis.

    REE-intensive sectors have low Armington elasticity (limited substitution).
    """
    n = len(sector_names)

    # Armington import elasticity: lower for REE-intensive sectors
    sigma_m = np.where(ree_intensity > 0.015, 0.20,
              np.where(ree_intensity > 0.005, 0.40, 0.80))

    # Export elasticity: moderate for all sectors
    sigma_e = np.full(n, 1.5)
    sigma_e[1] = 0.5   # REE sector: inelastic export supply (depleting resource)

    return ArmingtonTrade(
        sector_names=sector_names,
        sigma_m=sigma_m,
        sigma_e=sigma_e,
        import_shares_base=import_shares,
        imports_base=imports_base,
        exports_base=exports_base,
    )
