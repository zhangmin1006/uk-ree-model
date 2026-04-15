"""
production.py
=============
Nested CES production functions for the UK CGE model.

Production structure (top-down nesting):

  Y_j  = CES(VA_j, INT_j; σ_VA)          [top level: value added vs intermediates]
  VA_j = CES(L_j, K_j; σ_L)              [labour vs capital]
  INT_j = CES(REE_j, NREE_j; σ_INT)      [REE vs non-REE intermediates]
  REE_j = CES(REE_dom_j, REE_imp_j; σ_A) [Armington: domestic vs imported REE]

Calibration:
  σ_VA   ≈ 1.0 (Cobb-Douglas between VA and INT)
  σ_L    ≈ 0.5 (limited labour-capital substitution)
  σ_INT  ≈ 0.2 (very limited REE substitution in short run)
  σ_A    ≈ 0.3 (Armington elasticity for REE trade)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SectorParams:
    """Production parameters for a single sector."""
    name: str
    alpha_va: float        # value-added share at top level
    alpha_L: float         # labour share in VA nest
    alpha_ree: float       # REE share in intermediates nest
    alpha_ree_dom: float   # domestic share in REE Armington nest
    sigma_VA: float = 1.0  # elasticity top nest (VA vs INT)
    sigma_L: float = 0.5   # elasticity VA nest (L vs K)
    sigma_INT: float = 0.2 # elasticity INT nest (REE vs NREE)
    sigma_A: float = 0.3   # Armington elasticity (dom vs imp REE)
    TFP: float = 1.0       # Total factor productivity


def ces_quantity(
    x1: float,
    x2: float,
    alpha: float,
    sigma: float,
) -> float:
    """
    Two-input CES aggregator quantity.

    Q = [α x1^ρ + (1-α) x2^ρ]^{1/ρ}   where ρ = (σ-1)/σ

    Special case σ = 1: Cobb-Douglas Q = x1^α * x2^{1-α}
    """
    if abs(sigma - 1.0) < 1e-6:
        # Cobb-Douglas limit
        return (x1 ** alpha) * (x2 ** (1 - alpha))
    rho = (sigma - 1.0) / sigma
    return (alpha * x1 ** rho + (1 - alpha) * x2 ** rho) ** (1.0 / rho)


def ces_cost(
    p1: float,
    p2: float,
    alpha: float,
    sigma: float,
) -> float:
    """
    CES unit cost function dual to the CES quantity index.

    c(p1, p2) = [α^σ p1^{1-σ} + (1-α)^σ p2^{1-σ}]^{1/(1-σ)}

    Special case σ = 1: c = p1^α * p2^{1-α}
    """
    if abs(sigma - 1.0) < 1e-6:
        return (p1 ** alpha) * (p2 ** (1 - alpha))
    return (alpha ** sigma * p1 ** (1 - sigma) + (1 - alpha) ** sigma * p2 ** (1 - sigma)) ** (1.0 / (1 - sigma))


def ces_demand(
    p1: float,
    p2: float,
    alpha: float,
    sigma: float,
    Q: float,
) -> tuple[float, float]:
    """
    Conditional factor demands from CES cost minimisation.

    x1* = α^σ (c/p1)^σ Q
    x2* = (1-α)^σ (c/p2)^σ Q
    """
    c = ces_cost(p1, p2, alpha, sigma)
    if abs(sigma - 1.0) < 1e-6:
        x1 = alpha * (c / p1) * Q
        x2 = (1 - alpha) * (c / p2) * Q
    else:
        x1 = (alpha ** sigma) * (c / p1) ** sigma * Q
        x2 = ((1 - alpha) ** sigma) * (c / p2) ** sigma * Q
    return x1, x2


class NestedCESProduction:
    """
    Nested CES production function for a single UK sector.

    Computes factor demands and unit costs given prices and output target.
    """

    def __init__(self, params: SectorParams):
        self.p = params

    def unit_cost(
        self,
        w: float,      # wage
        r: float,      # capital rental rate
        p_ree_dom: float,   # domestic REE price
        p_ree_imp: float,   # imported REE price
        p_nree: float,      # non-REE intermediate price
    ) -> float:
        """
        Compute unit production cost via the CES dual cost functions.

        Nested structure:
          c_VA  = CES_cost(w, r;  α_L,  σ_L)
          c_REE = CES_cost(p_dom, p_imp; α_dom, σ_A)       [Armington]
          c_INT = CES_cost(c_REE, p_nree; α_ree, σ_INT)
          c_Y   = CES_cost(c_VA, c_INT; α_VA, σ_VA)
        """
        c_va = ces_cost(w, r, self.p.alpha_L, self.p.sigma_L)
        c_ree = ces_cost(p_ree_dom, p_ree_imp, self.p.alpha_ree_dom, self.p.sigma_A)
        c_int = ces_cost(c_ree, p_nree, self.p.alpha_ree, self.p.sigma_INT)
        c_y = ces_cost(c_va, c_int, self.p.alpha_va, self.p.sigma_VA)
        return c_y / self.p.TFP

    def factor_demands(
        self,
        Y: float,
        w: float,
        r: float,
        p_ree_dom: float,
        p_ree_imp: float,
        p_nree: float,
    ) -> dict:
        """
        Compute all conditional factor demands for output level Y.

        Returns
        -------
        dict with keys: VA, INT, L, K, REE, NREE, REE_dom, REE_imp
        """
        # Top level: VA vs INT
        c_va = ces_cost(w, r, self.p.alpha_L, self.p.sigma_L)
        c_ree_arm = ces_cost(p_ree_dom, p_ree_imp, self.p.alpha_ree_dom, self.p.sigma_A)
        c_int = ces_cost(c_ree_arm, p_nree, self.p.alpha_ree, self.p.sigma_INT)
        c_y = ces_cost(c_va, c_int, self.p.alpha_va, self.p.sigma_VA) / self.p.TFP

        Y_eff = Y * self.p.TFP
        VA, INT = ces_demand(c_va, c_int, self.p.alpha_va, self.p.sigma_VA, Y_eff)

        # VA nest: L vs K
        L, K = ces_demand(w, r, self.p.alpha_L, self.p.sigma_L, VA)

        # INT nest: REE vs NREE
        REE, NREE = ces_demand(c_ree_arm, p_nree, self.p.alpha_ree, self.p.sigma_INT, INT)

        # REE Armington: domestic vs imported
        REE_dom, REE_imp = ces_demand(
            p_ree_dom, p_ree_imp, self.p.alpha_ree_dom, self.p.sigma_A, REE
        )

        return {
            "output": Y,
            "unit_cost": c_y,
            "total_cost": c_y * Y,
            "VA": VA,
            "INT": INT,
            "L": L,
            "K": K,
            "REE": REE,
            "NREE": NREE,
            "REE_dom": REE_dom,
            "REE_imp": REE_imp,
        }

    def output_supply(
        self,
        p_y: float,
        w: float,
        r: float,
        p_ree_dom: float,
        p_ree_imp: float,
        p_nree: float,
        supply_elasticity: float = 1.5,
    ) -> float:
        """
        Compute profit-maximising output (simplified inverse supply).

        Y* = Y_0 * (p_y / c_y)^η  where η = supply elasticity
        """
        c_y = self.unit_cost(w, r, p_ree_dom, p_ree_imp, p_nree)
        # Zero-profit condition: p_y = c_y at equilibrium
        # Deviations drive output response
        price_cost_ratio = p_y / (c_y + 1e-12)
        return price_cost_ratio ** supply_elasticity


def build_sector_params(
    sector_names: list[str],
    ree_intensity: np.ndarray,
    china_import_share: np.ndarray,
) -> list[SectorParams]:
    """
    Build SectorParams list calibrated from IO REE intensity and import share data.

    Default elasticities from literature (short-run, REE-constrained):
      σ_INT = 0.15–0.25 for REE-intensive sectors (near-zero substitution)
      σ_INT = 0.40–0.60 for REE-low sectors
    """
    params = []
    for i, name in enumerate(sector_names):
        ri = ree_intensity[i]
        ci = china_import_share[i]

        # REE share in intermediates: derived from intensity
        # More REE-intensive sectors have higher alpha_ree
        alpha_ree = min(ri * 5, 0.40)   # cap at 40% of intermediates

        # Domestic REE share (inverse of import share)
        alpha_ree_dom = max(1 - ci, 0.05)

        # Substitution elasticity: lower for REE-intensive sectors
        if ri > 0.015:
            sigma_INT = 0.15   # very constrained (Aerospace, Wind, EV, Electronics)
        elif ri > 0.005:
            sigma_INT = 0.25
        else:
            sigma_INT = 0.50

        # Value-added share: ~50% for manufacturing, ~70% for services
        if "Service" in name:
            alpha_va = 0.70
        elif name in ("Agriculture & Mining", "REE & Critical Minerals"):
            alpha_va = 0.45
        else:
            alpha_va = 0.50

        params.append(SectorParams(
            name=name,
            alpha_va=alpha_va,
            alpha_L=0.65,           # Labour share in VA (~UK norm)
            alpha_ree=alpha_ree,
            alpha_ree_dom=alpha_ree_dom,
            sigma_VA=1.0,
            sigma_L=0.5,
            sigma_INT=sigma_INT,
            sigma_A=0.30,           # Armington: limited dom/imp REE substitution
            TFP=1.0,
        ))
    return params
