"""
scenarios.py
============
Scenario definitions for the UK REE DIO–CGE–ABM simulation.

Five scenarios from UK_REE_DIO_CGE_ABM_Plan.md:

  A: Moderate     — θ=0.30, 12 months
  B: Severe       — θ=0.75,  6 months  (replicates May 2025 magnet collapse)
  C: Sustained    — θ=0.50, 24 months  (chronic disruption + substitution onset)
  D: Complete     — θ=1.00,  3 months  (systemic stress test)
  E: Demand-side  — θ=0.00, 36 months  (Net Zero demand growth, no supply shock)

Each scenario returns:
  - theta_path    : list[float]  per-period shock value (monthly)
  - label         : str
  - description   : str
  - policy        : dict         government policy instrument settings
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Scenario:
    """Container for a single simulation scenario."""
    label: str
    description: str
    theta_path: list       # length = n_periods (months)
    policy: dict = field(default_factory=dict)
    n_periods: int = 0

    def __post_init__(self):
        self.n_periods = len(self.theta_path)

    def __repr__(self):
        return (
            f"Scenario({self.label!r}, "
            f"θ_max={max(self.theta_path):.2f}, "
            f"periods={self.n_periods})"
        )


def _ramp(start: float, end: float, n: int) -> list:
    """Linear ramp from start to end over n periods."""
    return list(np.linspace(start, end, n))


def _flat(value: float, n: int) -> list:
    """Constant θ for n periods."""
    return [value] * n


def _step_profile(
    pre: float, shock: float, post: float,
    n_pre: int, n_shock: int, n_post: int,
) -> list:
    """Flat pre-shock, flat shock, flat post-shock."""
    return _flat(pre, n_pre) + _flat(shock, n_shock) + _flat(post, n_post)


def _ramp_down_profile(
    shock: float, n_shock: int, n_recovery: int, post: float = 0.0
) -> list:
    """Flat shock then linear recovery."""
    return _flat(shock, n_shock) + _ramp(shock, post, n_recovery)


# ---------------------------------------------------------------------------
# Scenario A: Moderate Export Control (12 months)
# ---------------------------------------------------------------------------

def scenario_a(n_pre: int = 3, n_shock: int = 12, n_post: int = 6) -> Scenario:
    """
    Moderate disruption: China applies export licensing for 7 HREEs.
    ~70% of normal volumes continue under partial compliance.
    UK benefits from partial diplomatic exemption: effective θ ≈ 0.25.
    """
    theta_path = (
        _flat(0.0, n_pre)
        + _ramp(0.0, 0.25, 3)          # 3-month ramp up as licenses delayed
        + _flat(0.25, n_shock - 3)
        + _ramp(0.25, 0.05, n_post)    # gradual easing
    )
    return Scenario(
        label="A: Moderate",
        description=(
            "θ=0.25 effective, 12 months. China applies export licensing for 7 HREEs; "
            "UK receives partial exemption via diplomatic channels."
        ),
        theta_path=theta_path,
        policy={
            "stockpile_months": 3.0,
            "release_trigger_price": 2.5,
            "tariff_rate": 0.0,
            "subsidy_rate": 0.02,
        },
    )


# ---------------------------------------------------------------------------
# Scenario B: Severe Supply Shock (6 months)
# ---------------------------------------------------------------------------

def scenario_b(n_pre: int = 3, n_shock: int = 6, n_post: int = 12) -> Scenario:
    """
    Severe: replicates observed May 2025 Chinese magnet export collapse (–75% YoY).
    No diplomatic adjustment; UK faces same supply denial as EU.
    """
    theta_path = (
        _flat(0.0, n_pre)
        + _ramp(0.0, 0.75, 2)          # sharp shock onset (2 months)
        + _flat(0.75, n_shock - 2)
        + _ramp(0.75, 0.10, n_post)    # slow recovery
    )
    return Scenario(
        label="B: Severe",
        description=(
            "θ=0.75, 6 months. Replicates May 2025 Chinese magnet export collapse. "
            "UK exposed to full shock — no exemption. Emergency stockpile triggered."
        ),
        theta_path=theta_path,
        policy={
            "stockpile_months": 3.0,
            "release_trigger_price": 2.0,  # earlier trigger under severe shock
            "tariff_rate": 0.0,
            "subsidy_rate": 0.05,
        },
    )


# ---------------------------------------------------------------------------
# Scenario C: Sustained Chronic Disruption (24 months)
# ---------------------------------------------------------------------------

def scenario_c(n_pre: int = 3, n_shock: int = 24, n_post: int = 9) -> Scenario:
    """
    Chronic: θ=0.50 for 24 months. Firms begin substituting after 6 months.
    Partial non-Chinese capacity ramp-up reduces effective θ over time.
    """
    # θ starts at 0.50, eases slightly as alternative supply ramps up
    theta_path = (
        _flat(0.0, n_pre)
        + _ramp(0.0, 0.50, 3)
        + _flat(0.50, 6)
        + _ramp(0.50, 0.35, 15)        # alt supply ramp-up eases pressure
        + _ramp(0.35, 0.15, n_post)
    )
    return Scenario(
        label="C: Sustained",
        description=(
            "θ=0.50 for 24 months. Chronic disruption; firm substitution begins after "
            "6 months. Non-Chinese supply ramps up, reducing effective θ to 0.35 by month 18."
        ),
        theta_path=theta_path,
        policy={
            "stockpile_months": 6.0,   # larger reserve for sustained shock
            "release_trigger_price": 2.0,
            "tariff_rate": 0.05,       # modest tariff on Chinese REE
            "subsidy_rate": 0.10,      # significant subsidy to domestic processing
        },
    )


# ---------------------------------------------------------------------------
# Scenario D: Complete Disruption (3 months)
# ---------------------------------------------------------------------------

def scenario_d(n_pre: int = 3, n_shock: int = 3, n_post: int = 18) -> Scenario:
    """
    Worst-case systemic stress test: θ=1.0 for 3 months.
    Tests emergency stockpile, resilience threshold, and recovery capacity.
    """
    theta_path = (
        _flat(0.0, n_pre)
        + _ramp(0.0, 1.0, 1)           # immediate full shock
        + _flat(1.0, n_shock - 1)
        + _ramp(1.0, 0.30, 6)          # partial recovery as alt supply mobilises
        + _ramp(0.30, 0.05, n_post - 6)
    )
    return Scenario(
        label="D: Complete",
        description=(
            "θ=1.00, 3 months. Complete Chinese REE supply cutoff. "
            "Emergency stockpile release, maximum firm stress. Tests network resilience floor."
        ),
        theta_path=theta_path,
        policy={
            "stockpile_months": 6.0,
            "release_trigger_price": 1.5,  # immediate release
            "tariff_rate": 0.0,
            "subsidy_rate": 0.15,
        },
    )


# ---------------------------------------------------------------------------
# Scenario E: Demand-Side — Net Zero Growth (36 months, no shock)
# ---------------------------------------------------------------------------

def scenario_e(n_periods: int = 36) -> Scenario:
    """
    No supply shock (θ=0), but UK Net Zero policy drives REE demand up.
    EV targets + offshore wind expansion increase REE demand by ~40% over 3 years.
    Tests whether supply can keep pace with UK policy-driven demand growth.
    """
    return Scenario(
        label="E: Net Zero Demand",
        description=(
            "θ=0 (no shock), 36 months. UK Net Zero targets drive 40% REE demand growth "
            "from EV mandates and offshore wind expansion. Tests supply adequacy."
        ),
        theta_path=_flat(0.0, n_periods),
        policy={
            "stockpile_months": 3.0,
            "release_trigger_price": 3.0,
            "tariff_rate": 0.0,
            "subsidy_rate": 0.05,
            "demand_growth_rate_annual": 0.15,  # 15% annual REE demand growth
        },
    )


# ---------------------------------------------------------------------------
# Convenience: get all scenarios
# ---------------------------------------------------------------------------

def get_all_scenarios() -> dict[str, Scenario]:
    return {
        "A": scenario_a(),
        "B": scenario_b(),
        "C": scenario_c(),
        "D": scenario_d(),
        "E": scenario_e(),
    }


def get_theta_for_cge(scenario: Scenario, period: int) -> float:
    """Extract θ at a specific period for CGE static analysis."""
    if period < len(scenario.theta_path):
        return scenario.theta_path[period]
    return scenario.theta_path[-1]


def cge_scenario_thetas(scenarios: dict[str, Scenario]) -> dict[str, float]:
    """Return the peak θ for each scenario (used for CGE static runs)."""
    return {k: max(s.theta_path) for k, s in scenarios.items()}
