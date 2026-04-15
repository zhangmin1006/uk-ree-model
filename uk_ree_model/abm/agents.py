"""
agents.py
=========
Agent definitions for the UK REE Agent-Based Model.

Agent types:
  REESupplierAgent   — Chinese / non-Chinese REE exporters
  ManufacturerAgent  — UK firms in REE-intensive sectors
  HouseholdAgent     — UK consumer households (representative groups)
  GovernmentAgent    — UK policy instrument (stockpile, tariff, subsidy)
  ForeignAgent       — EU / US / RoW competing for REE supply

Agent decision rules are adaptive (bounded-rational):
  - Manufacturers use (s, S) inventory policy
  - Substitution decisions are probabilistic (logistic)
  - Price expectations use exponential smoothing
  - Suppliers respond to export control regimes
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    _id_counter = 0

    def __init__(self, model):
        BaseAgent._id_counter += 1
        self.unique_id = BaseAgent._id_counter
        self.model = model

    def step(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# REE Supplier Agent
# ---------------------------------------------------------------------------

class REESupplierAgent(BaseAgent):
    """
    Represents a REE-producing/exporting country or firm.

    Key attributes
    --------------
    country       : str    'China', 'Australia', 'US', 'Other'
    base_supply   : float  Base REE supply capacity (£bn)
    theta         : float  Export restriction severity (exogenous, set by model)
    price_markup  : float  Markup over marginal cost
    noise_std     : float  Production noise standard deviation
    """

    def __init__(
        self,
        model,
        country: str,
        base_supply: float,
        theta: float = 0.0,
        price_markup: float = 0.20,
        noise_std: float = 0.05,
    ):
        super().__init__(model)
        self.country = country
        self.base_supply = base_supply
        self.theta = theta
        self.price_markup = price_markup
        self.noise_std = noise_std
        self.current_supply = base_supply
        self.current_price = 1.0   # normalised price
        self.allocation = {}       # {buyer_id: quantity}

    def set_theta(self, theta: float):
        """Update export restriction level (called by scenario controller)."""
        self.theta = theta

    def produce(self) -> float:
        """
        Compute current supply output:
            x(t) = (1 - θ) * x_base * exp(ε)
        where ε ~ N(0, σ²) is production noise.
        """
        eps = self.model.rng.normal(0, self.noise_std)
        self.current_supply = self.base_supply * (1 - self.theta) * np.exp(eps)
        return max(self.current_supply, 0.0)

    def set_price(self, demand: float) -> float:
        """
        Set REE price based on supply-demand balance.
            p(t) = p_base * (D / S)^{1/ε}  where ε = supply elasticity
        """
        supply = max(self.current_supply, 1e-6)
        ratio = demand / supply
        price_multiplier = ratio ** (1 / 2.0)   # supply elasticity ≈ 2
        self.current_price = 1.0 * price_multiplier * (1 + self.price_markup)
        return self.current_price

    def allocate_to_buyers(self, buyers: list, total_supply: float):
        """
        Allocate supply pro-rata to buyers by historical trade share.
        """
        if not buyers:
            return
        # Each buyer's share based on their historical demand
        demands = {b.unique_id: b.ree_demand_target for b in buyers}
        total_demand = sum(demands.values()) + 1e-12
        for b in buyers:
            share = demands[b.unique_id] / total_demand
            self.allocation[b.unique_id] = share * total_supply

    def step(self):
        self.produce()


# ---------------------------------------------------------------------------
# Manufacturer Agent
# ---------------------------------------------------------------------------

class ManufacturerAgent(BaseAgent):
    """
    UK manufacturing firm in an REE-intensive sector.

    Decision rules:
      1. Production: Q = min(capacity, REE_available / ree_intensity)
      2. Inventory: (s, S) reorder policy
      3. Substitution: probabilistic switch based on REE price
      4. Expectations: exponential smoothing on REE price

    Parameters
    ----------
    sector_idx         : int    Sector index (0–11)
    base_output        : float  Baseline output (£bn)
    ree_intensity      : float  REE input per £ output
    china_import_share : float  Share of REE from China
    s_reorder          : float  Inventory reorder point (months of supply)
    S_target           : float  Target inventory level (months of supply)
    lambda_expect      : float  Learning rate for price expectations [0,1]
    sub_price_threshold: float  REE price at which substitution becomes viable
    sub_prob_scale     : float  Logistic scale for substitution probability
    """

    def __init__(
        self,
        model,
        sector_idx: int,
        base_output: float,
        ree_intensity: float,
        china_import_share: float,
        s_reorder: float = 1.5,     # reorder when < 1.5 months of REE
        S_target: float = 3.0,      # aim for 3 months of REE stock
        lambda_expect: float = 0.30,
        sub_price_threshold: float = 2.0,   # 2× base price triggers substitution
        sub_prob_scale: float = 1.0,
    ):
        super().__init__(model)
        self.sector_idx = sector_idx
        self.base_output = base_output
        self.output_capacity = base_output
        self.ree_intensity = ree_intensity
        self.china_import_share = china_import_share

        # Inventory policy parameters
        self.s_reorder = s_reorder
        self.S_target = S_target
        # base_output is already monthly (annual IO value divided by 12 in scheduler)
        monthly_use = base_output * ree_intensity
        self.inventory = S_target * monthly_use   # start at target
        self.ree_demand_target = monthly_use

        # State variables
        self.current_output = base_output
        self.current_ree_received = 0.0
        self.has_substituted = False
        self.substitution_investment = 0.0

        # Expectations
        self.p_ree_expected = 1.0
        self.lambda_expect = lambda_expect

        # Substitution parameters
        self.sub_price_threshold = sub_price_threshold
        self.sub_prob_scale = sub_prob_scale
        self.ree_intensity_effective = ree_intensity

        # History
        self.output_history: list[float] = []
        self.inventory_history: list[float] = []
        self.p_ree_history: list[float] = []

    # ------------------------------------------------------------------
    # Core decision rules
    # ------------------------------------------------------------------

    def update_expectations(self, p_ree_observed: float):
        """
        Adaptive (exponential smoothing) price expectations.

        p^e(t+1) = λ * p(t) + (1-λ) * p^e(t)
        """
        self.p_ree_expected = (
            self.lambda_expect * p_ree_observed
            + (1 - self.lambda_expect) * self.p_ree_expected
        )

    def decide_reorder(self) -> float:
        """
        (s, S) inventory policy.

        Order REE up to S_target months if inventory drops below s_reorder months.
        """
        monthly_use = self.base_output * self.ree_intensity_effective
        inventory_months = self.inventory / (monthly_use + 1e-12)

        if inventory_months <= self.s_reorder:
            order = (self.S_target - inventory_months) * monthly_use
            self.ree_demand_target = max(order, 0.0)
        else:
            self.ree_demand_target = monthly_use   # routine restocking

        return self.ree_demand_target

    def decide_substitution(self) -> bool:
        """
        Logistic probability of adopting REE-reduced technology.

        P(substitute) = σ((p_ree_expected - threshold) / scale)
        """
        if self.has_substituted:
            return True
        z = (self.p_ree_expected - self.sub_price_threshold) / self.sub_prob_scale
        prob = 1.0 / (1.0 + np.exp(-z))
        if self.model.rng.random() < prob:
            self.has_substituted = True
            # Substitution reduces REE intensity by up to 40% at 50% cost penalty
            self.ree_intensity_effective = self.ree_intensity * 0.60
            self.substitution_investment = self.base_output * 0.05  # 5% of output
            return True
        return False

    def produce(self, ree_available: float) -> float:
        """
        Determine output given available REE supply.

        Hard constraint: Q = min(capacity, REE_available / ree_intensity)

        Softened by REE criticality: sectors with very low intensity can substitute
        other inputs, so output loss is moderated by an intensity elasticity factor.

        intensity_elasticity = min(ree_intensity * 50, 1.0)
          → 1.0 for REE-critical sectors (aerospace, wind, EV): hard constraint
          → 0.05 for services (ree_intensity=0.001): only 5% output loss per unit shortage
        """
        if self.ree_intensity_effective <= 1e-9:
            self.current_output = self.output_capacity
            return self.current_output

        # Fraction of REE demand that is satisfied this period
        required = self.output_capacity * self.ree_intensity_effective
        ree_satisfied_frac = min(ree_available / (required + 1e-12), 1.0)

        if ree_satisfied_frac >= 1.0:
            self.current_output = self.output_capacity
        else:
            # REE criticality: how tightly is output bound to REE supply?
            # High-intensity sectors (Dy in motors) face hard constraint;
            # Low-intensity (services) can buy from alternative sources.
            criticality = min(self.ree_intensity_effective * 60.0, 1.0)
            shortage_fraction = 1.0 - ree_satisfied_frac
            output_loss_fraction = shortage_fraction * criticality
            self.current_output = self.output_capacity * (1.0 - output_loss_fraction)

        self.current_output = max(self.current_output, 0.0)
        return self.current_output

    def receive_ree(self, quantity: float):
        """Update inventory with received REE delivery."""
        self.current_ree_received = quantity
        self.inventory += quantity

    def consume_ree(self):
        """Deplete inventory by production requirements."""
        ree_used = self.current_output * self.ree_intensity_effective
        self.inventory = max(self.inventory - ree_used, 0.0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self):
        p_ree = self.model.current_ree_price

        # Update adaptive expectations
        self.update_expectations(p_ree)

        # Decide substitution
        self.decide_substitution()

        # Decide reorder quantity
        self.decide_reorder()

        # Produce (REE available = inventory from previous receipt)
        ree_for_production = self.inventory
        self.produce(ree_for_production)

        # Consume REE from inventory
        self.consume_ree()

        # Record history
        self.output_history.append(self.current_output)
        self.inventory_history.append(self.inventory)
        self.p_ree_history.append(p_ree)


# ---------------------------------------------------------------------------
# Household Agent
# ---------------------------------------------------------------------------

class HouseholdAgent(BaseAgent):
    """
    Representative household income group.

    Income: labour wage + capital returns + government transfers
    Expenditure: Cobb-Douglas across sectors
    Savings: fixed savings rate
    """

    def __init__(
        self,
        model,
        income_class: str,  # 'low', 'middle', 'high'
        base_income: float,
        expenditure_shares: np.ndarray,
        savings_rate: float = 0.08,
    ):
        super().__init__(model)
        self.income_class = income_class
        self.base_income = base_income
        self.current_income = base_income
        self.expenditure_shares = expenditure_shares
        self.savings_rate = savings_rate
        self.utility_history: list[float] = []

    def compute_utility(self, consumption: np.ndarray, prices: np.ndarray) -> float:
        """Cobb-Douglas utility: U = Π C_j^μ_j"""
        real_cons = np.where(prices > 0, consumption / prices, 0)
        return float(np.prod(np.maximum(real_cons, 1e-12) ** self.expenditure_shares))

    def demand(self, prices: np.ndarray) -> np.ndarray:
        """Cobb-Douglas demand: C_j = μ_j * I / p_j"""
        disposable = self.current_income * (1 - self.savings_rate)
        return self.expenditure_shares * disposable / (prices + 1e-12)

    def step(self):
        wages = self.model.current_wage
        self.current_income = self.base_income * (wages / 1.0)
        prices = self.model.current_prices
        cons = self.demand(prices)
        u = self.compute_utility(cons, prices)
        self.utility_history.append(u)


# ---------------------------------------------------------------------------
# Government Agent
# ---------------------------------------------------------------------------

class GovernmentAgent(BaseAgent):
    """
    UK government: manages REE stockpile, tariffs, and subsidies.

    Policy rules:
      - Stockpile release: if REE price > trigger_price, release strategic reserve
      - Import tariff: apply tariff_rate to REE imports
      - Subsidy: provide subsidy_rate to domestic REE processing investment
    """

    def __init__(
        self,
        model,
        stockpile_months: float = 3.0,     # months of REE consumption in reserve
        release_trigger_price: float = 2.5, # release if price > 2.5× base
        tariff_rate: float = 0.0,
        subsidy_rate: float = 0.0,
        base_ree_consumption: float = 0.5,  # £bn/year total UK REE consumption
    ):
        super().__init__(model)
        self.stockpile = stockpile_months * base_ree_consumption / 12
        self.release_trigger = release_trigger_price
        self.tariff_rate = tariff_rate
        self.subsidy_rate = subsidy_rate
        self.stockpile_history: list[float] = []
        self.releases: list[float] = []

    def release_stockpile(self, p_ree: float) -> float:
        """Release REE from strategic stockpile if price exceeds trigger."""
        if p_ree > self.release_trigger and self.stockpile > 0:
            release = min(self.stockpile * 0.20, self.stockpile)  # release 20% per period
            self.stockpile -= release
            self.releases.append(release)
            return release
        self.releases.append(0.0)
        return 0.0

    def step(self):
        p_ree = self.model.current_ree_price
        released = self.release_stockpile(p_ree)
        self.model.government_supply_release = released
        self.stockpile_history.append(self.stockpile)


# ---------------------------------------------------------------------------
# Foreign Agent (EU / US / RoW)
# ---------------------------------------------------------------------------

class ForeignAgent(BaseAgent):
    """
    Foreign economy competing with UK for REE supply.

    Competes for Chinese REE allocation; creates supply-demand competition.
    """

    def __init__(
        self,
        model,
        region: str,
        ree_demand_base: float,
        demand_elasticity: float = -0.3,
    ):
        super().__init__(model)
        self.region = region
        self.ree_demand_base = ree_demand_base
        self.current_demand = ree_demand_base
        self.demand_elasticity = demand_elasticity

    def compute_demand(self, p_ree: float) -> float:
        """
        REE demand response to price change.

        D(p) = D_0 * (p / p_0)^ε  where ε = demand elasticity < 0
        """
        self.current_demand = self.ree_demand_base * (p_ree ** self.demand_elasticity)
        return self.current_demand

    def step(self):
        p_ree = self.model.current_ree_price
        self.compute_demand(p_ree)
