"""
scheduler.py
============
ABM model scheduler — the main REE Agent-Based Model class.

Orchestrates:
  1. Agent initialisation from DIO/CGE equilibrium
  2. Time loop with agent stepping and market clearing
  3. REE price formation
  4. State variable collection for DIO/CGE coupling

The model follows a monthly time step. Each period:
  1. Government agent sets policy instruments
  2. Supplier agents produce (θ applied)
  3. REE market clears → price formed
  4. Manufacturer agents update expectations, reorder, produce
  5. Households consume
  6. Aggregate statistics recorded
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .agents import (
    REESupplierAgent,
    ManufacturerAgent,
    HouseholdAgent,
    GovernmentAgent,
    ForeignAgent,
)
from .network import SupplyChainNetwork
from .metrics import MetricsCollector


class UKREEModel:
    """
    UK REE Agent-Based Model.

    Parameters
    ----------
    io_data          : dict   UK IO data (from uk_io_synthetic).
    cge_equilibrium  : dict   CGE baseline equilibrium (from cge.equilibrium.CGEModel).
    theta_path       : list   REE supply shock θ at each time period.
    n_manufacturers  : int    Number of manufacturer agents (total across sectors).
    seed             : int    Random seed.
    """

    def __init__(
        self,
        io_data: dict,
        cge_equilibrium: Optional[dict] = None,
        theta_path: Optional[list] = None,
        n_manufacturers: int = 100,
        seed: int = 42,
    ):
        self.io_data = io_data
        self.cge_eq = cge_equilibrium
        self.theta_path = theta_path or [0.0]
        self.n_manufacturers = n_manufacturers
        self.rng = np.random.default_rng(seed)

        # Model state
        self.current_step = 0
        self.current_ree_price = 1.0
        self.current_wage = 1.0
        self.current_prices = np.ones(io_data["n_sectors"])
        self.government_supply_release = 0.0

        # Initialise agents and network
        self.all_agents: list = []
        self._init_agents()
        self.network = SupplyChainNetwork(self.all_agents, io_data, self.rng)
        self.metrics = MetricsCollector(self)

    # ------------------------------------------------------------------
    # Agent initialisation
    # ------------------------------------------------------------------

    def _init_agents(self):
        """Create all agent instances calibrated from IO/CGE data."""
        io = self.io_data
        x = io["x"] / 12.0          # convert annual £bn → monthly £bn
        ree_intensity = io["ree_intensity"]
        china_share = io["china_import_share"]
        n_sectors = io["n_sectors"]

        # Total monthly REE demand from all UK sectors (consistent with ree_intensity)
        # Use this as the supply reference so the market clears at baseline (no pre-shock spike)
        total_ree_demand = (x * ree_intensity).sum()  # £bn/month

        # --- Supplier agents ---
        # China: dominant REE supplier
        china_supplier = REESupplierAgent(
            model=self,
            country="China",
            base_supply=total_ree_demand * 0.70,   # 70% of monthly UK REE demand
            theta=0.0,
            price_markup=0.20,
            noise_std=0.03,
        )
        # Australia/US: alternative supplier
        alt_supplier = REESupplierAgent(
            model=self,
            country="Australia/US",
            base_supply=total_ree_demand * 0.20,
            theta=0.0,
            price_markup=0.30,
            noise_std=0.05,
        )
        # Other
        other_supplier = REESupplierAgent(
            model=self,
            country="Other",
            base_supply=total_ree_demand * 0.10,
            theta=0.0,
            price_markup=0.25,
            noise_std=0.08,
        )
        self.suppliers = [china_supplier, alt_supplier, other_supplier]
        self.all_agents.extend(self.suppliers)

        # --- Manufacturer agents ---
        self.manufacturers = []
        sector_agent_counts = self._allocate_agents_to_sectors(self.n_manufacturers, n_sectors, x)

        for sector_idx in range(n_sectors):
            n_agents = sector_agent_counts[sector_idx]
            if n_agents == 0:
                continue
            # Each agent gets an equal share of sector output
            agent_output = x[sector_idx] / n_agents

            for _ in range(n_agents):
                # Heterogeneity: add noise to key parameters
                intensity_noise = self.rng.normal(1.0, 0.10)
                lambda_noise = self.rng.beta(3, 7)   # learning rate ~ 0.3 on average
                sub_threshold = self.rng.uniform(1.5, 4.0)  # price threshold for substitution

                mfr = ManufacturerAgent(
                    model=self,
                    sector_idx=sector_idx,
                    base_output=agent_output,
                    ree_intensity=max(ree_intensity[sector_idx] * intensity_noise, 0.0),
                    china_import_share=china_share[sector_idx],
                    s_reorder=self.rng.uniform(1.0, 2.0),
                    S_target=self.rng.uniform(2.5, 4.5),
                    lambda_expect=lambda_noise,
                    sub_price_threshold=sub_threshold,
                )
                self.manufacturers.append(mfr)
        self.all_agents.extend(self.manufacturers)

        # --- Household agents (3 income classes) ---
        # mu = household expenditure shares; prefer CGE calibration params if available
        if self.cge_eq and "mu" in self.cge_eq:
            mu = self.cge_eq["mu"]
        else:
            y_ann = io["y"]
            mu = y_ann / (y_ann.sum() + 1e-12)
        income_split = [0.30, 0.50, 0.20]  # low, middle, high income shares
        income_total = (io["x"] * self.io_data.get("employment_coeff", np.ones(n_sectors))).sum() * 0.001
        self.households = []
        for i, (label, share) in enumerate(zip(["low", "middle", "high"], income_split)):
            hh = HouseholdAgent(
                model=self,
                income_class=label,
                base_income=income_total * share,
                expenditure_shares=mu,
                savings_rate=0.05 + i * 0.03,
            )
            self.households.append(hh)
        self.all_agents.extend(self.households)

        # --- Government agent ---
        self.government = GovernmentAgent(
            model=self,
            stockpile_months=3.0,
            release_trigger_price=2.5,
            base_ree_consumption=io["x"][1] * 0.10,
        )
        self.all_agents.append(self.government)

        # --- Foreign agents (EU, US, RoW) ---
        # Foreign agents demand proportional to UK total REE demand (EU demand ≈ 3× UK, US ≈ 1×)
        self.foreign_agents = [
            ForeignAgent(self, "EU", ree_demand_base=total_ree_demand * 3.0, demand_elasticity=-0.25),
            ForeignAgent(self, "US",  ree_demand_base=total_ree_demand * 1.0, demand_elasticity=-0.30),
            ForeignAgent(self, "RoW", ree_demand_base=total_ree_demand * 2.0, demand_elasticity=-0.40),
        ]
        self.all_agents.extend(self.foreign_agents)

    def _allocate_agents_to_sectors(
        self,
        n_total: int,
        n_sectors: int,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Allocate manufacturer agents to sectors proportional to REE-weighted output.
        More agents for REE-intensive sectors (richer heterogeneity where it matters).
        """
        ree_weight = self.io_data["ree_intensity"] * x
        ree_weight[1] = 0   # REE sector itself handled via suppliers
        weights = ree_weight / (ree_weight.sum() + 1e-12)

        # At least 2 agents per active sector
        allocation = np.maximum(np.round(weights * n_total).astype(int), 0)
        active = (self.io_data["ree_intensity"] > 0) | (x > 5)
        allocation[~active] = 0
        allocation[1] = 0   # REE sector is a supplier, not manufacturer here

        # Adjust to match n_total
        diff = n_total - allocation.sum()
        if diff > 0:
            # Add remaining agents to largest sector
            allocation[np.argmax(x)] += diff
        elif diff < 0:
            # Remove from smallest active sector
            active_idx = np.where(allocation > 2)[0]
            for _ in range(-diff):
                allocation[active_idx[np.argmin(allocation[active_idx])]] -= 1

        return allocation

    # ------------------------------------------------------------------
    # Market clearing
    # ------------------------------------------------------------------

    def _clear_ree_market(self) -> float:
        """
        Determine REE market price via supply-demand balance.

        Total REE supply (from suppliers + government stockpile release)
        vs. total REE demand (from manufacturer reorder decisions + foreign agents).
        """
        # Supply side
        total_supply = sum(s.current_supply for s in self.suppliers)
        total_supply += self.government_supply_release

        # Demand side: UK manufacturers
        uk_demand = sum(m.ree_demand_target for m in self.manufacturers)

        # Demand side: foreign agents
        for fa in self.foreign_agents:
            fa.compute_demand(self.current_ree_price)
        foreign_demand = sum(fa.current_demand for fa in self.foreign_agents)

        total_demand = uk_demand + foreign_demand

        # Price formation: Walrasian tâtonnement approximation
        if total_supply > 0:
            ratio = total_demand / total_supply
            price_adjust = ratio ** (1 / 2.0)   # inverse supply elasticity = 0.5
        else:
            price_adjust = 10.0  # extreme scarcity

        new_price = self.current_ree_price * 0.70 + price_adjust * 0.30  # partial adjustment
        self.current_ree_price = np.clip(new_price, 0.5, 50.0)

        # Allocate supply to UK manufacturers
        if total_demand > 0:
            uk_share = uk_demand / (total_demand + 1e-12)
            uk_supply_allocated = total_supply * uk_share
        else:
            uk_supply_allocated = 0.0

        # Distribute among UK manufacturers by demand proportion
        total_uk_dem = sum(m.ree_demand_target for m in self.manufacturers) + 1e-12
        for m in self.manufacturers:
            share = m.ree_demand_target / total_uk_dem
            m.receive_ree(share * uk_supply_allocated)

        return self.current_ree_price

    # ------------------------------------------------------------------
    # Time step
    # ------------------------------------------------------------------

    def step(self):
        """Execute one time period (month)."""
        # Set current theta from path
        theta = (
            self.theta_path[self.current_step]
            if self.current_step < len(self.theta_path)
            else self.theta_path[-1]
        )

        # Government policy
        self.government.step()

        # Supplier production
        for s in self.suppliers:
            if s.country == "China":
                s.set_theta(theta)
            s.step()

        # REE market clearing
        self._clear_ree_market()

        # Manufacturer decisions and production
        for m in self.manufacturers:
            m.step()

        # Household consumption
        for hh in self.households:
            hh.step()

        # Collect metrics
        self.metrics.collect(self.current_step, theta)

        self.current_step += 1

    def run(self, n_steps: Optional[int] = None) -> pd.DataFrame:
        """
        Run the full simulation.

        Parameters
        ----------
        n_steps : int  Number of time periods (months). Defaults to len(theta_path).

        Returns
        -------
        DataFrame of collected metrics.
        """
        steps = n_steps or len(self.theta_path)
        for _ in range(steps):
            self.step()
        return self.metrics.to_dataframe()
