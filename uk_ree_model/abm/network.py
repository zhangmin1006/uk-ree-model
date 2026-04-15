"""
network.py
==========
Supply chain network for the UK REE ABM.

Models the directed graph of supply relationships:
  Nodes = agents (suppliers, processors, manufacturers, households)
  Edges = supply relationships, weighted by transaction volume

Network topology follows:
  REE Suppliers → [Processors] → Manufacturers → Households

Risk propagation:
  If node i fails (supply disruption), risk propagates to downstream j:
  Risk_j(t) = 1 - Π_{i ∈ N(j)} (1 - ρ_ij * 1[fail_i])

Network resilience:
  R(G) = 1 - |V_fail| / |V|

Key finding from ECB 2025: >80% of large EU firms are within 3 supply
chain intermediaries of a Chinese REE producer. This module replicates
that topology for UK firms.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Optional
import pandas as pd


class SupplyChainNetwork:
    """
    Directed supply chain graph for the REE ABM.

    Parameters
    ----------
    agents       : list   All agents in the model.
    io_data      : dict   UK IO data (for edge weight calibration).
    rng          : np.random.Generator
    """

    def __init__(
        self,
        agents: list,
        io_data: dict,
        rng: Optional[np.random.Generator] = None,
    ):
        self.agents = agents
        self.io_data = io_data
        self.rng = rng or np.random.default_rng(42)
        self.G = nx.DiGraph()
        self._build_network()

    def _build_network(self):
        """Construct the supply chain graph from agent list and IO structure."""
        from .agents import (
            REESupplierAgent, ManufacturerAgent,
            HouseholdAgent, GovernmentAgent, ForeignAgent,
        )

        # Add all agents as nodes
        for agent in self.agents:
            node_type = type(agent).__name__
            self.G.add_node(
                agent.unique_id,
                agent=agent,
                agent_type=node_type,
                label=getattr(agent, "country", getattr(agent, "sector_idx", str(agent.unique_id))),
            )

        # Add edges: REE Suppliers → Manufacturers
        suppliers = [a for a in self.agents if isinstance(a, REESupplierAgent)]
        manufacturers = [a for a in self.agents if isinstance(a, ManufacturerAgent)]
        households = [a for a in self.agents if isinstance(a, HouseholdAgent)]
        government = [a for a in self.agents if isinstance(a, GovernmentAgent)]

        A = self.io_data["A"]
        x = self.io_data["x"]

        for supplier in suppliers:
            for mfr in manufacturers:
                # Edge weight = REE intensity × sector output (transaction volume proxy)
                weight = mfr.ree_intensity * mfr.base_output
                china_weight = (
                    weight * mfr.china_import_share
                    if getattr(supplier, "country", "") == "China"
                    else weight * (1 - mfr.china_import_share)
                )
                if china_weight > 0.001:
                    self.G.add_edge(
                        supplier.unique_id,
                        mfr.unique_id,
                        weight=china_weight,
                        dependency=mfr.china_import_share if supplier.country == "China" else 0.2,
                        ree_flow=True,
                    )

        # Add edges: Manufacturers → Households (through demand)
        for mfr in manufacturers:
            for hh in households:
                self.G.add_edge(
                    mfr.unique_id,
                    hh.unique_id,
                    weight=mfr.base_output * hh.expenditure_shares[mfr.sector_idx],
                    ree_flow=False,
                )

        # Government → Manufacturers (stockpile release, subsidies)
        for gov in government:
            for mfr in manufacturers:
                if mfr.ree_intensity > 0.01:
                    self.G.add_edge(
                        gov.unique_id,
                        mfr.unique_id,
                        weight=0.01,
                        ree_flow=True,
                        gov_link=True,
                    )

        # Inter-manufacturer edges (from IO A matrix)
        for i, mfr_i in enumerate(manufacturers):
            for j, mfr_j in enumerate(manufacturers):
                if i != j:
                    si, sj = mfr_i.sector_idx, mfr_j.sector_idx
                    flow = A[si, sj] * x[sj]
                    if flow > 0.5:   # only add significant flows (> £0.5bn)
                        self.G.add_edge(
                            mfr_i.unique_id,
                            mfr_j.unique_id,
                            weight=flow,
                            ree_flow=False,
                        )

    # ------------------------------------------------------------------
    # Risk propagation
    # ------------------------------------------------------------------

    def propagate_disruption(
        self,
        failed_node_ids: list[int],
        max_hops: int = 3,
    ) -> dict[int, float]:
        """
        Compute disruption risk for each node given a set of failed upstream nodes.

        Risk_j = 1 - Π_{i ∈ N(j)} (1 - ρ_ij * 1[fail_i])

        Parameters
        ----------
        failed_node_ids : list[int]  Nodes that have failed (supply disrupted).
        max_hops        : int        Maximum propagation depth (ECB found 3 hops covers 80%).

        Returns
        -------
        dict : node_id → disruption risk (0–1)
        """
        risk = {n: 0.0 for n in self.G.nodes()}
        for node in failed_node_ids:
            risk[node] = 1.0

        # BFS propagation up to max_hops
        frontier = set(failed_node_ids)
        for hop in range(max_hops):
            next_frontier = set()
            for upstream in frontier:
                for downstream in self.G.successors(upstream):
                    edge_data = self.G.edges[upstream, downstream]
                    dependency = edge_data.get("dependency", 0.5)
                    # Compound risk
                    current_risk = risk.get(downstream, 0.0)
                    propagated = dependency * risk[upstream]
                    risk[downstream] = 1 - (1 - current_risk) * (1 - propagated)
                    next_frontier.add(downstream)
            frontier = next_frontier
            if not frontier:
                break

        return risk

    def supply_chain_depth(self, source_ids: list[int]) -> dict[int, int]:
        """
        Compute shortest path length (supply chain distance) from sources to all nodes.

        Replicates the ECB finding of 3-intermediary concentration.
        """
        depths = {}
        for source in source_ids:
            lengths = nx.single_source_shortest_path_length(self.G, source)
            for node, dist in lengths.items():
                if node not in depths or depths[node] > dist:
                    depths[node] = dist
        return depths

    # ------------------------------------------------------------------
    # Resilience metrics
    # ------------------------------------------------------------------

    def resilience(self, failed_node_ids: list[int]) -> float:
        """
        Network resilience R(G) = 1 - |V_fail| / |V|

        Nodes are considered 'failed' if disruption risk > 0.5.
        """
        risk = self.propagate_disruption(failed_node_ids)
        n_failed = sum(1 for r in risk.values() if r > 0.5)
        return 1.0 - n_failed / (len(self.G.nodes()) + 1e-12)

    def critical_nodes(self, top_k: int = 5) -> list[int]:
        """
        Identify the most critical nodes by betweenness centrality.
        (Removal would maximise disruption to the network.)
        """
        bc = nx.betweenness_centrality(self.G, weight="weight")
        return sorted(bc, key=bc.get, reverse=True)[:top_k]

    def ree_exposure_by_manufacturer(self) -> pd.DataFrame:
        """
        For each manufacturer node, compute total REE flow weight from suppliers.
        """
        from .agents import ManufacturerAgent, REESupplierAgent
        records = []
        for node in self.G.nodes():
            agent = self.G.nodes[node]["agent"]
            if not isinstance(agent, ManufacturerAgent):
                continue
            in_edges = self.G.in_edges(node, data=True)
            ree_flow = sum(
                d.get("weight", 0) for _, _, d in in_edges if d.get("ree_flow", False)
            )
            records.append({
                "node_id": node,
                "sector_idx": agent.sector_idx,
                "ree_intensity": agent.ree_intensity,
                "base_output": agent.base_output,
                "ree_flow_in": ree_flow,
                "china_import_share": agent.china_import_share,
            })
        return pd.DataFrame(records)

    def intermediary_hops_analysis(
        self,
        chinese_supplier_ids: list[int],
    ) -> pd.DataFrame:
        """
        Replicate ECB analysis: count manufacturers within N hops of Chinese suppliers.
        Validates against ECB finding: >80% within 3 hops.
        """
        depths = self.supply_chain_depth(chinese_supplier_ids)
        from .agents import ManufacturerAgent
        records = []
        for node, depth in depths.items():
            agent = self.G.nodes[node]["agent"]
            if isinstance(agent, ManufacturerAgent):
                records.append({
                    "node_id": node,
                    "sector_idx": agent.sector_idx,
                    "hops_from_china_ree": depth,
                    "within_3_hops": depth <= 3,
                })
        df = pd.DataFrame(records)
        if len(df) > 0:
            pct_within_3 = df["within_3_hops"].mean() * 100
            print(f"Manufacturers within 3 hops of Chinese REE: {pct_within_3:.1f}% "
                  f"(ECB benchmark: >80%)")
        return df

    def summary_stats(self) -> dict:
        """Return network summary statistics."""
        return {
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
            "avg_degree": np.mean([d for _, d in self.G.degree()]),
            "density": nx.density(self.G),
            "is_dag": nx.is_directed_acyclic_graph(self.G),
            "n_weakly_connected": nx.number_weakly_connected_components(self.G),
        }
