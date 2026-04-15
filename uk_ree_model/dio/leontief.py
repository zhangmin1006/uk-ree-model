"""
leontief.py
===========
Static and Dynamic Leontief Input-Output model for UK REE impact analysis.

Mathematical foundation:
  Static:  x = (I - A)^{-1} y                         [demand-driven]
  Dynamic: (I-A)x(t) - B[x(t+1)-x(t)] = c(t)         [with capital accumulation]
  Time path: x(t+1) = B^{-1}[(I-A)x(t) - c(t)] + x(t)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import inv, solve, matrix_rank
from typing import Optional


class StaticLeontief:
    """
    Standard static Leontief demand-driven IO model.

    Parameters
    ----------
    A : np.ndarray (n x n)  Direct input coefficient matrix.
    x : np.ndarray (n,)     Observed total output vector (used for calibration checks).
    sector_names : list[str] Optional sector labels.
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
        self._validate()
        self._compute_leontief_inverse()

    def _validate(self):
        assert self.A.shape == (self.n, self.n), "A must be square"
        assert np.all(self.A >= 0), "Technical coefficients must be non-negative"
        col_sums = self.A.sum(axis=0)
        if np.any(col_sums >= 1.0):
            import warnings
            warnings.warn(
                f"Some column sums of A >= 1 (max={col_sums.max():.3f}). "
                "Check that value added is positive for all sectors.",
                UserWarning,
            )
        if matrix_rank(np.eye(self.n) - self.A) < self.n:
            raise ValueError("(I - A) is singular; check the A matrix.")

    def _compute_leontief_inverse(self):
        I = np.eye(self.n)
        self.I_minus_A = I - self.A
        self.L = inv(self.I_minus_A)  # Leontief inverse

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def final_demand_to_output(self, y: np.ndarray) -> np.ndarray:
        """x = L * y  — compute total output from final demand."""
        return self.L @ y

    def output_multipliers(self) -> np.ndarray:
        """Column sums of L: total output induced per unit of final demand."""
        return self.L.sum(axis=0)

    def value_added_multipliers(self, va_coeff: np.ndarray) -> np.ndarray:
        """
        GDP/value-added multiplier for each sector.

        Parameters
        ----------
        va_coeff : np.ndarray (n,)  Value-added coefficient v_j = VA_j / x_j.
        """
        return va_coeff @ self.L

    def employment_multipliers(self, emp_coeff: np.ndarray) -> np.ndarray:
        """
        Employment multiplier: total jobs per unit of final demand.

        Parameters
        ----------
        emp_coeff : np.ndarray (n,)  Employment per unit output (e.g., FTE / £bn).
        """
        return emp_coeff @ self.L

    def ree_requirement_vector(self, ree_intensity: np.ndarray) -> np.ndarray:
        """
        Total (direct + indirect) REE requirement per unit of final demand.

        TRR_j = r @ L  where r = REE intensity vector.

        Parameters
        ----------
        ree_intensity : np.ndarray (n,)  REE input per £ of output by sector.
        """
        return ree_intensity @ self.L

    def forward_linkages(self) -> np.ndarray:
        """
        Ghosh-based forward (supply-side) linkage index.

        Forward linkage measures how much a sector's supply propagates to
        downstream users.  The correct measure is row sums of the Ghosh
        inverse H = (I - G)^{-1}, where G_{ij} = z_{ij}/x_i.

        Using row sums of the Leontief inverse L (a common mistake) understates
        upstream supply importance — especially for net-import sectors like REE.
        """
        # Build Ghosh output-allocation matrix G from A and x
        Z = self.A * self.x[np.newaxis, :]          # Z_ij = A_ij * x_j
        with np.errstate(divide="ignore", invalid="ignore"):
            G = np.where(
                self.x[:, np.newaxis] > 0,
                Z / self.x[:, np.newaxis],
                0.0,
            )
        from numpy.linalg import inv as _inv
        H = _inv(np.eye(self.n) - G)                # Ghosh inverse
        return H.sum(axis=1)                        # row sums = forward multipliers

    def backward_linkages(self) -> np.ndarray:
        """Column sums of L: backward (demand-side) linkage index."""
        return self.L.sum(axis=0)

    def key_sectors(self) -> pd.DataFrame:
        """Classify sectors as key / forward-linked / backward-linked / weak."""
        bl = self.backward_linkages()
        fl = self.forward_linkages()
        bl_norm = bl / bl.mean()
        fl_norm = fl / fl.mean()

        def classify(b, f):
            if b > 1 and f > 1:
                return "Key sector"
            elif b > 1:
                return "Backward-linked"
            elif f > 1:
                return "Forward-linked"
            else:
                return "Weak"

        return pd.DataFrame({
            "sector": self.sector_names,
            "backward_linkage": bl,
            "forward_linkage": fl,
            "bl_normalised": bl_norm,
            "fl_normalised": fl_norm,
            "classification": [classify(b, f) for b, f in zip(bl_norm, fl_norm)],
        }).set_index("sector")

    def summary(self, y: np.ndarray) -> pd.DataFrame:
        """Full output, multiplier and REE summary given a final demand vector."""
        x_hat = self.final_demand_to_output(y)
        bl = self.backward_linkages()
        fl = self.forward_linkages()
        return pd.DataFrame({
            "sector": self.sector_names,
            "final_demand_£bn": y,
            "total_output_£bn": x_hat,
            "output_multiplier": bl,
            "forward_linkage": fl,
        }).set_index("sector")


class DynamicLeontief(StaticLeontief):
    """
    Dynamic Leontief IO model with capital accumulation matrix B.

    Governing equation:
        (I - A) x(t) - B [x(t+1) - x(t)] = c(t)
    Rearranged:
        x(t+1) = B^{-1} [(I - A) x(t) - c(t)] + x(t)

    Parameters
    ----------
    A : np.ndarray (n x n)   Input coefficient matrix.
    B : np.ndarray (n x n)   Capital coefficient matrix.
    x : np.ndarray (n,)      Base-year output vector.
    sector_names : list[str]
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        x: np.ndarray,
        sector_names: Optional[list] = None,
    ):
        super().__init__(A, x, sector_names)
        self.B = B.copy()
        self._B_inv = inv(B) if matrix_rank(B) == B.shape[0] else None

    def simulate(
        self,
        c_path: np.ndarray,
        x0: Optional[np.ndarray] = None,
        method: str = "forward",
    ) -> np.ndarray:
        """
        Simulate the time path of sectoral outputs.

        Parameters
        ----------
        c_path : np.ndarray (T, n)  Non-investment final demand at each period.
        x0     : np.ndarray (n,)    Initial output vector (defaults to self.x).
        method : str                'forward' (explicit difference equation) or
                                    'matrix_exp' (matrix exponential approximation).

        Returns
        -------
        x_path : np.ndarray (T+1, n)  Sectoral output at each period.
        """
        T = c_path.shape[0]
        n = self.n
        x_path = np.zeros((T + 1, n))
        x_path[0] = x0 if x0 is not None else self.x.copy()

        if method == "forward":
            if self._B_inv is None:
                raise ValueError("Capital matrix B is singular; cannot invert.")
            for t in range(T):
                # x(t+1) = B^{-1} [(I-A) x(t) - c(t)] + x(t)
                x_path[t + 1] = (
                    self._B_inv @ (self.I_minus_A @ x_path[t] - c_path[t])
                    + x_path[t]
                )
        else:
            raise NotImplementedError(f"Method '{method}' not implemented.")

        return x_path

    def simulate_shock(
        self,
        c_path: np.ndarray,
        shock_vector: np.ndarray,
        shock_start: int,
        shock_end: int,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate baseline and shocked output paths.

        Parameters
        ----------
        c_path       : np.ndarray (T, n)  Baseline non-investment demand.
        shock_vector : np.ndarray (n,)    Additive shock to final demand per period.
        shock_start  : int                First period of shock.
        shock_end    : int                Last period of shock (inclusive).
        x0           : np.ndarray (n,)    Initial output.

        Returns
        -------
        x_base, x_shocked : np.ndarray (T+1, n) each.
        """
        c_shocked = c_path.copy()
        c_shocked[shock_start : shock_end + 1] += shock_vector

        x_base = self.simulate(c_path, x0)
        x_shocked = self.simulate(c_shocked, x0)
        return x_base, x_shocked

    def gdp_path(
        self,
        x_path: np.ndarray,
        va_coeff: np.ndarray,
    ) -> np.ndarray:
        """
        Convert output path to GDP (value added) path.

        Parameters
        ----------
        x_path   : np.ndarray (T+1, n)
        va_coeff : np.ndarray (n,)  Value-added coefficient per sector.
        """
        return x_path @ va_coeff

    def employment_path(
        self,
        x_path: np.ndarray,
        emp_coeff: np.ndarray,
    ) -> np.ndarray:
        """
        Convert output path to employment path.

        Parameters
        ----------
        x_path   : np.ndarray (T+1, n)
        emp_coeff: np.ndarray (n,)  FTE per £bn of output.
        """
        return x_path @ emp_coeff


class REEDependenceAnalyser:
    """
    Compute REE dependence metrics for UK sectors.

    Direct REE dependence:   RD_i = REE inputs to i / Total inputs to i
    Total REE dependence:    CRD  = r @ L
    REE exposure to shock:   EXP  = CRD * Δx / x
    """

    def __init__(self, leontief: StaticLeontief, ree_intensity: np.ndarray):
        self.leontief = leontief
        self.ree_intensity = ree_intensity

    def direct_dependence(self) -> np.ndarray:
        """RD_i = REE intensity of sector i (direct)."""
        return self.ree_intensity.copy()

    def total_dependence(self) -> np.ndarray:
        """CRD_j = r @ L  — total REE requirement per unit final demand of sector j."""
        return self.ree_intensity @ self.leontief.L

    def exposure_to_shock(
        self,
        theta: float,
        ree_sector_idx: int = 1,
    ) -> np.ndarray:
        """
        Output loss vector when REE supply is reduced by fraction theta.

        Parameters
        ----------
        theta           : float  Supply shock magnitude (0–1).
        ree_sector_idx  : int    Index of REE sector in the IO table.
        """
        crd = self.total_dependence()
        x = self.leontief.x
        ree_output = x[ree_sector_idx]
        delta_ree = -theta * ree_output
        # Approximate output loss via total REE dependence coefficient
        delta_x = crd * delta_ree
        return delta_x

    def summary_table(self) -> pd.DataFrame:
        s = self.leontief.sector_names
        rd = self.direct_dependence()
        crd = self.total_dependence()
        df = pd.DataFrame({
            "direct_ree_dependence": rd,
            "total_ree_dependence": crd,
            "multiplier_ratio": np.where(rd > 0, crd / rd, np.nan),
        }, index=s)
        df.index.name = "sector"
        return df
