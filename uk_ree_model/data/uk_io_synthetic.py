"""
uk_io_synthetic.py
==================
Generates a synthetic but structurally realistic UK Input-Output table
for model development and testing, calibrated to ONS sector proportions.

Replace with real ONS Supply & Use Tables once downloaded from:
https://www.ons.gov.uk/economy/nationalaccounts/supplyandusetables

Sectors (12 aggregate, expandable to 105 ONS detail):
  0  Agriculture & Mining
  1  REE & Critical Minerals (disaggregated from Mining)
  2  Food & Beverages
  3  Chemicals & Pharmaceuticals
  4  Steel & Basic Metals
  5  Electronics & Electrical Equipment
  6  Automotive & Transport Equipment
  7  Aerospace & Defence
  8  Energy (Oil, Gas, Electricity)
  9  Offshore Wind & Renewables
  10 Construction
  11 Services (aggregated)
"""

import numpy as np
import pandas as pd

SECTOR_NAMES = [
    "Agriculture & Mining",
    "REE & Critical Minerals",
    "Food & Beverages",
    "Chemicals & Pharma",
    "Steel & Basic Metals",
    "Electronics & Electrical",
    "Automotive & Transport",
    "Aerospace & Defence",
    "Energy",
    "Offshore Wind & Renewables",
    "Construction",
    "Services",
]
N = len(SECTOR_NAMES)

# Total output vector x (£bn, approximate UK 2022 scale)
X_BASE = np.array([
    45.0,    # Agriculture & Mining
     2.5,    # REE & Critical Minerals
    120.0,   # Food & Beverages
    95.0,    # Chemicals & Pharma
    60.0,    # Steel & Basic Metals
    85.0,    # Electronics & Electrical
    75.0,    # Automotive & Transport
    40.0,    # Aerospace & Defence
    150.0,   # Energy
    25.0,    # Offshore Wind & Renewables
    130.0,   # Construction
    1500.0,  # Services
])

# Direct input coefficient matrix A (manually calibrated, row = input sector, col = using sector)
# Each column sums to roughly 0.5 (leaving ~50% value added)
_A_RAW = np.array([
#   Agr  REE  Food Chem Steel Elec  Auto  Aero  Engy  Wind  Cons  Serv
    [0.10, 0.00, 0.15, 0.02, 0.01, 0.00, 0.01, 0.00, 0.02, 0.00, 0.01, 0.01],  # Agriculture
    [0.00, 0.05, 0.00, 0.01, 0.02, 0.05, 0.04, 0.06, 0.01, 0.08, 0.01, 0.00],  # REE
    [0.05, 0.00, 0.12, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02],  # Food
    [0.02, 0.03, 0.03, 0.15, 0.04, 0.03, 0.04, 0.03, 0.06, 0.04, 0.03, 0.01],  # Chemicals
    [0.01, 0.02, 0.01, 0.02, 0.18, 0.04, 0.10, 0.08, 0.03, 0.07, 0.10, 0.01],  # Steel
    [0.01, 0.04, 0.01, 0.03, 0.02, 0.15, 0.06, 0.08, 0.04, 0.10, 0.02, 0.03],  # Electronics
    [0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.12, 0.04, 0.03, 0.05, 0.04, 0.01],  # Automotive
    [0.00, 0.01, 0.00, 0.01, 0.02, 0.03, 0.02, 0.18, 0.02, 0.06, 0.01, 0.01],  # Aerospace
    [0.05, 0.04, 0.03, 0.05, 0.04, 0.02, 0.03, 0.02, 0.10, 0.05, 0.04, 0.03],  # Energy
    [0.00, 0.02, 0.00, 0.01, 0.01, 0.02, 0.01, 0.01, 0.03, 0.08, 0.02, 0.01],  # Offshore Wind
    [0.02, 0.01, 0.01, 0.02, 0.03, 0.02, 0.02, 0.02, 0.03, 0.06, 0.12, 0.02],  # Construction
    [0.08, 0.06, 0.08, 0.08, 0.06, 0.08, 0.07, 0.07, 0.08, 0.07, 0.09, 0.18],  # Services
])

# REE intensity vector (£ of REE input per £ of output), by sector
# Derived from BGS CMIC estimates and JRC/IRENA engineering data
REE_INTENSITY = np.array([
    0.000,  # Agriculture
    1.000,  # REE sector itself
    0.000,  # Food
    0.002,  # Chemicals (catalyst use)
    0.003,  # Steel (cerium additions)
    0.025,  # Electronics (Nd, Eu, Y in components)
    0.018,  # Automotive (NdFeB motors, ~1.5 kg/EV)
    0.022,  # Aerospace & Defence (Dy, Tb, Sm, Y)
    0.002,  # Energy (refining catalysts)
    0.030,  # Offshore Wind (4t magnets/turbine, Nd, Dy)
    0.001,  # Construction
    0.001,  # Services
])

# Capital coefficients matrix B (Leontief dynamic IO)
# b_ij = £ of capital good i per £ increase in capacity of sector j
_B_RAW = np.array([
#   Agr  REE  Food Chem Steel Elec  Auto  Aero  Engy  Wind  Cons  Serv
    [0.05, 0.00, 0.03, 0.01, 0.01, 0.00, 0.01, 0.00, 0.01, 0.00, 0.01, 0.00],
    [0.00, 0.02, 0.00, 0.01, 0.01, 0.02, 0.02, 0.03, 0.01, 0.04, 0.00, 0.00],
    [0.01, 0.00, 0.05, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01],
    [0.01, 0.01, 0.01, 0.06, 0.02, 0.02, 0.02, 0.02, 0.03, 0.02, 0.01, 0.00],
    [0.01, 0.01, 0.01, 0.02, 0.08, 0.02, 0.05, 0.04, 0.02, 0.04, 0.05, 0.01],
    [0.01, 0.02, 0.01, 0.02, 0.01, 0.07, 0.03, 0.04, 0.02, 0.05, 0.01, 0.02],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.06, 0.02, 0.01, 0.03, 0.02, 0.01],
    [0.00, 0.01, 0.00, 0.01, 0.01, 0.02, 0.01, 0.08, 0.01, 0.03, 0.01, 0.00],
    [0.02, 0.02, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.06, 0.03, 0.02, 0.01],
    [0.00, 0.01, 0.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.10, 0.01, 0.00],
    [0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02, 0.04, 0.08, 0.01],
    [0.03, 0.02, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.06],
])

# UK REE import share from China (by REE type, weighted average per sector)
# Source: BGS CMIC 2024; EU import data as proxy for UK pre-Brexit
UK_CHINA_IMPORT_SHARE = np.array([
    0.00,   # Agriculture
    0.70,   # REE sector (Chinese ore/compounds)
    0.00,   # Food
    0.50,   # Chemicals (REE catalysts)
    0.65,   # Steel
    0.72,   # Electronics
    0.75,   # Automotive
    0.60,   # Aerospace (some diversification via US supply chains)
    0.50,   # Energy
    0.78,   # Offshore Wind (NdFeB magnets almost entirely Chinese)
    0.40,   # Construction
    0.10,   # Services
])

# Final demand vector y (£bn): derived from Leontief identity y = (I-A)@x
# This ensures L_inv @ y == x exactly (Leontief identity).
# Sectors where y < 0 (e.g. REE, Steel) are net-import sectors: domestic output
# is fully absorbed by intermediate demand; net final demand is met via imports.
# Negative y is economically valid and must NOT be floored — flooring breaks the
# Leontief identity and corrupts multiplier and REE-dependence calculations.
def _compute_y_base(A, x):
    return (np.eye(len(x)) - A) @ x   # exact identity; negatives = net imports

Y_BASE = _compute_y_base(_A_RAW, X_BASE)

# Employment coefficients (thousand FTE per £bn output)
EMPLOYMENT_COEFF = np.array([
    15.0,   # Agriculture
     8.0,   # REE
    12.0,   # Food
     9.0,   # Chemicals
    10.0,   # Steel
    11.0,   # Electronics
    13.0,   # Automotive
    12.0,   # Aerospace
     5.0,   # Energy
    10.0,   # Offshore Wind
    18.0,   # Construction
    22.0,   # Services
])


def get_io_data() -> dict:
    """Return the synthetic UK IO dataset as a dictionary of arrays and DataFrames."""
    A = _A_RAW.copy()
    B = _B_RAW.copy()
    return {
        "sector_names": SECTOR_NAMES,
        "n_sectors": N,
        "A": A,
        "B": B,
        "x": X_BASE.copy(),
        "y": Y_BASE.copy(),
        "ree_intensity": REE_INTENSITY.copy(),
        "china_import_share": UK_CHINA_IMPORT_SHARE.copy(),
        "employment_coeff": EMPLOYMENT_COEFF.copy(),
    }


def as_dataframes(data: dict) -> dict:
    """Wrap matrices as labelled DataFrames for inspection."""
    s = data["sector_names"]
    return {
        "A": pd.DataFrame(data["A"], index=s, columns=s),
        "B": pd.DataFrame(data["B"], index=s, columns=s),
        "x": pd.Series(data["x"], index=s, name="output_£bn"),
        "y": pd.Series(data["y"], index=s, name="final_demand_£bn"),
        "ree_intensity": pd.Series(data["ree_intensity"], index=s, name="ree_intensity"),
        "china_import_share": pd.Series(data["china_import_share"], index=s, name="china_share"),
        "employment_coeff": pd.Series(data["employment_coeff"], index=s, name="emp_per_£bn"),
    }


if __name__ == "__main__":
    data = get_io_data()
    dfs = as_dataframes(data)
    print("=== UK Synthetic IO Table (£bn) ===")
    print(dfs["A"].round(3).to_string())
    print("\n=== Final Demand (£bn) ===")
    print(dfs["y"])
    print("\n=== REE Intensity by Sector ===")
    print(dfs["ree_intensity"])
