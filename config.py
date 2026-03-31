"""
config.py — Central parameter store for the ResearchPNGE project.

All base-case reservoir inputs and known validation targets from the
screenshot are defined here. Every other module imports from this file.
"""

# ---------------------------------------------------------------------------
# Base-case reservoir and well parameters (from screenshot)
# ---------------------------------------------------------------------------
BASE_CASE = {
    # Fluid
    "pi": 4000.0,        # psia  — initial reservoir pressure
    "T_F": 120.0,        # °F    — reservoir temperature
    "gamma_g": 0.47,     # —     — gas specific gravity (air = 1)
    "y_N2": 0.00,        # mole fraction N2
    "y_CO2": 0.00,       # mole fraction CO2
    "y_H2S": 0.00,       # mole fraction H2S

    # Geometry
    "A_acres": 5000.0,   # acres — total drainage area
    "h": 50.0,           # ft    — net pay thickness
    "phi": 0.10,         # —     — porosity (fraction)
    "Swi": 0.35,         # —     — irreducible water saturation

    # Rock / flow
    "k": 20.0,           # md    — permeability
    "s": 0.0,            # —     — skin factor
    "D": 0.0003,         # Mcfd⁻¹— non-Darcy (turbulence) coefficient

    # Wells and production
    "N_wells": 5,
    "rw": 0.5,           # ft    — wellbore radius
    "q_MMcfd": 3.0,      # MMcfd — total field production rate

    # Simulation control
    "pwf_min": 500.0,    # psia  — abandonment flowing bottomhole pressure
    "dt_days": 30.0,     # days  — timestep
    "t_max_days": 3650.0,# days  — total simulation period (10 years)
}

# ---------------------------------------------------------------------------
# Known validation targets (computed externally and shown in screenshot)
# ---------------------------------------------------------------------------
VALIDATION_TARGETS = {
    "zi": 0.926,              # z-factor at initial conditions
    "mu_i_cp": 0.0194,        # cp   — gas viscosity at initial conditions
    "Bgi_RB_Mcf": 0.677,      # RB/Mcf — initial gas FVF
    "re_ft": 3724.0,          # ft   — per-well drainage radius
    "pi_over_zi_psia": 4319.0,# psia — initial p/z
    "G_MMCF": 186342.0,       # MMCF — volumetric OGIP
    "A_psia_MMcfd": 15.105,   # psia/MMcfd  — Darcy inflow coefficient
    "B_psia_MMcfd2": 0.55,    # psia/MMcfd² — non-Darcy inflow coefficient
}

# ---------------------------------------------------------------------------
# Tolerances for validation checks
# ---------------------------------------------------------------------------
TOL = {
    # PVT correlations carry inherent ±1-2% uncertainty between different implementations.
    # Sutton (1985) pseudo-critical props vs other correlations can differ by ~1.6% in z.
    "zi": 0.020,              # ±2 % (correlation-to-correlation uncertainty)
    "mu_cp": 0.003,           # ±0.003 cp
    "Bg_RB_Mcf": 0.015,       # ±1.5 % (propagated from zi)
    "re_ft": 5.0,             # ±5 ft (geometric, essentially exact)
    "G_MMCF": 3000.0,         # ±3000 MMCF (~1.6 %, propagated from Bgi)
    "A_psia_MMcfd": 1.0,      # ±1.0 psia/MMcfd (~6 %, from μ·z product)
    "B_psia_MMcfd2": 0.05,    # ±0.05 psia/MMcfd² (~9 %, from μ·z product)
    "G_fmb_pct": 1.0,         # FMB must recover G within 1 % on noise-free data
}

# ---------------------------------------------------------------------------
# Non-Darcy sensitivity range
# ---------------------------------------------------------------------------
D_SENSITIVITY_VALUES = [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.002]  # Mcfd⁻¹

# ---------------------------------------------------------------------------
# ML / Monte Carlo settings
# ---------------------------------------------------------------------------
MC_SETTINGS = {
    "n_samples": 5000,
    "seed": 42,
    # Sampling ranges [low, high]
    "pi_range": [1000.0, 8000.0],
    "T_F_range": [100.0, 300.0],
    "gamma_g_range": [0.55, 0.75],
    "A_acres_range": [500.0, 20000.0],
    "h_range": [10.0, 200.0],
    "phi_range": [0.05, 0.25],
    "Swi_range": [0.15, 0.45],
    "k_range": [0.1, 100.0],
    "D_range": [0.0, 0.002],
    "s_range": [-3.0, 10.0],
    "N_wells_range": [1, 20],
    "q_MMcfd_range": [0.5, 15.0],
    "pwf_min_range": [200.0, 1000.0],
}
