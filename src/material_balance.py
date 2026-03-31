"""
material_balance.py — Gas material balance and Mattar & Anderson FMB.

Standard gas MBE (p-form):
    p̄/z = (pi/zi) · (1 − Gp/G)

Mattar & Anderson FMB workflow (pseudopressure form):
    1. Measure pwf and q at each timestep.
    2. Compute m(pwf) from PVT table.
    3. Back-calculate m(p̄) = m(pwf) + a·q + b·q²  (LIT inflow, m(p)-form).
    4. Invert m(p̄) → p̄ using bisection on the pseudopressure table.
    5. Compute p̄/z at p̄.
    6. Plot p̄/z vs Gp → linear; slope gives G.

Reference: Mattar, L. & Anderson, D.M. (2005). "A Systematic and Comprehensive
Methodology for Advanced Analysis of Production Data." SPE 98931.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.gas_properties import (
    calc_z_factor,
    calc_pseudopressure,
    invert_pseudopressure,
)


# ---------------------------------------------------------------------------
# Standard MBE utilities
# ---------------------------------------------------------------------------

def calc_p_over_z(
    p: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    method: str = "hall_yarborough",
) -> float:
    """
    Return p/z at a given pressure.

    Returns
    -------
    p_over_z : float  [psia]
    """
    z = calc_z_factor(p, T_R, Tpc, Ppc, method)
    return p / z


def calc_pbar_from_mbe(
    Gp_MMCF: float,
    G_MMCF: float,
    pi: float,
    zi: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    p_low: float = 14.7,
    method: str = "hall_yarborough",
) -> float:
    """
    Invert the MBE to recover average reservoir pressure given Gp.

        p̄/z(p̄) = (pi/zi) · (1 − Gp/G)

    Solve via scipy.optimize.brentq on the interval [p_low, pi].

    Parameters
    ----------
    Gp_MMCF : float  Cumulative production [MMCF]
    G_MMCF  : float  OGIP [MMCF]
    pi      : float  Initial pressure [psia]
    zi      : float  z-factor at pi
    T_R     : float  [°R]
    Tpc     : float  [°R]
    Ppc     : float  [psia]

    Returns
    -------
    p_bar : float  [psia]
    """
    target = (pi / zi) * (1.0 - Gp_MMCF / G_MMCF)
    if target <= 0.0:
        return p_low

    def residual(p):
        z = calc_z_factor(p, T_R, Tpc, Ppc, method)
        return p / z - target

    # If target equals pi/zi (Gp=0) return pi directly
    if abs(residual(pi)) < 1e-6:
        return pi

    try:
        p_bar = brentq(residual, p_low, pi, xtol=1e-4, maxiter=200)
    except ValueError:
        # Fallback: linear interpolation
        p_bar = pi * (1.0 - Gp_MMCF / G_MMCF)

    return p_bar


def run_standard_mbe(
    Gp_array: np.ndarray,
    G_MMCF: float,
    pi: float,
    zi: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    method: str = "hall_yarborough",
) -> pd.DataFrame:
    """
    Compute the full standard MBE depletion path.

    Returns
    -------
    pd.DataFrame with columns: Gp_MMCF, p_bar, z, p_over_z
    """
    rows = []
    for Gp in Gp_array:
        p_bar = calc_pbar_from_mbe(Gp, G_MMCF, pi, zi, T_R, Tpc, Ppc, method=method)
        z = calc_z_factor(p_bar, T_R, Tpc, Ppc, method)
        rows.append(
            {"Gp_MMCF": Gp, "p_bar": p_bar, "z": z, "p_over_z": p_bar / z}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FMB — Mattar & Anderson (pseudopressure form)
# ---------------------------------------------------------------------------

def calc_pbar_from_flowing_mp(
    mp_wf: float,
    q_Mcfd: float,
    a_mp: float,
    b_mp: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    gamma_g: float,
    pi: float,
) -> tuple:
    """
    FMB back-calculation of average reservoir pressure using pseudopressure.

    Workflow:
        m(p̄) = m(pwf) + a·q + b·q²
        p̄    = invert_pseudopressure(m(p̄))

    Parameters
    ----------
    mp_wf   : float  m(pwf) [psia²/cp]
    q_Mcfd  : float  Production rate [Mcfd]
    a_mp    : float  Darcy mp coefficient [psia²/cp per Mcfd]
    b_mp    : float  Non-Darcy mp coefficient [psia²/cp per Mcfd²]

    Returns
    -------
    (p_bar, mp_bar) : (float [psia], float [psia²/cp])
    """
    mp_bar = mp_wf + a_mp * q_Mcfd + b_mp * q_Mcfd ** 2
    p_bar = invert_pseudopressure(
        mp_bar, T_R, Tpc, Ppc, gamma_g, p_high=pi * 1.05
    )
    return p_bar, mp_bar


def calc_pbar_from_flowing_p(
    pwf: float,
    q_MMcfd: float,
    A: float,
    B: float,
) -> float:
    """
    FMB back-calculation using simplified pressure-based inflow (approximate).

        p̄ = pwf + A·q + B·q²

    Useful for quick verification and comparison with pseudopressure method.
    Accuracy degrades at lower pressures where μz varies significantly.

    Returns
    -------
    p_bar : float  [psia]
    """
    return pwf + A * q_MMcfd + B * q_MMcfd ** 2


# ---------------------------------------------------------------------------
# FMB regression — estimate G from flowing history
# ---------------------------------------------------------------------------

def estimate_G_from_fmb(
    Gp_array: np.ndarray,
    p_bar_array: np.ndarray,
    z_array: np.ndarray,
    pi: float,
    zi: float,
) -> dict:
    """
    Linear regression on the FMB plot (p̄/z vs Gp) to estimate OGIP.

    MBE line: p̄/z = (pi/zi) − (pi/zi)/G · Gp
        Slope     m = −(pi/zi) / G   =>  G = −(pi/zi) / m
        Intercept b = pi/zi           (cross-check)

    Uses numpy.polyfit (degree 1) weighted equally.

    Parameters
    ----------
    Gp_array    : np.ndarray  Cumulative production [MMCF]
    p_bar_array : np.ndarray  Average reservoir pressure [psia]
    z_array     : np.ndarray  z-factor at p̄
    pi          : float       Initial pressure [psia]
    zi          : float       z-factor at pi

    Returns
    -------
    dict with:
        G_fmb_MMCF : float
        slope      : float
        intercept  : float
        R_squared  : float
        pi_zi_fitted : float  (fitted intercept, should ≈ pi/zi)
    """
    pz_array = p_bar_array / z_array

    coeffs = np.polyfit(Gp_array, pz_array, 1)
    slope, intercept = coeffs[0], coeffs[1]

    # G from slope
    pi_zi = pi / zi
    G_fmb = -pi_zi / slope if abs(slope) > 1e-12 else np.inf

    # R²
    pz_fit = np.polyval(coeffs, Gp_array)
    ss_res = np.sum((pz_array - pz_fit) ** 2)
    ss_tot = np.sum((pz_array - np.mean(pz_array)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "G_fmb_MMCF": G_fmb,
        "slope": slope,
        "intercept": intercept,
        "R_squared": R2,
        "pi_zi_target": pi_zi,
        "pi_zi_fitted": intercept,
    }


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------

def calc_fmb_error(G_fmb: float, G_volumetric: float) -> dict:
    """
    Compute the FMB estimation error relative to the volumetric OGIP.

    Returns
    -------
    dict with: G_fmb, G_volumetric, absolute_error_MMCF, percent_error
    """
    abs_err = abs(G_fmb - G_volumetric)
    pct_err = 100.0 * (G_fmb - G_volumetric) / G_volumetric
    return {
        "G_fmb_MMCF": G_fmb,
        "G_volumetric_MMCF": G_volumetric,
        "absolute_error_MMCF": abs_err,
        "percent_error": pct_err,
    }
