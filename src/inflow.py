"""
inflow.py — LIT (Laminar-Inertial-Turbulent) inflow performance for gas wells.

Two forms are implemented:

  Pressure-based (p-form):
    Δp = p̄ − pwf = A·q + B·q²
    A [psia/MMcfd]   = 1422·μ·z·T_R·(ln(re/rw) − 0.75 + s) / (k·h)
    B [psia/MMcfd²]  = 1422·μ·z·T_R·D / (k·h)
    — matches screenshot values (μ, z evaluated at initial conditions)

  Pseudopressure-based (m(p)-form):
    Δm(p) = m(p̄) − m(pwf) = a·q + b·q²
    a [psia²/cp per Mcfd]   = 711·T_R·(ln(re/rw) − 0.75 + s) / (k·h)
    b [psia²/cp per Mcfd²]  = 711·T_R·D / (k·h)
    — used by Mattar & Anderson FMB (pressure-independent, rigorous)

Note on units
-------------
The constant 1422 arises from field-unit conversion for the p-form
(q in MMcfd).  The constant 711 arises for the m(p)-form (q in Mcfd,
because the 1/μz dependence is absorbed into the integral).

When converting:
    a [psia²/cp / Mcfd] = A_pressure_form · (μ·z) / (2·p̄) × unit_factor
but this shortcut is approximate. The pseudopressure form is always preferred.
"""

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Coefficient helpers — pressure-based form
# ---------------------------------------------------------------------------

def calc_A_coefficient(
    mu_g: float,
    z: float,
    T_R: float,
    re: float,
    rw: float,
    s: float,
    k: float,
    h: float,
    pi_psia: float = 4000.0,
) -> float:
    """
    Darcy (laminar) flow coefficient — linearized pressure form.

    Derivation
    ----------
    Starting from the pressure-squared Darcy inflow equation:
        p̄² − pwf² = A_sq · q   where A_sq [psia²/Mcfd] = 1422·μ·z·T·ln_term/(k·h)

    Linearizing at initial conditions (p̄ + pwf ≈ 2·pi):
        p̄ − pwf ≈ A_sq · q / (2·pi)

    Converting q from Mcfd to MMcfd (×1000):
        A [psia/MMcfd] = A_sq · 1000 / (2·pi)
                       = 1422·μ·z·T·ln_term·1000 / (k·h·2·pi)

    Parameters
    ----------
    mu_g     : float  [cp]
    z        : float  z-factor at initial conditions
    T_R      : float  [°R]
    re       : float  drainage radius [ft]
    rw       : float  wellbore radius [ft]
    s        : float  skin factor
    k        : float  permeability [md]
    h        : float  net pay [ft]
    pi_psia  : float  initial reservoir pressure [psia]  (for linearization)

    Returns
    -------
    A : float  [psia/MMcfd]

    Validation target: A = 15.105 psia/MMcfd
    """
    ln_term = math.log(re / rw) - 0.75 + s
    A_sq = 1422.0 * mu_g * z * T_R * ln_term / (k * h)   # psia²/Mcfd
    return A_sq * 1000.0 / (2.0 * pi_psia)               # psia/MMcfd


def calc_B_coefficient(
    mu_g: float,
    z: float,
    T_R: float,
    D: float,
    k: float,
    h: float,
    pi_psia: float = 4000.0,
) -> float:
    """
    Non-Darcy (inertial/turbulent) flow coefficient — linearized pressure form.

    Derivation
    ----------
    Starting from the pressure-squared non-Darcy term:
        B_sq [psia²/Mcfd²] = 1422·μ·z·T·D / (k·h)   where D in Mcfd⁻¹

    Linearizing and converting q to MMcfd (×1000 per q factor):
        B [psia/MMcfd²] = B_sq · (1000)² / (2·pi)
                        = 1422·μ·z·T·D·10⁶ / (k·h·2·pi)

    Parameters
    ----------
    D        : float  Non-Darcy coefficient [Mcfd⁻¹]
    pi_psia  : float  initial reservoir pressure [psia]  (for linearization)

    Returns
    -------
    B : float  [psia/MMcfd²]

    Validation target: B = 0.55 psia/MMcfd²
    """
    B_sq = 1422.0 * mu_g * z * T_R * D / (k * h)         # psia²/Mcfd²
    return B_sq * 1.0e6 / (2.0 * pi_psia)                 # psia/MMcfd²


# ---------------------------------------------------------------------------
# Coefficient helpers — pseudopressure form (Mattar & Anderson)
# ---------------------------------------------------------------------------

def calc_a_mp_coefficient(
    T_R: float,
    re: float,
    rw: float,
    s: float,
    k: float,
    h: float,
) -> float:
    """
    Darcy coefficient in pseudopressure form.

        a [psia²/cp per Mcfd] = 711 · T_R · (ln(re/rw) − 0.75 + s) / (k · h)

    The constant 711 = 1422 / 2 because m(p) already includes the factor
    of 2 in its definition (m(p) = 2 ∫ p/(μz) dp).
    """
    ln_term = math.log(re / rw) - 0.75 + s
    return 711.0 * T_R * ln_term / (k * h)


def calc_b_mp_coefficient(
    T_R: float,
    D: float,
    k: float,
    h: float,
) -> float:
    """
    Non-Darcy coefficient in pseudopressure form.

        b [psia²/cp per Mcfd²] = 711 · T_R · D / (k · h)
    """
    return 711.0 * T_R * D / (k * h)


# ---------------------------------------------------------------------------
# Pressure-based solvers
# ---------------------------------------------------------------------------

def calc_pwf_from_rate(
    q_MMcfd: float,
    p_bar: float,
    A: float,
    B: float,
) -> float:
    """
    Given average reservoir pressure and rate, compute flowing BHP.

        pwf = p̄ − A·q − B·q²

    Parameters
    ----------
    q_MMcfd : float  [MMcfd]
    p_bar   : float  [psia]
    A       : float  [psia/MMcfd]
    B       : float  [psia/MMcfd²]

    Returns
    -------
    pwf : float  [psia]
    """
    return p_bar - A * q_MMcfd - B * q_MMcfd ** 2


def calc_rate_from_pressures(
    p_bar: float,
    pwf: float,
    A: float,
    B: float,
) -> float:
    """
    Given p̄ and pwf, solve the quadratic for q.

        B·q² + A·q − (p̄ − pwf) = 0
        q = (−A + sqrt(A² + 4·B·Δp)) / (2·B)   [positive root]

    If B ≈ 0 (pure Darcy): q = Δp / A.

    Returns
    -------
    q : float  [MMcfd]
    """
    delta_p = p_bar - pwf
    if delta_p <= 0.0:
        return 0.0
    if abs(B) < 1e-12:
        return delta_p / A
    discriminant = A ** 2 + 4.0 * B * delta_p
    return (-A + math.sqrt(discriminant)) / (2.0 * B)


def calc_AOF(
    p_bar: float,
    A: float,
    B: float,
    pwf_min: float = 14.7,
) -> float:
    """
    Absolute Open Flow potential — rate at pwf = atmospheric.

    Returns
    -------
    AOF : float  [MMcfd]
    """
    return calc_rate_from_pressures(p_bar, pwf_min, A, B)


def build_IPR_curve(
    p_bar: float,
    A: float,
    B: float,
    pwf_min: float = 14.7,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Compute the full IPR (inflow performance relationship) curve.

    Returns
    -------
    pd.DataFrame with columns: q_MMcfd, pwf
    """
    aof = calc_AOF(p_bar, A, B, pwf_min)
    q_arr = np.linspace(0.0, aof, n_points)
    pwf_arr = p_bar - A * q_arr - B * q_arr ** 2
    pwf_arr = np.maximum(pwf_arr, pwf_min)
    return pd.DataFrame({"q_MMcfd": q_arr, "pwf": pwf_arr})


# ---------------------------------------------------------------------------
# Pressure-drop decomposition
# ---------------------------------------------------------------------------

def decompose_pressure_drop(
    q_MMcfd: float,
    A: float,
    B: float,
) -> dict:
    """
    Decompose total pressure drop into Darcy and non-Darcy components.

    Returns
    -------
    dict with keys:
        delta_p_darcy     : float  [psia]   A·q
        delta_p_nondarcy  : float  [psia]   B·q²
        delta_p_total     : float  [psia]
        nondarcy_fraction : float  [0–1]    B·q² / total
    """
    dp_darcy = A * q_MMcfd
    dp_nd = B * q_MMcfd ** 2
    dp_total = dp_darcy + dp_nd
    nd_frac = dp_nd / dp_total if dp_total > 0 else 0.0
    return {
        "delta_p_darcy": dp_darcy,
        "delta_p_nondarcy": dp_nd,
        "delta_p_total": dp_total,
        "nondarcy_fraction": nd_frac,
    }


# ---------------------------------------------------------------------------
# Pseudopressure-based solvers (for Mattar & Anderson FMB)
# ---------------------------------------------------------------------------

def calc_mp_bar_from_rate(
    q_Mcfd: float,
    mp_wf: float,
    a_mp: float,
    b_mp: float,
) -> float:
    """
    Recover average reservoir pseudo-pressure from flowing conditions.

        m(p̄) = m(pwf) + a·q + b·q²   [psia²/cp]

    Parameters
    ----------
    q_Mcfd  : float  Production rate [Mcfd]
    mp_wf   : float  Pseudo-pressure at pwf  [psia²/cp]
    a_mp    : float  Darcy mp coefficient    [psia²/cp per Mcfd]
    b_mp    : float  Non-Darcy mp coefficient[psia²/cp per Mcfd²]

    Returns
    -------
    mp_bar : float  [psia²/cp]
    """
    return mp_wf + a_mp * q_Mcfd + b_mp * q_Mcfd ** 2


def calc_q_from_mp(
    mp_bar: float,
    mp_wf: float,
    a_mp: float,
    b_mp: float,
) -> float:
    """
    Solve for rate given pseudo-pressures (quadratic in q).

        b·q² + a·q − (m(p̄) − m(pwf)) = 0

    Returns
    -------
    q : float  [Mcfd]
    """
    delta_mp = mp_bar - mp_wf
    if delta_mp <= 0.0:
        return 0.0
    if abs(b_mp) < 1e-12:
        return delta_mp / a_mp
    disc = a_mp ** 2 + 4.0 * b_mp * delta_mp
    return (-a_mp + math.sqrt(disc)) / (2.0 * b_mp)
