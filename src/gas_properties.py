"""
gas_properties.py — PVT calculations for natural gas.

Implements:
  - Pseudo-critical properties  (Sutton 1985 + Wichert-Aziz acid-gas correction)
  - z-factor                    (Hall-Yarborough 1974 iterative; Papay explicit)
  - Gas viscosity               (Lee-Gonzalez-Eakin 1966)
  - Gas formation volume factor (Bg)
  - Pseudo-pressure             (m(p) = 2 ∫ p/(μ·z) dp)
  - Full PVT table builder

Units: field units throughout (psia, °R, cp, RB/Mcf, psia²/cp).
"""

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pseudo-critical properties
# ---------------------------------------------------------------------------

def calc_pseudocritical_properties(
    gamma_g: float,
    y_N2: float = 0.0,
    y_CO2: float = 0.0,
    y_H2S: float = 0.0,
) -> tuple:
    """
    Sutton (1985) correlations with Wichert-Aziz (1972) acid-gas correction.

    Parameters
    ----------
    gamma_g : float
        Gas specific gravity (air = 1.0).
    y_N2, y_CO2, y_H2S : float
        Mole fractions of non-hydrocarbon components.

    Returns
    -------
    Tpc : float  [°R]
    Ppc : float  [psia]
    """
    # Sutton hydrocarbon correlations
    Tpc_hc = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g ** 2
    Ppc_hc = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g ** 2

    # Wichert-Aziz correction (only matters when CO2 or H2S > 0)
    A = y_CO2 + y_H2S
    B = y_H2S
    epsilon = 120.0 * (A ** 0.9 - A ** 1.6) + 15.0 * (B ** 0.5 - B ** 4.0)

    Tpc = Tpc_hc - epsilon
    Ppc = Ppc_hc * Tpc / (Tpc_hc + B * (1.0 - B) * epsilon)

    # Apply N2 correction (Kay's mixing rule adjustment)
    # For small N2 fractions this is negligible; included for completeness
    Tpc_N2 = 227.0   # °R
    Ppc_N2 = 493.0   # psia
    y_hc = 1.0 - y_N2 - y_CO2 - y_H2S
    Tpc = y_hc * Tpc + y_N2 * Tpc_N2
    Ppc = y_hc * Ppc + y_N2 * Ppc_N2

    return Tpc, Ppc


# ---------------------------------------------------------------------------
# z-factor
# ---------------------------------------------------------------------------

def calc_z_factor_hall_yarborough(Ppr: float, Tpr: float) -> float:
    """
    Hall-Yarborough (1974) — iterative Newton-Raphson solution.

    Equation to solve:  F(Y) = 0
        F(Y) = -A·Ppr + (Y+Y²+Y³-Y⁴)/(1-Y)³
               - (14.76t - 9.76t² + 4.58t³)·Y²
               + (90.7t - 242.2t² + 42.4t³)·Y^(2.18+2.82t)

    where A = 0.06125·t·exp(-1.2·(1-t)²)  [strictly positive]
    and z = A·Ppr / Y

    Parameters
    ----------
    Ppr : float   Pseudo-reduced pressure   = p / Ppc
    Tpr : float   Pseudo-reduced temperature = T_R / Tpc

    Returns
    -------
    z : float
    """
    t = 1.0 / Tpr
    # A is POSITIVE
    A = 0.06125 * t * math.exp(-1.2 * (1.0 - t) ** 2)

    # Composite coefficients
    c1 = 14.76 * t - 9.76 * t ** 2 + 4.58 * t ** 3
    c2 = 90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3
    c3 = 2.18 + 2.82 * t

    # Initial guess (from original HY paper)
    y = max(A * Ppr / (1.0 + A * Ppr + (A * Ppr) ** 2), 1e-8)

    for _ in range(2000):
        y2 = y * y
        y3 = y2 * y
        y4 = y3 * y
        one_minus_y = 1.0 - y

        # F(Y) — correct sign convention
        F = (
            -A * Ppr
            + (y + y2 + y3 - y4) / one_minus_y ** 3
            - c1 * y2
            + c2 * y ** c3
        )

        # F'(Y) — analytical derivative
        dFdy = (
            (1.0 + 4.0 * y + 4.0 * y2 - 4.0 * y3 + y4) / one_minus_y ** 4
            - 2.0 * c1 * y
            + c3 * c2 * y ** (c3 - 1.0)
        )

        if abs(dFdy) < 1e-30:
            break

        y_new = y - F / dFdy
        y_new = max(min(y_new, 0.999), 1e-8)

        if abs(y_new - y) < 1e-10:
            y = y_new
            break
        y = y_new

    z = A * Ppr / y
    return z


def calc_z_factor_papay(Ppr: float, Tpr: float) -> float:
    """
    Papay (1968) — explicit, fast, less accurate.
    Good for cross-checking and low-accuracy scenarios.
    """
    z = (
        1.0
        - (3.52 * Ppr) / (10.0 ** (0.9813 * Tpr))
        + (0.274 * Ppr ** 2) / (10.0 ** (0.8157 * Tpr))
    )
    return z


def calc_z_factor(
    p: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    method: str = "hall_yarborough",
) -> float:
    """
    Dispatcher for z-factor correlations.

    Parameters
    ----------
    p     : float  [psia]
    T_R   : float  [°R]
    Tpc   : float  [°R]
    Ppc   : float  [psia]
    method: str    'hall_yarborough' | 'papay'
    """
    Ppr = p / Ppc
    Tpr = T_R / Tpc
    if method == "papay":
        return calc_z_factor_papay(Ppr, Tpr)
    return calc_z_factor_hall_yarborough(Ppr, Tpr)


# ---------------------------------------------------------------------------
# Gas viscosity
# ---------------------------------------------------------------------------

def calc_gas_viscosity_lee_gonzalez(
    p: float,
    T_R: float,
    z: float,
    gamma_g: float,
) -> float:
    """
    Lee-Gonzalez-Eakin (1966) gas viscosity correlation.

    Parameters
    ----------
    p       : float  [psia]
    T_R     : float  [°R]
    z       : float  z-factor (dimensionless)
    gamma_g : float  gas specific gravity

    Returns
    -------
    mu_g : float  [cp]
    """
    M_g = 28.97 * gamma_g                          # molecular weight [lb/lb-mol]
    rho_g = (p * M_g) / (z * 10.73 * T_R)         # density [lb/ft³]
    rho_g_gcc = rho_g / 62.4                        # convert to g/cc

    K = ((9.4 + 0.02 * M_g) * T_R ** 1.5) / (209.0 + 19.0 * M_g + T_R)
    X = 3.5 + (986.0 / T_R) + 0.01 * M_g
    Y = 2.4 - 0.2 * X

    mu_g = 1e-4 * K * math.exp(X * rho_g_gcc ** Y)   # cp
    return mu_g


# ---------------------------------------------------------------------------
# Gas formation volume factor
# ---------------------------------------------------------------------------

def calc_Bg(p: float, T_R: float, z: float) -> float:
    """
    Gas formation volume factor [RB/Mcf].

    Derivation (field units):
        Bg [res bbl / scf] = (14.7 / 520) * (z * T_R / p) / 5.615
        Bg [RB / Mcf]      = Bg_per_scf * 1000
        => Bg = (14.7 * 1000) / (520 * 5.615) * z * T_R / p
               = 5.035 * z * T_R / p   ... but common constant is 0.02827

    Correct derivation:
        Bg [RB/Mcf] = (14.7 * 1000) / (520 * 5.615) * z * T_R / p
                    = 5.035 * z * T_R / p

    Note: 0.02827 = 14.7/520 gives Bg in ft³/scf (NOT RB/Mcf).
    """
    return (14.7 * 1000.0 / (520.0 * 5.615)) * z * T_R / p


# ---------------------------------------------------------------------------
# Pseudo-pressure
# ---------------------------------------------------------------------------

def calc_pseudopressure(
    p: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    gamma_g: float,
    p_base: float = 14.7,
    n_points: int = 200,
    method: str = "hall_yarborough",
) -> float:
    """
    Real-gas pseudo-pressure [psia²/cp]:
        m(p) = 2 ∫_{p_base}^{p}  p' / (μ(p') · z(p'))  dp'

    Uses numpy.trapz (trapezoidal rule) over n_points pressure points.
    """
    if abs(p - p_base) < 1e-6:
        return 0.0

    p_arr = np.linspace(p_base, p, n_points)
    integrand = np.empty(n_points)

    for i, pi in enumerate(p_arr):
        zi = calc_z_factor(pi, T_R, Tpc, Ppc, method)
        mu_i = calc_gas_viscosity_lee_gonzalez(pi, T_R, zi, gamma_g)
        integrand[i] = pi / (mu_i * zi)

    return 2.0 * float(np.trapezoid(integrand, p_arr))


def invert_pseudopressure(
    mp_target: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    gamma_g: float,
    p_low: float = 14.7,
    p_high: float = 15000.0,
    tol: float = 1.0,
    n_points: int = 200,
) -> float:
    """
    Find p such that m(p) = mp_target via bisection.

    Parameters
    ----------
    mp_target : float  [psia²/cp]
    p_low, p_high : float  search bracket [psia]
    tol : float  convergence tolerance [psia²/cp]

    Returns
    -------
    p : float  [psia]
    """
    for _ in range(100):
        p_mid = 0.5 * (p_low + p_high)
        mp_mid = calc_pseudopressure(p_mid, T_R, Tpc, Ppc, gamma_g, n_points=n_points)
        if abs(mp_mid - mp_target) < tol:
            return p_mid
        if mp_mid < mp_target:
            p_low = p_mid
        else:
            p_high = p_mid
    return 0.5 * (p_low + p_high)


# ---------------------------------------------------------------------------
# PVT table builder
# ---------------------------------------------------------------------------

def build_pvt_table(
    p_array: np.ndarray,
    T_R: float,
    Tpc: float,
    Ppc: float,
    gamma_g: float,
    p_base: float = 14.7,
    method: str = "hall_yarborough",
) -> pd.DataFrame:
    """
    Build a PVT table over the given pressure array.

    Returns
    -------
    pd.DataFrame with columns: p, z, mu_g, Bg, m_p
    """
    rows = []
    mp_prev = 0.0
    p_prev = p_base

    # Sort ascending for cumulative integration
    p_sorted = np.sort(p_array)

    for p in p_sorted:
        z = calc_z_factor(p, T_R, Tpc, Ppc, method)
        mu = calc_gas_viscosity_lee_gonzalez(p, T_R, z, gamma_g)
        Bg = calc_Bg(p, T_R, z)
        # Incremental pseudo-pressure addition for efficiency
        mp = calc_pseudopressure(p, T_R, Tpc, Ppc, gamma_g, p_base=p_base)
        rows.append({"p": p, "z": z, "mu_g": mu, "Bg": Bg, "m_p": mp})

    df = pd.DataFrame(rows).set_index("p").sort_index()
    df.index.name = "p"
    df = df.reset_index()
    return df
