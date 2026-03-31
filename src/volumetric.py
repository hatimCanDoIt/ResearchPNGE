"""
volumetric.py — Volumetric OGIP and reservoir geometry calculations.

Equations
---------
    G [MMCF] = (43560 · A [acres] · h [ft] · φ · (1 − Swi)) / (Bgi [RB/Mcf] · 5.615 · 1000)

    Derivation:
        Pore volume [ft³] / (Bgi [RB/Mcf] × 5.615 [ft³/RB] × 1000 [Mcf/MMCF])
        = gas in place in MMCF at standard conditions.
    Note: 43560 ft²/acre; 5.615 ft³/RB; divide by 1000 to convert Mcf → MMCF.

Validation target (from screenshot): G = 186,342 MMCF, re = 3,724 ft.
"""

import math
import warnings
from config import VALIDATION_TARGETS, TOL


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def calc_drainage_radius(A_acres: float, N_wells: int) -> float:
    """
    Per-well drainage radius from total field area.

        re = sqrt( A_total_ft² / (N · π) )

    Parameters
    ----------
    A_acres : float  Total drainage area [acres]
    N_wells : int    Number of wells

    Returns
    -------
    re : float  [ft]
    """
    A_per_well_ft2 = A_acres * 43560.0 / N_wells
    return math.sqrt(A_per_well_ft2 / math.pi)


def calc_pore_volume(A_acres: float, h: float, phi: float) -> float:
    """
    Total pore volume of the reservoir.

        Vp [ft³] = 43560 · A [acres] · h [ft] · φ

    Returns
    -------
    Vp : float  [ft³]
    """
    return 43560.0 * A_acres * h * phi


# ---------------------------------------------------------------------------
# OGIP
# ---------------------------------------------------------------------------

def calc_OGIP_volumetric(
    A_acres: float,
    h: float,
    phi: float,
    Swi: float,
    Bgi: float,
) -> float:
    """
    Standard volumetric original gas in place (OGIP).

        G [MMCF] = (43560 · A · h · φ · (1 − Swi)) / (Bgi · 1000)

    Parameters
    ----------
    A_acres : float  Drainage area  [acres]
    h       : float  Net pay        [ft]
    phi     : float  Porosity       [fraction]
    Swi     : float  Irreducible water saturation [fraction]
    Bgi     : float  Initial gas FVF [RB/Mcf]

    Returns
    -------
    G : float  [MMCF]
    """
    # Pore volume [ft³] → gas in place [Mcf] by dividing by Bgi [RB/Mcf] × 5.615 [ft³/RB]
    G_Mcf = (43560.0 * A_acres * h * phi * (1.0 - Swi)) / (Bgi * 5.615)
    return G_Mcf / 1000.0   # convert Mcf → MMCF


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_volumetric(G_computed: float, G_target: float = None) -> dict:
    """
    Compare computed OGIP to the known target.

    Parameters
    ----------
    G_computed : float  [MMCF]
    G_target   : float  [MMCF]  defaults to VALIDATION_TARGETS['G_MMCF']

    Returns
    -------
    dict with keys: G_computed, G_target, absolute_error, percent_error, passed
    """
    if G_target is None:
        G_target = VALIDATION_TARGETS["G_MMCF"]

    abs_err = abs(G_computed - G_target)
    pct_err = 100.0 * (G_computed - G_target) / G_target
    passed = abs_err <= TOL["G_MMCF"]

    if not passed:
        warnings.warn(
            f"OGIP validation FAILED: computed={G_computed:.1f} MMCF, "
            f"target={G_target:.1f} MMCF, error={pct_err:.3f}%"
        )

    return {
        "G_computed_MMCF": G_computed,
        "G_target_MMCF": G_target,
        "absolute_error_MMCF": abs_err,
        "percent_error": pct_err,
        "passed": passed,
    }
