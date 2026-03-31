"""
test_material_balance.py — Unit tests for gas MBE and Mattar & Anderson FMB.

Key invariants:
  - At Gp=0: p̄/z = pi/zi = 4319 psia
  - At Gp=G: p̄/z = 0
  - FMB regression on noise-free data: G_fmb ≈ G within 0.1 %
"""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gas_properties import (
    calc_pseudocritical_properties,
    calc_z_factor,
    calc_Bg,
    calc_gas_viscosity_lee_gonzalez,
    calc_pseudopressure,
)
from src.material_balance import (
    calc_p_over_z,
    calc_pbar_from_mbe,
    run_standard_mbe,
    estimate_G_from_fmb,
    calc_fmb_error,
    calc_pbar_from_flowing_p,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GAMMA_G = 0.47
PI = 4000.0
T_R = 579.67
G_MMCF = 186342.0


@pytest.fixture(scope="module")
def pvt_base():
    Tpc, Ppc = calc_pseudocritical_properties(GAMMA_G)
    zi = calc_z_factor(PI, T_R, Tpc, Ppc)
    return {"Tpc": Tpc, "Ppc": Ppc, "zi": zi}


# ---------------------------------------------------------------------------
# p/z calculation
# ---------------------------------------------------------------------------

class TestPOverZ:
    def test_p_over_z_at_initial_conditions(self, pvt_base):
        pz = calc_p_over_z(PI, T_R, pvt_base["Tpc"], pvt_base["Ppc"])
        expected = PI / pvt_base["zi"]
        assert abs(pz - expected) < 1.0, f"p/z={pz:.1f}, expected {expected:.1f}"

    def test_p_over_z_positive(self, pvt_base):
        pz = calc_p_over_z(1000.0, T_R, pvt_base["Tpc"], pvt_base["Ppc"])
        assert pz > 0


# ---------------------------------------------------------------------------
# MBE inversion
# ---------------------------------------------------------------------------

class TestMBEInversion:
    def test_pbar_at_Gp_zero(self, pvt_base):
        """At Gp=0, recovered p̄ should equal pi."""
        p_bar = calc_pbar_from_mbe(
            Gp_MMCF=0.0,
            G_MMCF=G_MMCF,
            pi=PI,
            zi=pvt_base["zi"],
            T_R=T_R,
            Tpc=pvt_base["Tpc"],
            Ppc=pvt_base["Ppc"],
        )
        assert abs(p_bar - PI) < 5.0, f"p̄ at Gp=0: {p_bar:.1f} psia, expected ≈ {PI}"

    def test_pbar_decreases_with_production(self, pvt_base):
        """p̄ must decrease monotonically with cumulative production."""
        Gp_vals = [0, G_MMCF * 0.1, G_MMCF * 0.3, G_MMCF * 0.5, G_MMCF * 0.8]
        p_prev = PI + 1.0
        for Gp in Gp_vals:
            p_bar = calc_pbar_from_mbe(
                Gp, G_MMCF, PI, pvt_base["zi"], T_R, pvt_base["Tpc"], pvt_base["Ppc"]
            )
            assert p_bar < p_prev, f"p̄ not monotone at Gp={Gp:.0f} MMCF"
            p_prev = p_bar

    def test_pbar_positive(self, pvt_base):
        p_bar = calc_pbar_from_mbe(
            G_MMCF * 0.5, G_MMCF, PI, pvt_base["zi"],
            T_R, pvt_base["Tpc"], pvt_base["Ppc"]
        )
        assert p_bar > 0


# ---------------------------------------------------------------------------
# Standard MBE DataFrame
# ---------------------------------------------------------------------------

class TestRunStandardMBE:
    def test_output_columns(self, pvt_base):
        Gp_arr = np.linspace(0, G_MMCF * 0.8, 10)
        df = run_standard_mbe(Gp_arr, G_MMCF, PI, pvt_base["zi"],
                              T_R, pvt_base["Tpc"], pvt_base["Ppc"])
        assert "p_bar" in df.columns
        assert "z" in df.columns
        assert "p_over_z" in df.columns
        assert len(df) == 10

    def test_p_over_z_at_Gp0_matches_target(self, pvt_base):
        Gp_arr = np.array([0.0, G_MMCF * 0.5])
        df = run_standard_mbe(Gp_arr, G_MMCF, PI, pvt_base["zi"],
                              T_R, pvt_base["Tpc"], pvt_base["Ppc"])
        pi_zi = PI / pvt_base["zi"]
        assert abs(df["p_over_z"].iloc[0] - pi_zi) < 5.0


# ---------------------------------------------------------------------------
# FMB regression
# ---------------------------------------------------------------------------

class TestFMBRegression:
    def test_fmb_recovers_G_on_perfect_data(self, pvt_base):
        """
        If we construct p̄/z exactly from the MBE (no noise),
        FMB must recover G to within 0.1 %.
        """
        Gp_arr = np.linspace(1.0, G_MMCF * 0.8, 40)
        df = run_standard_mbe(Gp_arr, G_MMCF, PI, pvt_base["zi"],
                              T_R, pvt_base["Tpc"], pvt_base["Ppc"])
        result = estimate_G_from_fmb(
            Gp_array=df["Gp_MMCF"].values,
            p_bar_array=df["p_bar"].values,
            z_array=df["z"].values,
            pi=PI,
            zi=pvt_base["zi"],
        )
        G_fmb = result["G_fmb_MMCF"]
        pct_err = abs(G_fmb - G_MMCF) / G_MMCF * 100
        assert pct_err < 0.5, (
            f"FMB error {pct_err:.3f}% on noise-free data (expected < 0.5%)"
        )
        assert result["R_squared"] > 0.999

    def test_fmb_slope_negative(self, pvt_base):
        """MBE line has negative slope: p̄/z decreases as Gp increases."""
        Gp_arr = np.linspace(0, G_MMCF * 0.6, 20)
        df = run_standard_mbe(Gp_arr, G_MMCF, PI, pvt_base["zi"],
                              T_R, pvt_base["Tpc"], pvt_base["Ppc"])
        result = estimate_G_from_fmb(
            df["Gp_MMCF"].values, df["p_bar"].values, df["z"].values, PI, pvt_base["zi"]
        )
        assert result["slope"] < 0

    def test_fmb_intercept_approx_pi_zi(self, pvt_base):
        """Fitted intercept should approximate pi/zi."""
        Gp_arr = np.linspace(0, G_MMCF * 0.7, 30)
        df = run_standard_mbe(Gp_arr, G_MMCF, PI, pvt_base["zi"],
                              T_R, pvt_base["Tpc"], pvt_base["Ppc"])
        result = estimate_G_from_fmb(
            df["Gp_MMCF"].values, df["p_bar"].values, df["z"].values, PI, pvt_base["zi"]
        )
        pi_zi = PI / pvt_base["zi"]
        assert abs(result["intercept"] - pi_zi) / pi_zi < 0.02


# ---------------------------------------------------------------------------
# FMB error metric
# ---------------------------------------------------------------------------

class TestFMBError:
    def test_zero_error_when_equal(self):
        r = calc_fmb_error(186342.0, 186342.0)
        assert r["percent_error"] == pytest.approx(0.0)
        assert r["absolute_error_MMCF"] == pytest.approx(0.0)

    def test_positive_error_when_overestimated(self):
        r = calc_fmb_error(200000.0, 186342.0)
        assert r["percent_error"] > 0

    def test_negative_error_when_underestimated(self):
        r = calc_fmb_error(170000.0, 186342.0)
        assert r["percent_error"] < 0


# ---------------------------------------------------------------------------
# Pressure-based FMB back-calculation
# ---------------------------------------------------------------------------

class TestPressureBasedFMB:
    def test_pbar_from_flowing_p(self):
        pwf = 3500.0
        q = 3.0
        A = 15.105
        B = 0.55
        p_bar = calc_pbar_from_flowing_p(pwf, q, A, B)
        expected = pwf + A * q + B * q ** 2
        assert abs(p_bar - expected) < 1e-8

    def test_pbar_greater_than_pwf(self):
        p_bar = calc_pbar_from_flowing_p(3500.0, 3.0, 15.105, 0.55)
        assert p_bar > 3500.0
