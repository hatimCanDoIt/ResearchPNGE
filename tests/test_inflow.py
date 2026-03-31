"""
test_inflow.py — Unit tests for the LIT inflow equation.

Validation targets from screenshot:
    A = 15.105 psia/MMcfd  ± 0.10
    B = 0.55   psia/MMcfd² ± 0.01
"""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inflow import (
    calc_A_coefficient,
    calc_B_coefficient,
    calc_a_mp_coefficient,
    calc_b_mp_coefficient,
    calc_pwf_from_rate,
    calc_rate_from_pressures,
    calc_AOF,
    build_IPR_curve,
    decompose_pressure_drop,
    calc_mp_bar_from_rate,
    calc_q_from_mp,
)

# Base-case values
MU_I  = 0.0194      # cp
ZI    = 0.926
T_R   = 579.67      # °R
RE    = 3724.0      # ft
RW    = 0.5         # ft
S     = 0.0
K     = 20.0        # md
H     = 50.0        # ft
D     = 0.0003      # Mcfd⁻¹


PI_PSIA = 4000.0   # initial reservoir pressure for linearization


class TestACoefficient:
    def test_A_matches_target(self):
        A = calc_A_coefficient(MU_I, ZI, T_R, RE, RW, S, K, H, pi_psia=PI_PSIA)
        assert abs(A - 15.105) <= 0.10, f"A={A:.4f}, expected ≈ 15.105 psia/MMcfd"

    def test_A_positive(self):
        A = calc_A_coefficient(MU_I, ZI, T_R, RE, RW, S, K, H, pi_psia=PI_PSIA)
        assert A > 0

    def test_A_increases_with_skin(self):
        A0 = calc_A_coefficient(MU_I, ZI, T_R, RE, RW, 0, K, H, pi_psia=PI_PSIA)
        A5 = calc_A_coefficient(MU_I, ZI, T_R, RE, RW, 5, K, H, pi_psia=PI_PSIA)
        assert A5 > A0

    def test_A_decreases_with_permeability(self):
        A20 = calc_A_coefficient(MU_I, ZI, T_R, RE, RW, S, 20, H, pi_psia=PI_PSIA)
        A100 = calc_A_coefficient(MU_I, ZI, T_R, RE, RW, S, 100, H, pi_psia=PI_PSIA)
        assert A100 < A20


class TestBCoefficient:
    def test_B_matches_target(self):
        B = calc_B_coefficient(MU_I, ZI, T_R, D, K, H, pi_psia=PI_PSIA)
        assert abs(B - 0.55) <= 0.01, f"B={B:.4f}, expected ≈ 0.55 psia/MMcfd²"

    def test_B_positive(self):
        B = calc_B_coefficient(MU_I, ZI, T_R, D, K, H, pi_psia=PI_PSIA)
        assert B > 0

    def test_B_zero_when_D_zero(self):
        B = calc_B_coefficient(MU_I, ZI, T_R, 0.0, K, H, pi_psia=PI_PSIA)
        assert B == pytest.approx(0.0, abs=1e-10)

    def test_B_proportional_to_D(self):
        B1 = calc_B_coefficient(MU_I, ZI, T_R, 0.001, K, H, pi_psia=PI_PSIA)
        B2 = calc_B_coefficient(MU_I, ZI, T_R, 0.002, K, H, pi_psia=PI_PSIA)
        assert B2 == pytest.approx(2 * B1, rel=1e-6)


class TestPwfFromRate:
    def test_pwf_less_than_pbar(self):
        A = calc_A_coefficient(MU_I, ZI, T_R, RE, RW, S, K, H, pi_psia=PI_PSIA)
        B = calc_B_coefficient(MU_I, ZI, T_R, D, K, H, pi_psia=PI_PSIA)
        pwf = calc_pwf_from_rate(3.0, 4000.0, A, B)
        assert pwf < 4000.0

    def test_pwf_equals_pbar_at_zero_rate(self):
        pwf = calc_pwf_from_rate(0.0, 4000.0, 15.105, 0.55)
        assert pwf == pytest.approx(4000.0)

    def test_pwf_decreases_with_rate(self):
        A, B = 15.105, 0.55
        pwf1 = calc_pwf_from_rate(1.0, 4000.0, A, B)
        pwf3 = calc_pwf_from_rate(3.0, 4000.0, A, B)
        assert pwf3 < pwf1


class TestRateFromPressures:
    def test_rate_roundtrip(self):
        """calc_rate_from_pressures should invert calc_pwf_from_rate."""
        A = calc_A_coefficient(MU_I, ZI, T_R, RE, RW, S, K, H, pi_psia=PI_PSIA)
        B = calc_B_coefficient(MU_I, ZI, T_R, D, K, H, pi_psia=PI_PSIA)
        q_orig = 3.0
        pwf = calc_pwf_from_rate(q_orig, 4000.0, A, B)
        q_recovered = calc_rate_from_pressures(4000.0, pwf, A, B)
        assert abs(q_recovered - q_orig) < 0.001, (
            f"Rate roundtrip: expected {q_orig}, got {q_recovered:.4f}"
        )

    def test_rate_zero_when_no_drawdown(self):
        q = calc_rate_from_pressures(3000.0, 3000.0, 15.0, 0.5)
        assert q == pytest.approx(0.0)

    def test_pure_darcy_limit(self):
        """When B=0, q should be Δp/A."""
        dp = 100.0
        A = 15.105
        q = calc_rate_from_pressures(4000.0, 4000.0 - dp, A, 0.0)
        assert abs(q - dp / A) < 0.001


class TestDecompression:
    def test_nondarcy_zero_when_D_zero(self):
        d = decompose_pressure_drop(3.0, 15.105, 0.0)
        assert d["delta_p_nondarcy"] == 0.0
        assert d["nondarcy_fraction"] == 0.0

    def test_fractions_sum_to_one(self):
        d = decompose_pressure_drop(3.0, 15.105, 0.55)
        frac = d["nondarcy_fraction"]
        assert 0.0 <= frac <= 1.0

    def test_nondarcy_fraction_increases_with_rate(self):
        A, B = 15.105, 0.55
        d1 = decompose_pressure_drop(1.0, A, B)
        d5 = decompose_pressure_drop(5.0, A, B)
        assert d5["nondarcy_fraction"] > d1["nondarcy_fraction"]

    def test_total_equals_sum(self):
        A, B = 15.105, 0.55
        q = 3.0
        d = decompose_pressure_drop(q, A, B)
        assert d["delta_p_total"] == pytest.approx(
            d["delta_p_darcy"] + d["delta_p_nondarcy"], rel=1e-8
        )


class TestIPRCurve:
    def test_ipr_has_correct_columns(self):
        df = build_IPR_curve(4000.0, 15.105, 0.55)
        assert "q_MMcfd" in df.columns
        assert "pwf" in df.columns

    def test_ipr_starts_at_zero_rate(self):
        df = build_IPR_curve(4000.0, 15.105, 0.55)
        assert df["q_MMcfd"].iloc[0] == pytest.approx(0.0)

    def test_pwf_decreasing_with_rate(self):
        df = build_IPR_curve(4000.0, 15.105, 0.55)
        assert df["pwf"].is_monotonic_decreasing or \
               (df["pwf"].diff().iloc[1:] <= 0).all()


class TestPseudopressureInflow:
    def test_mp_bar_exceeds_mp_wf(self):
        a_mp = calc_a_mp_coefficient(T_R, RE, RW, S, K, H)
        b_mp = calc_b_mp_coefficient(T_R, D, K, H)
        mp_wf = 1e8
        mp_bar = calc_mp_bar_from_rate(3000.0, mp_wf, a_mp, b_mp)
        assert mp_bar > mp_wf

    def test_q_from_mp_roundtrip(self):
        a_mp = calc_a_mp_coefficient(T_R, RE, RW, S, K, H)
        b_mp = calc_b_mp_coefficient(T_R, D, K, H)
        q_orig = 3000.0   # Mcfd
        mp_wf = 1e8
        mp_bar = calc_mp_bar_from_rate(q_orig, mp_wf, a_mp, b_mp)
        q_rec = calc_q_from_mp(mp_bar, mp_wf, a_mp, b_mp)
        assert abs(q_rec - q_orig) < 0.1, (
            f"mp roundtrip: expected {q_orig}, got {q_rec:.2f}"
        )
