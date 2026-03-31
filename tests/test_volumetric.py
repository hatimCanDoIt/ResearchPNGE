"""
test_volumetric.py — Unit tests for volumetric OGIP and geometry.

Validation targets from screenshot:
    re = 3724 ft   (A=5000 acres, N=5 wells)
    G  = 186342 MMCF
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.volumetric import calc_drainage_radius, calc_OGIP_volumetric, validate_volumetric


class TestDrainageRadius:
    def test_base_case_re(self):
        re = calc_drainage_radius(A_acres=5000.0, N_wells=5)
        assert abs(re - 3724.0) <= 5.0, f"re={re:.1f} ft, expected ≈ 3724 ft"

    def test_re_positive(self):
        assert calc_drainage_radius(1000.0, 3) > 0

    def test_re_decreases_with_more_wells(self):
        re_1 = calc_drainage_radius(5000.0, 1)
        re_5 = calc_drainage_radius(5000.0, 5)
        assert re_5 < re_1

    def test_re_scales_with_area(self):
        re_small = calc_drainage_radius(1000.0, 1)
        re_large = calc_drainage_radius(4000.0, 1)
        assert re_large == pytest.approx(re_small * 2.0, rel=1e-3)

    def test_single_well(self):
        A = 1000.0
        re = calc_drainage_radius(A, 1)
        expected = math.sqrt(A * 43560.0 / math.pi)
        assert abs(re - expected) < 0.1


class TestOGIP:
    def test_base_case_G(self):
        G = calc_OGIP_volumetric(
            A_acres=5000.0, h=50.0, phi=0.10, Swi=0.35, Bgi=0.677
        )
        assert abs(G - 186342.0) <= 500.0, f"G={G:.0f} MMCF, expected ≈ 186342 MMCF"

    def test_G_positive(self):
        assert calc_OGIP_volumetric(1000, 20, 0.1, 0.3, 0.5) > 0

    def test_G_increases_with_area(self):
        G1 = calc_OGIP_volumetric(1000, 50, 0.1, 0.35, 0.677)
        G2 = calc_OGIP_volumetric(2000, 50, 0.1, 0.35, 0.677)
        assert G2 == pytest.approx(2 * G1, rel=1e-6)

    def test_G_decreases_with_higher_Swi(self):
        G_low = calc_OGIP_volumetric(5000, 50, 0.1, 0.2, 0.677)
        G_high = calc_OGIP_volumetric(5000, 50, 0.1, 0.5, 0.677)
        assert G_high < G_low

    def test_G_decreases_with_higher_Bgi(self):
        G_small = calc_OGIP_volumetric(5000, 50, 0.1, 0.35, 0.5)
        G_large = calc_OGIP_volumetric(5000, 50, 0.1, 0.35, 1.0)
        assert G_large < G_small

    def test_G_zero_porosity(self):
        G = calc_OGIP_volumetric(5000, 50, 0.0, 0.35, 0.677)
        assert G == pytest.approx(0.0, abs=0.01)

    def test_G_full_water_saturation(self):
        G = calc_OGIP_volumetric(5000, 50, 0.1, 1.0, 0.677)
        assert G == pytest.approx(0.0, abs=0.01)


class TestValidation:
    def test_passes_for_correct_value(self):
        result = validate_volumetric(186342.0, G_target=186342.0)
        assert result["passed"] is True
        assert result["absolute_error_MMCF"] == 0.0

    def test_fails_for_large_error(self):
        import warnings
        with warnings.catch_warnings(record=True):
            result = validate_volumetric(100000.0, G_target=186342.0)
        assert result["passed"] is False

    def test_percent_error_sign(self):
        result = validate_volumetric(190000.0, G_target=186342.0)
        assert result["percent_error"] > 0   # overestimate

        result2 = validate_volumetric(180000.0, G_target=186342.0)
        assert result2["percent_error"] < 0  # underestimate
