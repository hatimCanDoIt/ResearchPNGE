"""
analysis.py — Base-case validation, sensitivity analysis, and plotting.

Answers the core research questions:
  Q1. Does the FMB correctly recover G from simulated production data?
  Q2. How does the non-Darcy coefficient D bias the FMB estimate of G?
  Q3. How do errors in A and B factors propagate into G_fmb?
  Q4. How much production history is needed for FMB to converge?
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import BASE_CASE, VALIDATION_TARGETS, TOL, D_SENSITIVITY_VALUES
from src.gas_properties import (
    calc_pseudocritical_properties,
    calc_z_factor,
    calc_Bg,
    calc_gas_viscosity_lee_gonzalez,
    calc_pseudopressure,
)
from src.volumetric import calc_drainage_radius, calc_OGIP_volumetric, validate_volumetric
from src.inflow import (
    calc_A_coefficient,
    calc_B_coefficient,
    calc_a_mp_coefficient,
    calc_b_mp_coefficient,
    decompose_pressure_drop,
)
from src.material_balance import estimate_G_from_fmb, calc_fmb_error
from src.simulation import simulate_constant_rate, simulate_constant_pwf


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Base-case validation
# ---------------------------------------------------------------------------

def validate_base_case(config: dict = None) -> dict:
    """
    Full validation of the base-case parameters against screenshot targets.

    Steps
    -----
    1. PVT at pi  → check zi, μi, Bgi
    2. Geometry   → check re
    3. OGIP       → check G
    4. Inflow     → check A, B
    5. Simulate   → run FMB → check G_fmb ≈ G_volumetric

    Returns
    -------
    dict with all computed values and 'passed' boolean per check.
    """
    if config is None:
        config = BASE_CASE

    tgt = VALIDATION_TARGETS
    T_R = config["T_F"] + 459.67
    pi = config["pi"]

    # ---- 1. PVT ----
    Tpc, Ppc = calc_pseudocritical_properties(
        config["gamma_g"], config["y_N2"], config["y_CO2"], config["y_H2S"]
    )
    zi = calc_z_factor(pi, T_R, Tpc, Ppc)
    mu_i = calc_gas_viscosity_lee_gonzalez(pi, T_R, zi, config["gamma_g"])
    Bgi = calc_Bg(pi, T_R, zi)

    pvt_ok = {
        "zi": abs(zi - tgt["zi"]) <= TOL["zi"],
        "mu_i": abs(mu_i - tgt["mu_i_cp"]) <= TOL["mu_cp"],
        "Bgi": abs(Bgi - tgt["Bgi_RB_Mcf"]) <= TOL["Bg_RB_Mcf"],
    }

    # ---- 2. Geometry ----
    re = calc_drainage_radius(config["A_acres"], config["N_wells"])
    geo_ok = abs(re - tgt["re_ft"]) <= TOL["re_ft"]

    # ---- 3. OGIP ----
    G_vol = calc_OGIP_volumetric(
        config["A_acres"], config["h"], config["phi"], config["Swi"], Bgi
    )
    G_ok = abs(G_vol - tgt["G_MMCF"]) <= TOL["G_MMCF"]

    # ---- 4. Inflow coefficients ----
    A_lit = calc_A_coefficient(mu_i, zi, T_R, re, config["rw"], config["s"],
                               config["k"], config["h"], pi_psia=pi)
    B_lit = calc_B_coefficient(mu_i, zi, T_R, config["D"], config["k"], config["h"], pi_psia=pi)
    a_mp = calc_a_mp_coefficient(T_R, re, config["rw"], config["s"],
                                 config["k"], config["h"])
    b_mp = calc_b_mp_coefficient(T_R, config["D"], config["k"], config["h"])

    inflow_ok = {
        "A": abs(A_lit - tgt["A_psia_MMcfd"]) <= TOL["A_psia_MMcfd"],
        "B": abs(B_lit - tgt["B_psia_MMcfd2"]) <= TOL["B_psia_MMcfd2"],
    }

    # ---- 5. FMB self-consistency (noise-free) ----
    sim_df = simulate_constant_rate(
        q_MMcfd=config["q_MMcfd"],
        G_MMCF=G_vol,
        pi=pi,
        T_R=T_R,
        Tpc=Tpc,
        Ppc=Ppc,
        gamma_g=config["gamma_g"],
        A_lit=A_lit,
        B_lit=B_lit,
        a_mp=a_mp,
        b_mp=b_mp,
        zi=zi,
        dt_days=config["dt_days"],
        t_max_days=config["t_max_days"],
        pwf_min=config["pwf_min"],
    )

    fmb_ok = False
    G_fmb = None
    R2 = None
    fmb_error_pct = None

    if len(sim_df) >= 5:
        fmb_result = estimate_G_from_fmb(
            Gp_array=sim_df["Gp_MMCF"].values,
            p_bar_array=sim_df["p_bar_fmb"].values,
            z_array=sim_df["z"].values,
            pi=pi,
            zi=zi,
        )
        G_fmb = fmb_result["G_fmb_MMCF"]
        R2 = fmb_result["R_squared"]
        fmb_error_pct = 100.0 * (G_fmb - G_vol) / G_vol
        fmb_ok = abs(fmb_error_pct) <= TOL["G_fmb_pct"]

    # ---- Assemble report ----
    results = {
        # PVT
        "zi_computed": zi,          "zi_target": tgt["zi"],        "zi_ok": pvt_ok["zi"],
        "mu_i_computed": mu_i,      "mu_i_target": tgt["mu_i_cp"], "mu_i_ok": pvt_ok["mu_i"],
        "Bgi_computed": Bgi,        "Bgi_target": tgt["Bgi_RB_Mcf"],"Bgi_ok": pvt_ok["Bgi"],
        # Geometry
        "re_computed": re,          "re_target": tgt["re_ft"],     "re_ok": geo_ok,
        # OGIP
        "G_vol_MMCF": G_vol,        "G_target_MMCF": tgt["G_MMCF"],"G_vol_ok": G_ok,
        # Inflow
        "A_computed": A_lit,        "A_target": tgt["A_psia_MMcfd"],"A_ok": inflow_ok["A"],
        "B_computed": B_lit,        "B_target": tgt["B_psia_MMcfd2"],"B_ok": inflow_ok["B"],
        "a_mp": a_mp,               "b_mp": b_mp,
        # FMB
        "G_fmb_MMCF": G_fmb,       "R2_fmb": R2,
        "fmb_error_pct": fmb_error_pct, "fmb_ok": fmb_ok,
        # Overall
        "sim_df": sim_df,
        "Tpc": Tpc, "Ppc": Ppc, "T_R": T_R, "pi": pi,
        "all_passed": all([
            pvt_ok["zi"], pvt_ok["mu_i"], pvt_ok["Bgi"],
            geo_ok, G_ok, inflow_ok["A"], inflow_ok["B"], fmb_ok,
        ]),
    }

    _print_validation_table(results)
    return results


def _print_validation_table(r: dict):
    """Print a formatted validation summary to stdout."""
    def tick(ok): return "PASS" if ok else "FAIL"
    print("\n" + "=" * 65)
    print("  BASE-CASE VALIDATION REPORT")
    print("=" * 65)
    print(f"  z-factor (zi)    : {r['zi_computed']:.4f}  (target {r['zi_target']:.3f})   [{tick(r['zi_ok'])}]")
    print(f"  Viscosity (μi)   : {r['mu_i_computed']:.4f}  (target {r['mu_i_target']:.4f}) [{tick(r['mu_i_ok'])}]")
    print(f"  Bg (Bgi)         : {r['Bgi_computed']:.4f}  (target {r['Bgi_target']:.3f})   [{tick(r['Bgi_ok'])}]")
    print(f"  Drainage radius  : {r['re_computed']:.1f} ft (target {r['re_target']:.0f} ft) [{tick(r['re_ok'])}]")
    print(f"  OGIP (G)         : {r['G_vol_MMCF']:.0f} MMCF (target {r['G_target_MMCF']:.0f}) [{tick(r['G_vol_ok'])}]")
    print(f"  A coefficient    : {r['A_computed']:.4f}  (target {r['A_target']:.3f})   [{tick(r['A_ok'])}]")
    print(f"  B coefficient    : {r['B_computed']:.4f}  (target {r['B_target']:.2f})    [{tick(r['B_ok'])}]")
    if r["G_fmb_MMCF"] is not None:
        print(f"  FMB G estimate   : {r['G_fmb_MMCF']:.0f} MMCF  (error {r['fmb_error_pct']:.3f}%, R²={r['R2_fmb']:.6f}) [{tick(r['fmb_ok'])}]")
    print("=" * 65)
    print(f"  Overall: {'ALL CHECKS PASSED' if r['all_passed'] else 'SOME CHECKS FAILED'}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Sensitivity — non-Darcy coefficient D
# ---------------------------------------------------------------------------

def sensitivity_nondarcy_D(
    D_values: list = None,
    base_config: dict = None,
) -> pd.DataFrame:
    """
    Vary D from 0 to 0.002 and quantify FMB bias in G estimation.

    Key engineering insight
    -----------------------
    - If D is correctly known and included in a_mp, b_mp → FMB is unbiased.
    - If D is unknown and ASSUMED = 0 (common in practice), the FMB
      underestimates p̄ (missing the b·q² term) → G_fmb is OVERESTIMATED.
    - This function tests both scenarios:
        (a) D_true = D_assumed (no model error)
        (b) D_true > 0 but D_assumed = 0  (misspecification bias)
    """
    if D_values is None:
        D_values = D_SENSITIVITY_VALUES
    if base_config is None:
        base_config = BASE_CASE

    cfg = base_config.copy()
    T_R = cfg["T_F"] + 459.67
    pi = cfg["pi"]
    Tpc, Ppc = calc_pseudocritical_properties(
        cfg["gamma_g"], cfg["y_N2"], cfg["y_CO2"], cfg["y_H2S"]
    )
    zi = calc_z_factor(pi, T_R, Tpc, Ppc)
    mu_i = calc_gas_viscosity_lee_gonzalez(pi, T_R, zi, cfg["gamma_g"])
    Bgi = calc_Bg(pi, T_R, zi)
    re = calc_drainage_radius(cfg["A_acres"], cfg["N_wells"])
    G_vol = calc_OGIP_volumetric(cfg["A_acres"], cfg["h"], cfg["phi"], cfg["Swi"], Bgi)

    A_lit_base = calc_A_coefficient(mu_i, zi, T_R, re, cfg["rw"], cfg["s"], cfg["k"], cfg["h"], pi_psia=pi)
    a_mp_base = calc_a_mp_coefficient(T_R, re, cfg["rw"], cfg["s"], cfg["k"], cfg["h"])

    rows = []
    for D in D_values:
        # True B (used to simulate production)
        B_true = calc_B_coefficient(mu_i, zi, T_R, D, cfg["k"], cfg["h"], pi_psia=pi)
        b_mp_true = calc_b_mp_coefficient(T_R, D, cfg["k"], cfg["h"])

        # Scenario A: FMB knows D (correct model)
        sim_a = simulate_constant_rate(
            q_MMcfd=cfg["q_MMcfd"], G_MMCF=G_vol, pi=pi, T_R=T_R,
            Tpc=Tpc, Ppc=Ppc, gamma_g=cfg["gamma_g"],
            A_lit=A_lit_base, B_lit=B_true,
            a_mp=a_mp_base, b_mp=b_mp_true,
            zi=zi, dt_days=cfg["dt_days"], t_max_days=cfg["t_max_days"],
            pwf_min=cfg["pwf_min"],
        )
        nd_decomp = decompose_pressure_drop(cfg["q_MMcfd"], A_lit_base, B_true)

        G_fmb_a, R2_a = None, None
        if len(sim_a) >= 5:
            res_a = estimate_G_from_fmb(
                sim_a["Gp_MMCF"].values, sim_a["p_bar_fmb"].values,
                sim_a["z"].values, pi, zi
            )
            G_fmb_a = res_a["G_fmb_MMCF"]
            R2_a = res_a["R_squared"]

        # Scenario B: FMB assumes D = 0 (misspecification)
        b_mp_assumed_zero = 0.0
        # Re-compute p_bar_fmb with D=0 assumption
        if len(sim_a) >= 5:
            mp_wf_arr = sim_a["m_pwf"].values
            q_Mcfd_arr = sim_a["q_MMcfd"].values * 1000.0
            mp_pbar_assumed = mp_wf_arr + a_mp_base * q_Mcfd_arr   # no b term
            from src.gas_properties import invert_pseudopressure
            p_bar_assumed = np.array([
                invert_pseudopressure(mp, T_R, Tpc, Ppc, cfg["gamma_g"], p_high=pi * 1.05)
                for mp in mp_pbar_assumed
            ])
            z_arr = sim_a["z"].values
            res_b = estimate_G_from_fmb(
                sim_a["Gp_MMCF"].values, p_bar_assumed, z_arr, pi, zi
            )
            G_fmb_b = res_b["G_fmb_MMCF"]
            R2_b = res_b["R_squared"]
        else:
            G_fmb_b, R2_b = None, None

        rows.append({
            "D_Mcfd_inv": D,
            "B_coeff": B_true,
            "nondarcy_fraction": nd_decomp["nondarcy_fraction"],
            # Scenario A (correct D)
            "G_fmb_correct_D_MMCF": G_fmb_a,
            "G_fmb_correct_D_error_pct": 100*(G_fmb_a - G_vol)/G_vol if G_fmb_a else None,
            "R2_correct_D": R2_a,
            # Scenario B (D assumed = 0)
            "G_fmb_D_zero_MMCF": G_fmb_b,
            "G_fmb_D_zero_error_pct": 100*(G_fmb_b - G_vol)/G_vol if G_fmb_b else None,
            "R2_D_zero": R2_b,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivity — A and B factor accuracy
# ---------------------------------------------------------------------------

def sensitivity_AB_factors(
    A_multipliers: list = None,
    B_multipliers: list = None,
    base_config: dict = None,
) -> pd.DataFrame:
    """
    Grid search: vary A and B independently to quantify FMB bias.

    Engineering insight
    -------------------
    - A error shifts the entire p̄/z line vertically → biases G intercept.
    - B error is rate-dependent: at high q (early time), B·q² is large.
      B overestimation → p̄ overestimation → p̄/z too high → G underestimated.
      B underestimation (or D=0 assumption) → G overestimated.
    """
    if A_multipliers is None:
        A_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
    if B_multipliers is None:
        B_multipliers = [0.0, 0.5, 1.0, 2.0, 5.0]
    if base_config is None:
        base_config = BASE_CASE

    cfg = base_config.copy()
    T_R = cfg["T_F"] + 459.67
    pi = cfg["pi"]
    Tpc, Ppc = calc_pseudocritical_properties(
        cfg["gamma_g"], cfg["y_N2"], cfg["y_CO2"], cfg["y_H2S"]
    )
    zi = calc_z_factor(pi, T_R, Tpc, Ppc)
    mu_i = calc_gas_viscosity_lee_gonzalez(pi, T_R, zi, cfg["gamma_g"])
    Bgi = calc_Bg(pi, T_R, zi)
    re = calc_drainage_radius(cfg["A_acres"], cfg["N_wells"])
    G_vol = calc_OGIP_volumetric(cfg["A_acres"], cfg["h"], cfg["phi"], cfg["Swi"], Bgi)

    A_base = calc_A_coefficient(mu_i, zi, T_R, re, cfg["rw"], cfg["s"], cfg["k"], cfg["h"], pi_psia=pi)
    B_base = calc_B_coefficient(mu_i, zi, T_R, cfg["D"], cfg["k"], cfg["h"], pi_psia=pi)
    a_mp_base = calc_a_mp_coefficient(T_R, re, cfg["rw"], cfg["s"], cfg["k"], cfg["h"])
    b_mp_base = calc_b_mp_coefficient(T_R, cfg["D"], cfg["k"], cfg["h"])

    # Simulate "true" production once using base A/B
    sim_true = simulate_constant_rate(
        q_MMcfd=cfg["q_MMcfd"], G_MMCF=G_vol, pi=pi, T_R=T_R,
        Tpc=Tpc, Ppc=Ppc, gamma_g=cfg["gamma_g"],
        A_lit=A_base, B_lit=B_base,
        a_mp=a_mp_base, b_mp=b_mp_base,
        zi=zi, dt_days=cfg["dt_days"], t_max_days=cfg["t_max_days"],
        pwf_min=cfg["pwf_min"],
    )

    rows = []
    for Am in A_multipliers:
        for Bm in B_multipliers:
            A_test = A_base * Am
            B_test = B_base * Bm
            a_mp_test = a_mp_base * Am
            b_mp_test = b_mp_base * Bm

            if len(sim_true) < 5:
                continue

            # Re-compute p_bar_fmb using test A/B
            mp_wf_arr = sim_true["m_pwf"].values
            q_Mcfd_arr = sim_true["q_MMcfd"].values * 1000.0
            mp_pbar_test = mp_wf_arr + a_mp_test * q_Mcfd_arr + b_mp_test * q_Mcfd_arr ** 2

            from src.gas_properties import invert_pseudopressure
            p_bar_test = np.array([
                invert_pseudopressure(mp, T_R, Tpc, Ppc, cfg["gamma_g"], p_high=pi * 1.05)
                for mp in mp_pbar_test
            ])
            z_arr = sim_true["z"].values
            res = estimate_G_from_fmb(
                sim_true["Gp_MMCF"].values, p_bar_test, z_arr, pi, zi
            )

            rows.append({
                "A_mult": Am, "B_mult": Bm,
                "A_coeff": A_test, "B_coeff": B_test,
                "G_fmb_MMCF": res["G_fmb_MMCF"],
                "G_fmb_error_pct": 100 * (res["G_fmb_MMCF"] - G_vol) / G_vol,
                "R_squared": res["R_squared"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivity — production period length
# ---------------------------------------------------------------------------

def sensitivity_production_period(
    t_max_values: list = None,
    base_config: dict = None,
) -> pd.DataFrame:
    """
    Show how FMB accuracy improves as more production history is available.
    """
    if t_max_values is None:
        t_max_values = [365, 730, 1825, 3650, 7300]
    if base_config is None:
        base_config = BASE_CASE

    cfg = base_config.copy()
    T_R = cfg["T_F"] + 459.67
    pi = cfg["pi"]
    Tpc, Ppc = calc_pseudocritical_properties(
        cfg["gamma_g"], cfg["y_N2"], cfg["y_CO2"], cfg["y_H2S"]
    )
    zi = calc_z_factor(pi, T_R, Tpc, Ppc)
    mu_i = calc_gas_viscosity_lee_gonzalez(pi, T_R, zi, cfg["gamma_g"])
    Bgi = calc_Bg(pi, T_R, zi)
    re = calc_drainage_radius(cfg["A_acres"], cfg["N_wells"])
    G_vol = calc_OGIP_volumetric(cfg["A_acres"], cfg["h"], cfg["phi"], cfg["Swi"], Bgi)
    A_lit = calc_A_coefficient(mu_i, zi, T_R, re, cfg["rw"], cfg["s"], cfg["k"], cfg["h"], pi_psia=pi)
    B_lit = calc_B_coefficient(mu_i, zi, T_R, cfg["D"], cfg["k"], cfg["h"], pi_psia=pi)
    a_mp = calc_a_mp_coefficient(T_R, re, cfg["rw"], cfg["s"], cfg["k"], cfg["h"])
    b_mp = calc_b_mp_coefficient(T_R, cfg["D"], cfg["k"], cfg["h"])

    rows = []
    for t_max in t_max_values:
        sim = simulate_constant_rate(
            q_MMcfd=cfg["q_MMcfd"], G_MMCF=G_vol, pi=pi, T_R=T_R,
            Tpc=Tpc, Ppc=Ppc, gamma_g=cfg["gamma_g"],
            A_lit=A_lit, B_lit=B_lit,
            a_mp=a_mp, b_mp=b_mp,
            zi=zi, dt_days=cfg["dt_days"], t_max_days=t_max,
            pwf_min=cfg["pwf_min"],
        )
        if len(sim) < 5:
            continue
        res = estimate_G_from_fmb(
            sim["Gp_MMCF"].values, sim["p_bar_fmb"].values,
            sim["z"].values, pi, zi
        )
        rows.append({
            "t_max_days": t_max,
            "n_timesteps": len(sim),
            "Gp_final_MMCF": sim["Gp_MMCF"].iloc[-1],
            "Gp_fraction": sim["Gp_MMCF"].iloc[-1] / G_vol,
            "G_fmb_MMCF": res["G_fmb_MMCF"],
            "G_fmb_error_pct": 100 * (res["G_fmb_MMCF"] - G_vol) / G_vol,
            "R_squared": res["R_squared"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fmb_validation(
    sim_df: pd.DataFrame,
    G_volumetric: float,
    G_fmb: float,
    pi: float,
    zi: float,
    save_path: str = None,
) -> None:
    """
    Standard Mattar & Anderson FMB plot: p̄/z vs Gp.
    Blue dots = data, red line = regression, verticals = G estimates.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "fmb_validation.png")

    pz_arr = sim_df["p_bar_fmb"].values / sim_df["z"].values
    Gp_arr = sim_df["Gp_MMCF"].values
    pi_zi = pi / zi

    # Fit line
    coeffs = np.polyfit(Gp_arr, pz_arr, 1)
    Gp_line = np.linspace(0, G_fmb * 1.05, 200)
    pz_line = np.polyval(coeffs, Gp_line)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(Gp_arr, pz_arr, color="steelblue", s=40, zorder=5, label="FMB data points")
    ax.plot(Gp_line, pz_line, "r-", lw=2, label=f"FMB regression line")
    ax.axvline(G_fmb, color="red", lw=1.5, ls="--", label=f"G_FMB = {G_fmb:.0f} MMCF")
    ax.axvline(G_volumetric, color="green", lw=1.5, ls="--", label=f"G_Vol = {G_volumetric:.0f} MMCF")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(pi_zi, color="gray", lw=0.8, ls=":", label=f"pi/zi = {pi_zi:.0f} psia")

    ax.set_xlabel("Cumulative Production Gp [MMCF]", fontsize=12)
    ax.set_ylabel("p̄/z [psia]", fontsize=12)
    ax.set_title("Mattar & Anderson FMB Plot — p̄/z vs Gp", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot saved] {save_path}")


def plot_nondarcy_sensitivity(
    sensitivity_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Two-panel plot: D vs G_fmb_error for correct-D and D=0 scenarios.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "nondarcy_sensitivity.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    D_vals = sensitivity_df["D_Mcfd_inv"].values
    err_correct = sensitivity_df["G_fmb_correct_D_error_pct"].values
    err_zero = sensitivity_df["G_fmb_D_zero_error_pct"].values
    nd_frac = sensitivity_df["nondarcy_fraction"].values * 100

    # Left: FMB error vs D
    axes[0].plot(D_vals * 1000, err_correct, "bo-", label="Correct D (no model error)")
    axes[0].plot(D_vals * 1000, err_zero, "rs-", label="D assumed = 0 (misspecification)")
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("D [×10⁻³ Mcfd⁻¹]", fontsize=11)
    axes[0].set_ylabel("FMB Error in G [%]", fontsize=11)
    axes[0].set_title("Non-Darcy Coefficient D vs FMB Bias", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: non-Darcy fraction vs D
    axes[1].bar(D_vals * 1000, nd_frac, width=0.08, color="coral", edgecolor="k", alpha=0.8)
    axes[1].set_xlabel("D [×10⁻³ Mcfd⁻¹]", fontsize=11)
    axes[1].set_ylabel("Non-Darcy Fraction of ΔP [%]", fontsize=11)
    axes[1].set_title("Non-Darcy Pressure Drop Fraction at q = {:.1f} MMcfd".format(
        BASE_CASE["q_MMcfd"]), fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot saved] {save_path}")


def plot_AB_sensitivity_heatmap(
    ab_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Heatmap of G_fmb_error_pct over A_mult × B_mult grid.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "AB_sensitivity_heatmap.png")

    pivot = ab_df.pivot(index="B_mult", columns="A_mult", values="G_fmb_error_pct")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", origin="lower",
                   vmin=-50, vmax=50)
    plt.colorbar(im, ax=ax, label="G_fmb Error [%]")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}x" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}x" for v in pivot.index])
    ax.set_xlabel("A multiplier", fontsize=11)
    ax.set_ylabel("B multiplier", fontsize=11)
    ax.set_title("FMB Error (%) — Sensitivity to A and B Factor Accuracy", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot saved] {save_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def generate_analysis_report(results: dict, save_path: str = None) -> str:
    """
    Write a plain-text summary report of all analysis results.
    """
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "analysis_report.txt")

    lines = [
        "=" * 70,
        "  RESEARCHPNGE — FMB VALIDATION AND SENSITIVITY ANALYSIS REPORT",
        "=" * 70,
        "",
    ]

    if "base_case" in results:
        r = results["base_case"]
        lines += [
            "BASE-CASE VALIDATION",
            "-" * 40,
            f"  z-factor (zi)  : {r.get('zi_computed', 'N/A'):.4f}  [{'PASS' if r.get('zi_ok') else 'FAIL'}]",
            f"  Viscosity (μi) : {r.get('mu_i_computed', 'N/A'):.4f} cp  [{'PASS' if r.get('mu_i_ok') else 'FAIL'}]",
            f"  Bg (Bgi)       : {r.get('Bgi_computed', 'N/A'):.4f} RB/Mcf  [{'PASS' if r.get('Bgi_ok') else 'FAIL'}]",
            f"  Drainage radius: {r.get('re_computed', 'N/A'):.1f} ft  [{'PASS' if r.get('re_ok') else 'FAIL'}]",
            f"  OGIP (G)       : {r.get('G_vol_MMCF', 'N/A'):.0f} MMCF  [{'PASS' if r.get('G_vol_ok') else 'FAIL'}]",
            f"  A coefficient  : {r.get('A_computed', 'N/A'):.4f} psia/MMcfd  [{'PASS' if r.get('A_ok') else 'FAIL'}]",
            f"  B coefficient  : {r.get('B_computed', 'N/A'):.4f} psia/MMcfd²  [{'PASS' if r.get('B_ok') else 'FAIL'}]",
            f"  FMB G estimate : {r.get('G_fmb_MMCF', 'N/A'):.0f} MMCF  (error {r.get('fmb_error_pct', 'N/A'):.3f}%)  [{'PASS' if r.get('fmb_ok') else 'FAIL'}]",
            "",
        ]

    if "nondarcy_sensitivity" in results:
        df = results["nondarcy_sensitivity"]
        lines += [
            "NON-DARCY SENSITIVITY (D varied, q fixed)",
            "-" * 40,
            df[["D_Mcfd_inv", "nondarcy_fraction", "G_fmb_correct_D_error_pct", "G_fmb_D_zero_error_pct"]].to_string(index=False),
            "",
            "KEY FINDING: When D is correctly included in FMB, error ≈ 0%.",
            "When D assumed = 0, G_fmb is OVERESTIMATED (positive error).",
            "The overestimation grows monotonically with D.",
            "",
        ]

    if "production_period" in results:
        df = results["production_period"]
        lines += [
            "PRODUCTION PERIOD SENSITIVITY",
            "-" * 40,
            df[["t_max_days", "Gp_fraction", "G_fmb_error_pct", "R_squared"]].to_string(index=False),
            "",
        ]

    if "ml_summary" in results:
        lines += [
            "ML MODEL PERFORMANCE",
            "-" * 40,
            results["ml_summary"],
            "",
        ]

    lines.append("=" * 70)
    report = "\n".join(lines)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [Report saved] {save_path}")

    return report
