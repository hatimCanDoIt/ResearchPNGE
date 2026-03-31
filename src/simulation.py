"""
simulation.py — Production simulation connecting inflow equation to MBE.

Two production scenarios:
  1. Constant rate  — produce at fixed q until pwf < pwf_min
  2. Constant pwf   — produce at declining rate to maintain fixed pwf

Monte Carlo mode generates a large synthetic dataset of
(reservoir params → production history → FMB error) for ML training.
"""

import math
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.gas_properties import (
    calc_pseudocritical_properties,
    calc_z_factor,
    calc_Bg,
    calc_gas_viscosity_lee_gonzalez,
    calc_pseudopressure,
    invert_pseudopressure,
)
from src.volumetric import calc_drainage_radius, calc_OGIP_volumetric
from src.inflow import (
    calc_A_coefficient,
    calc_B_coefficient,
    calc_a_mp_coefficient,
    calc_b_mp_coefficient,
    calc_pwf_from_rate,
    calc_rate_from_pressures,
    decompose_pressure_drop,
    calc_mp_bar_from_rate,
)
from src.material_balance import (
    calc_pbar_from_mbe,
    calc_pbar_from_flowing_mp,
    estimate_G_from_fmb,
    calc_fmb_error,
    calc_p_over_z,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pvt_at_pi(pi, T_R, gamma_g, y_N2=0.0, y_CO2=0.0, y_H2S=0.0):
    """Return (Tpc, Ppc, zi, mu_i, Bgi) at initial conditions."""
    Tpc, Ppc = calc_pseudocritical_properties(gamma_g, y_N2, y_CO2, y_H2S)
    zi = calc_z_factor(pi, T_R, Tpc, Ppc)
    mu_i = calc_gas_viscosity_lee_gonzalez(pi, T_R, zi, gamma_g)
    Bgi = calc_Bg(pi, T_R, zi)
    return Tpc, Ppc, zi, mu_i, Bgi


# ---------------------------------------------------------------------------
# Constant-rate simulation
# ---------------------------------------------------------------------------

def simulate_constant_rate(
    q_MMcfd: float,
    G_MMCF: float,
    pi: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    gamma_g: float,
    A_lit: float,
    B_lit: float,
    a_mp: float,
    b_mp: float,
    zi: float,
    dt_days: float = 30.0,
    t_max_days: float = 3650.0,
    pwf_min: float = 500.0,
) -> pd.DataFrame:
    """
    Simulate constant-rate production using MBE + LIT inflow.

    At each timestep:
        1. Advance Gp += q · dt
        2. Compute p̄ from MBE  (iterative z-factor inversion)
        3. Compute pwf = p̄ − A·q − B·q²  (pressure-based, quick check)
        4. Compute m(pwf) and back-calculate m(p̄)_FMB via inflow
        5. Store all variables

    Production stops when pwf < pwf_min or Gp > 0.95·G.

    Returns
    -------
    pd.DataFrame with columns:
        time_days, Gp_MMCF, q_MMcfd, p_bar, z, pwf,
        m_pwf, m_pbar_true, m_pbar_fmb, p_bar_fmb, p_over_z, p_bar_over_z_fmb
    """
    q_Mcfd = q_MMcfd * 1000.0     # convert for pseudopressure form
    rows = []
    Gp = 0.0
    t = 0.0
    mp_pi = calc_pseudopressure(pi, T_R, Tpc, Ppc, gamma_g)

    while t <= t_max_days:
        # MBE: recover p_bar from Gp
        p_bar = calc_pbar_from_mbe(Gp, G_MMCF, pi, zi, T_R, Tpc, Ppc)
        z_bar = calc_z_factor(p_bar, T_R, Tpc, Ppc)
        p_over_z = p_bar / z_bar

        # Pressure-based inflow (matches screenshot form)
        pwf = calc_pwf_from_rate(q_MMcfd, p_bar, A_lit, B_lit)

        if pwf < pwf_min:
            break

        # Pseudopressure at pwf (rigorous)
        mp_wf = calc_pseudopressure(pwf, T_R, Tpc, Ppc, gamma_g)
        # True m(p̄)
        mp_pbar_true = calc_pseudopressure(p_bar, T_R, Tpc, Ppc, gamma_g)
        # FMB back-calculation of m(p̄) — what Mattar & Anderson does
        mp_pbar_fmb = calc_mp_bar_from_rate(q_Mcfd, mp_wf, a_mp, b_mp)
        # Invert to get p_bar_fmb
        p_bar_fmb = invert_pseudopressure(mp_pbar_fmb, T_R, Tpc, Ppc, gamma_g, p_high=pi * 1.05)
        z_fmb = calc_z_factor(p_bar_fmb, T_R, Tpc, Ppc)
        p_bar_over_z_fmb = p_bar_fmb / z_fmb

        rows.append({
            "time_days": t,
            "Gp_MMCF": Gp,
            "q_MMcfd": q_MMcfd,
            "p_bar": p_bar,
            "z": z_bar,
            "pwf": pwf,
            "m_pwf": mp_wf,
            "m_pbar_true": mp_pbar_true,
            "m_pbar_fmb": mp_pbar_fmb,
            "p_bar_fmb": p_bar_fmb,
            "p_over_z": p_over_z,
            "p_bar_over_z_fmb": p_bar_over_z_fmb,
        })

        # Advance time and Gp
        t += dt_days
        Gp += q_MMcfd * dt_days / 365.25   # MMcfd × days / days_per_year → MMCF/yr·yr

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Constant-pwf simulation
# ---------------------------------------------------------------------------

def simulate_constant_pwf(
    pwf_target: float,
    G_MMCF: float,
    pi: float,
    T_R: float,
    Tpc: float,
    Ppc: float,
    gamma_g: float,
    A_lit: float,
    B_lit: float,
    a_mp: float,
    b_mp: float,
    zi: float,
    dt_days: float = 30.0,
    t_max_days: float = 3650.0,
    q_min_MMcfd: float = 0.01,
) -> pd.DataFrame:
    """
    Simulate declining-rate production at constant flowing BHP.

    At each timestep:
        1. Compute q = calc_rate_from_pressures(p_bar, pwf_target, A, B)
        2. Advance Gp += q · dt
        3. Update p_bar via MBE

    Returns
    -------
    pd.DataFrame with same columns as simulate_constant_rate (plus q_MMcfd)
    """
    rows = []
    Gp = 0.0
    t = 0.0

    while t <= t_max_days:
        # MBE: recover p_bar
        p_bar = calc_pbar_from_mbe(Gp, G_MMCF, pi, zi, T_R, Tpc, Ppc)
        z_bar = calc_z_factor(p_bar, T_R, Tpc, Ppc)
        p_over_z = p_bar / z_bar

        # Inflow: compute declining rate
        q_MMcfd = calc_rate_from_pressures(p_bar, pwf_target, A_lit, B_lit)
        if q_MMcfd < q_min_MMcfd:
            break

        q_Mcfd = q_MMcfd * 1000.0
        mp_wf = calc_pseudopressure(pwf_target, T_R, Tpc, Ppc, gamma_g)
        mp_pbar_true = calc_pseudopressure(p_bar, T_R, Tpc, Ppc, gamma_g)
        mp_pbar_fmb = calc_mp_bar_from_rate(q_Mcfd, mp_wf, a_mp, b_mp)
        p_bar_fmb = invert_pseudopressure(mp_pbar_fmb, T_R, Tpc, Ppc, gamma_g, p_high=pi * 1.05)
        z_fmb = calc_z_factor(p_bar_fmb, T_R, Tpc, Ppc)
        p_bar_over_z_fmb = p_bar_fmb / z_fmb

        rows.append({
            "time_days": t,
            "Gp_MMCF": Gp,
            "q_MMcfd": q_MMcfd,
            "p_bar": p_bar,
            "z": z_bar,
            "pwf": pwf_target,
            "m_pwf": mp_wf,
            "m_pbar_true": mp_pbar_true,
            "m_pbar_fmb": mp_pbar_fmb,
            "p_bar_fmb": p_bar_fmb,
            "p_over_z": p_over_z,
            "p_bar_over_z_fmb": p_bar_over_z_fmb,
        })

        t += dt_days
        Gp += q_MMcfd * dt_days / 365.25

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monte Carlo — generate synthetic ML training data
# ---------------------------------------------------------------------------

def run_monte_carlo(
    n_samples: int = 5000,
    seed: int = 42,
    mc_settings: dict = None,
    dt_days: float = 30.0,
    t_max_days: float = 3650.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic (reservoir params → FMB error) dataset for ML training.

    For each sample:
        1. Draw parameters from uniform distributions.
        2. Compute PVT, Bgi, re, G_volumetric, A, B, a_mp, b_mp.
        3. Simulate constant-pwf production for t_max_days.
        4. Run FMB regression on simulated history.
        5. Compute G_correction = G_volumetric / G_fmb.

    Returns
    -------
    pd.DataFrame with all input params, derived properties, and FMB outputs.
    """
    if mc_settings is None:
        from config import MC_SETTINGS
        mc_settings = MC_SETTINGS

    rng = np.random.default_rng(seed)

    def U(lo, hi, n=1):
        return rng.uniform(lo, hi, n) if n > 1 else float(rng.uniform(lo, hi))

    def RI(lo, hi):
        return int(rng.integers(lo, hi + 1))

    records = []
    iterator = range(n_samples)
    if verbose:
        iterator = tqdm(iterator, desc="Monte Carlo")

    for _ in iterator:
        try:
            # Sample parameters
            pi = U(*mc_settings["pi_range"])
            T_F = U(*mc_settings["T_F_range"])
            gamma_g = U(*mc_settings["gamma_g_range"])
            A_acres = U(*mc_settings["A_acres_range"])
            h = U(*mc_settings["h_range"])
            phi = U(*mc_settings["phi_range"])
            Swi = U(*mc_settings["Swi_range"])
            k = U(*mc_settings["k_range"])
            D = U(*mc_settings["D_range"])
            s = U(*mc_settings["s_range"])
            N_wells = RI(*mc_settings["N_wells_range"])
            q_MMcfd = U(*mc_settings["q_MMcfd_range"])
            pwf_min = U(*mc_settings["pwf_min_range"])

            T_R = T_F + 459.67

            # PVT at initial conditions
            Tpc, Ppc, zi, mu_i, Bgi = _pvt_at_pi(pi, T_R, gamma_g)

            # Reject unphysical samples
            if Bgi <= 0 or zi <= 0 or mu_i <= 0:
                continue
            if Swi >= 0.9 or phi <= 0:
                continue

            # Geometry
            re = calc_drainage_radius(A_acres, N_wells)
            if re <= 1.0:
                continue

            # OGIP
            G_MMCF = calc_OGIP_volumetric(A_acres, h, phi, Swi, Bgi)
            if G_MMCF <= 0:
                continue

            # Inflow coefficients (pressure-based, linearized at initial conditions)
            A_lit = calc_A_coefficient(mu_i, zi, T_R, re, 0.5, s, k, h, pi_psia=pi)
            B_lit = calc_B_coefficient(mu_i, zi, T_R, D, k, h, pi_psia=pi)

            # Pseudopressure coefficients (Mattar & Anderson)
            a_mp = calc_a_mp_coefficient(T_R, re, 0.5, s, k, h)
            b_mp = calc_b_mp_coefficient(T_R, D, k, h)

            # Simulate production
            sim_df = simulate_constant_pwf(
                pwf_target=pwf_min,
                G_MMCF=G_MMCF,
                pi=pi,
                T_R=T_R,
                Tpc=Tpc,
                Ppc=Ppc,
                gamma_g=gamma_g,
                A_lit=A_lit,
                B_lit=B_lit,
                a_mp=a_mp,
                b_mp=b_mp,
                zi=zi,
                dt_days=dt_days,
                t_max_days=t_max_days,
            )

            if len(sim_df) < 5:
                continue

            # FMB regression using pseudopressure back-calculated p_bar
            fmb_result = estimate_G_from_fmb(
                Gp_array=sim_df["Gp_MMCF"].values,
                p_bar_array=sim_df["p_bar_fmb"].values,
                z_array=sim_df["z"].values,
                pi=pi,
                zi=zi,
            )
            G_fmb = fmb_result["G_fmb_MMCF"]

            if G_fmb <= 0 or not np.isfinite(G_fmb):
                continue

            G_correction = G_MMCF / G_fmb
            G_fmb_error_pct = 100.0 * (G_fmb - G_MMCF) / G_MMCF

            # Derived features
            nd_decomp = decompose_pressure_drop(q_MMcfd, A_lit, B_lit)
            nd_frac = nd_decomp["nondarcy_fraction"]
            Gp_final = sim_df["Gp_MMCF"].iloc[-1]
            avg_rate = sim_df["q_MMcfd"].mean()
            p_final = sim_df["p_bar"].iloc[-1]

            records.append({
                # Raw parameters
                "pi": pi, "T_F": T_F, "gamma_g": gamma_g,
                "A_acres": A_acres, "h": h, "phi": phi, "Swi": Swi,
                "k": k, "D": D, "s": s, "N_wells": N_wells,
                "q_MMcfd_input": q_MMcfd, "pwf_min": pwf_min,
                # PVT at pi
                "zi": zi, "mu_i": mu_i, "Bgi": Bgi, "Tpc": Tpc, "Ppc": Ppc,
                "Tpr_i": T_R / Tpc, "Ppr_i": pi / Ppc,
                # Flow features
                "re": re, "A_coeff": A_lit, "B_coeff": B_lit,
                "a_mp": a_mp, "b_mp": b_mp,
                "ln_re_rw": math.log(re / 0.5),
                "B_over_A": B_lit / A_lit if A_lit > 0 else 0.0,
                "D_q_dimensionless": D * q_MMcfd * 1000,
                "nondarcy_fraction_initial": nd_frac,
                # Production history features
                "Gp_final_MMCF": Gp_final,
                "Gp_final_fraction": Gp_final / G_MMCF,
                "t_max_days": t_max_days,
                "avg_rate_MMcfd": avg_rate,
                "pressure_depletion_fraction": (pi - p_final) / pi,
                "n_timesteps": len(sim_df),
                # Targets
                "G_volumetric_MMCF": G_MMCF,
                "G_fmb_MMCF": G_fmb,
                "G_fmb_error_pct": G_fmb_error_pct,
                "G_correction": G_correction,
                "R_squared_fmb": fmb_result["R_squared"],
            })

        except Exception:
            continue

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Noise injection (for robustness testing)
# ---------------------------------------------------------------------------

def add_measurement_noise(
    df: pd.DataFrame,
    pressure_noise_pct: float = 0.5,
    rate_noise_pct: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Add Gaussian measurement noise to pressure and rate columns.

    Parameters
    ----------
    pressure_noise_pct : float  Std dev as % of pressure value
    rate_noise_pct     : float  Std dev as % of rate value

    Returns
    -------
    Noisy copy of df.
    """
    rng = np.random.default_rng(seed)
    df_noisy = df.copy()

    if "pwf" in df_noisy.columns:
        sigma_p = pressure_noise_pct / 100.0 * df_noisy["pwf"]
        df_noisy["pwf"] += rng.normal(0, sigma_p)

    if "p_bar" in df_noisy.columns:
        sigma_p = pressure_noise_pct / 100.0 * df_noisy["p_bar"]
        df_noisy["p_bar"] += rng.normal(0, sigma_p)

    if "q_MMcfd" in df_noisy.columns:
        sigma_q = rate_noise_pct / 100.0 * df_noisy["q_MMcfd"]
        df_noisy["q_MMcfd"] += rng.normal(0, sigma_q)
        df_noisy["q_MMcfd"] = df_noisy["q_MMcfd"].clip(lower=0.0)

    return df_noisy
