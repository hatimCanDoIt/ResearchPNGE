"""
main.py — Orchestrator for the ResearchPNGE FMB validation pipeline.

Workflow
--------
1. Base-case validation  → verify all computed values match screenshot targets
2. Sensitivity analysis  → non-Darcy D, A/B factors, production period
3. Monte Carlo           → generate synthetic dataset (5000 samples)
4. ML pipeline           → train RF, XGBoost, MLP, PINN; compare; save best
5. Analysis report       → write data/results/analysis_report.txt

Usage
-----
    python main.py                   # full pipeline
    python main.py --validate-only   # step 1 only (fast, ~30 s)
    python main.py --no-ml           # steps 1-2 only
    python main.py --mc-samples 500  # smaller Monte Carlo run
"""

import sys
import os
import argparse
import pandas as pd

# Ensure src/ is importable from root
sys.path.insert(0, os.path.dirname(__file__))

from config import BASE_CASE, MC_SETTINGS
from src.analysis import (
    validate_base_case,
    sensitivity_nondarcy_D,
    sensitivity_AB_factors,
    sensitivity_production_period,
    plot_fmb_validation,
    plot_nondarcy_sensitivity,
    plot_AB_sensitivity_heatmap,
    generate_analysis_report,
    RESULTS_DIR,
)
from src.simulation import run_monte_carlo
from src.ml_model import run_ml_pipeline


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_validation_pipeline() -> dict:
    """Step 1: validate all base-case computed values vs screenshot targets."""
    print("\n" + "=" * 65)
    print("  STEP 1 — BASE-CASE VALIDATION")
    print("=" * 65)
    results = validate_base_case(BASE_CASE)
    return results


def run_sensitivity_analysis(base_results: dict) -> dict:
    """Step 2: sensitivity studies."""
    print("\n" + "=" * 65)
    print("  STEP 2 — SENSITIVITY ANALYSIS")
    print("=" * 65)

    print("\n  2a. Non-Darcy coefficient D …")
    nd_df = sensitivity_nondarcy_D(base_config=BASE_CASE)
    nd_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_nondarcy_D.csv"), index=False)
    print(nd_df[["D_Mcfd_inv", "nondarcy_fraction",
                 "G_fmb_correct_D_error_pct",
                 "G_fmb_D_zero_error_pct"]].to_string(index=False))
    plot_nondarcy_sensitivity(nd_df)

    print("\n  2b. A and B factor accuracy …")
    ab_df = sensitivity_AB_factors(base_config=BASE_CASE)
    ab_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_AB_factors.csv"), index=False)
    plot_AB_sensitivity_heatmap(ab_df)
    print(ab_df[["A_mult", "B_mult", "G_fmb_error_pct"]].to_string(index=False))

    print("\n  2c. Production period length …")
    pp_df = sensitivity_production_period(base_config=BASE_CASE)
    pp_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_production_period.csv"), index=False)
    print(pp_df[["t_max_days", "Gp_fraction", "G_fmb_error_pct", "R_squared"]].to_string(index=False))

    # FMB validation plot from base-case simulation
    sim_df = base_results.get("sim_df")
    if sim_df is not None and len(sim_df) > 0:
        plot_fmb_validation(
            sim_df=sim_df,
            G_volumetric=base_results["G_vol_MMCF"],
            G_fmb=base_results["G_fmb_MMCF"],
            pi=base_results["pi"],
            zi=base_results["zi_computed"],
        )

    return {
        "nondarcy_sensitivity": nd_df,
        "AB_sensitivity": ab_df,
        "production_period": pp_df,
    }


def run_mc_and_ml_pipeline(n_samples: int = 5000) -> dict:
    """Steps 3 + 4: Monte Carlo data generation and ML training."""
    print("\n" + "=" * 65)
    print(f"  STEP 3 — MONTE CARLO  (n={n_samples} samples)")
    print("=" * 65)

    mc_settings = MC_SETTINGS.copy()
    mc_settings["n_samples"] = n_samples

    mc_df = run_monte_carlo(
        n_samples=n_samples,
        seed=mc_settings["seed"],
        mc_settings=mc_settings,
        verbose=True,
    )
    mc_path = os.path.join(RESULTS_DIR, "monte_carlo_dataset.csv")
    mc_df.to_csv(mc_path, index=False)
    print(f"  Monte Carlo dataset: {len(mc_df)} valid samples saved to {mc_path}")

    print("\n" + "=" * 65)
    print("  STEP 4 — ML PIPELINE")
    print("=" * 65)
    ml_results = run_ml_pipeline(mc_df, target="G_correction", seed=42)

    return {"mc_df": mc_df, "ml_results": ml_results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ResearchPNGE FMB pipeline")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run only base-case validation (fast)")
    parser.add_argument("--no-ml", action="store_true",
                        help="Skip Monte Carlo and ML (steps 1+2 only)")
    parser.add_argument("--mc-samples", type=int, default=5000,
                        help="Number of Monte Carlo samples (default: 5000)")
    args = parser.parse_args()

    all_results = {}

    # Step 1
    base_results = run_validation_pipeline()
    all_results["base_case"] = base_results

    if args.validate_only:
        print("\nValidation-only mode — done.")
        return

    # Step 2
    sens_results = run_sensitivity_analysis(base_results)
    all_results.update(sens_results)

    if args.no_ml:
        print("\nNo-ML mode — skipping Monte Carlo and ML.")
        generate_analysis_report(all_results)
        return

    # Steps 3 + 4
    mc_ml = run_mc_and_ml_pipeline(n_samples=args.mc_samples)
    mc_df = mc_ml["mc_df"]
    ml_results = mc_ml["ml_results"]

    # Summarise ML results for report
    comp_df = ml_results["comparison_df"]
    all_results["ml_summary"] = comp_df.to_string(index=False)

    # Step 5: report
    print("\n" + "=" * 65)
    print("  STEP 5 — ANALYSIS REPORT")
    print("=" * 65)
    generate_analysis_report(all_results)

    print("\n✓ Full pipeline complete. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
