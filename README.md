# ResearchPNGE — FMB / Volumetric Validation & ML Correction

**Author:** Hatem AlSakhboori  

Python research pipeline for gas material balance (FMB) validation, sensitivity analysis, Monte Carlo dataset generation, and machine-learning correction of FMB OGIP estimates.

## What it does

1. **Base-case validation** — Compares PVT, geometry, volumetric OGIP, inflow A/B coefficients, and FMB self-consistency against known targets.
2. **Sensitivity analysis** — Studies non-Darcy coefficient D, A/B factor accuracy, and production-period length effects on FMB bias.
3. **Monte Carlo** — Generates synthetic reservoir scenarios with FMB errors.
4. **ML pipeline** — Trains Random Forest, XGBoost, MLP, and PINN models to predict FMB correction factors; saves the best model.
5. **Reports & plots** — Writes CSVs, PNGs, and `data/results/analysis_report.txt`.

## Quick start

```bash
pip install -r requirements.txt
python main.py --validate-only
```

Expected output ends with `Overall: ALL CHECKS PASSED` (~30 seconds).

### Other run modes

```bash
# Validation + sensitivity only (no ML; ~1–2 min)
python main.py --no-ml

# Full pipeline with smaller Monte Carlo sample (faster test)
python main.py --mc-samples 200

# Full pipeline (default 5000 MC samples; can take a while)
python main.py
```

### Run tests

```bash
python -m pytest tests/ -q
```

## Project layout

```
ResearchPNGE/
├── main.py              # CLI entry point
├── config.py            # Base-case inputs, validation targets, MC ranges
├── requirements.txt
├── src/
│   ├── gas_properties.py
│   ├── volumetric.py
│   ├── inflow.py
│   ├── material_balance.py
│   ├── simulation.py
│   ├── analysis.py
│   └── ml_model.py
├── tests/               # Unit tests (pytest)
└── data/results/        # Generated outputs (gitignored except .gitkeep)
```

## Outputs

After a full or partial run, check `data/results/`:

| File | Description |
|------|-------------|
| `fmb_validation.png` | Mattar & Anderson p/z vs Gp plot |
| `nondarcy_sensitivity.png` | FMB bias vs non-Darcy D |
| `AB_sensitivity_heatmap.png` | FMB error vs A/B multipliers |
| `sensitivity_*.csv` | Sensitivity tables |
| `monte_carlo_dataset.csv` | Synthetic training data (full pipeline) |
| `analysis_report.txt` | Text summary |
| `best_model.pkl` | Best ML model (full pipeline) |

## Configuration

Edit `config.py` to change:

- `BASE_CASE` — reservoir, well, and simulation parameters
- `VALIDATION_TARGETS` — screenshot / reference values
- `MC_SETTINGS` — Monte Carlo sampling ranges

## Notes

- **Windows console:** `main.py` configures UTF-8 output so Greek symbols in validation reports do not crash on cp1252 terminals.
- **XGBoost / PyTorch:** Optional at import time; XGBoost is skipped if not installed. PyTorch is required only for the PINN step in the full ML pipeline.
- **Results folder:** CSV/PNG/PKL outputs are gitignored; only `data/results/.gitkeep` is tracked.

## Related work

A separate PINN workbench for field history prediction lives in the FMB research repository. This repo focuses on FMB physics validation, sensitivity, and tabular/ML correction of FMB OGIP error.
