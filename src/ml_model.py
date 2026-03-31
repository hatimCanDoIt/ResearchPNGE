"""
ml_model.py — ML model to predict and correct FMB estimation errors.

Problem formulation
-------------------
Target (preferred): G_correction = G_volumetric / G_fmb   (dimensionless, ~1.0)
  - Multiply G_fmb × G_correction to get an improved OGIP estimate.
  - Value < 1 → FMB overestimated G.
  - Value > 1 → FMB underestimated G.

Alternative target: G_fmb_error_pct = 100 × (G_fmb − G_vol) / G_vol

Algorithms
----------
1. Random Forest    (sklearn)  — baseline, interpretable feature importance
2. XGBoost          (xgboost)  — best tabular performance, early stopping
3. MLP Regressor    (sklearn)  — neural baseline
4. PINN             (PyTorch)  — physics-informed: adds MBE consistency loss

Libraries: scikit-learn, xgboost, torch, pandas, numpy, joblib, matplotlib
Platform:  Python 3.10+, local CPU or Google Colab (GPU for PINN)
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("xgboost not installed — XGBoost model will be skipped.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed — PINN model will be skipped.")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Features used by all ML models
FEATURE_COLS = [
    # Raw reservoir params
    "pi", "T_F", "gamma_g", "A_acres", "h", "phi", "Swi",
    "k", "D", "s", "N_wells",
    # PVT at initial conditions
    "zi", "mu_i", "Bgi", "Tpc", "Ppc", "Tpr_i", "Ppr_i",
    # Flow features
    "re", "A_coeff", "B_coeff", "a_mp", "b_mp", "ln_re_rw",
    # Non-Darcy significance
    "B_over_A", "D_q_dimensionless", "nondarcy_fraction_initial",
    # Production history
    "Gp_final_fraction", "t_max_days", "avg_rate_MMcfd",
    "pressure_depletion_fraction",
]

TARGET_COL = "G_correction"          # primary target
ALT_TARGET_COL = "G_fmb_error_pct"  # alternative


def prepare_features(
    df: pd.DataFrame,
    target: str = TARGET_COL,
) -> tuple:
    """
    Select and clean features from Monte Carlo DataFrame.

    Returns
    -------
    X : pd.DataFrame  Feature matrix (cleaned, no NaN/inf)
    y : pd.Series     Target vector
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    y = df[target].copy()

    # Remove rows with NaN or inf in features or target
    mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Clip extreme target values (> 5x or < 0.2x correction is likely a simulation artifact)
    clip_mask = (y >= 0.1) & (y <= 10.0)
    X = X[clip_mask].reset_index(drop=True)
    y = y[clip_mask].reset_index(drop=True)

    return X, y


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Standard random split (Monte Carlo data is i.i.d.)."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# ---------------------------------------------------------------------------
# Model 1: Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparams: dict = None,
    tune: bool = False,
    seed: int = 42,
) -> RandomForestRegressor:
    """
    Train a Random Forest regressor.

    Hyperparameters (defaults)
    --------------------------
    n_estimators=300, max_depth=None, min_samples_leaf=5,
    max_features='sqrt', random_state=seed, n_jobs=-1

    If tune=True: RandomizedSearchCV with 20 iterations.
    """
    if hyperparams is None:
        hyperparams = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": seed,
            "n_jobs": -1,
        }

    if tune:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 3, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        }
        base_model = RandomForestRegressor(random_state=seed, n_jobs=-1)
        search = RandomizedSearchCV(
            base_model, param_dist, n_iter=20, cv=5,
            scoring="r2", n_jobs=-1, random_state=seed
        )
        search.fit(X_train, y_train)
        return search.best_estimator_

    model = RandomForestRegressor(**hyperparams)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Model 2: XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    hyperparams: dict = None,
    seed: int = 42,
):
    """
    Train an XGBoost regressor with optional early stopping.

    Hyperparameters (defaults)
    --------------------------
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=50
    """
    if not XGB_AVAILABLE:
        warnings.warn("XGBoost not available — returning None.")
        return None

    if hyperparams is None:
        hyperparams = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }

    model = xgb.XGBRegressor(**hyperparams)

    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Model 3: MLP (sklearn)
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparams: dict = None,
    seed: int = 42,
) -> tuple:
    """
    Train an MLP regressor (StandardScaler + MLPRegressor).

    Architecture: (128, 64, 32) with ReLU, Adam optimizer.

    Returns
    -------
    (scaler, model) — must apply scaler.transform before predicting.
    """
    if hyperparams is None:
        hyperparams = {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": seed,
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(**hyperparams)
    model.fit(X_scaled, y_train)
    return scaler, model


# ---------------------------------------------------------------------------
# Model 4: Physics-Informed Neural Network (PyTorch)
# ---------------------------------------------------------------------------

class PINNRegressor(nn.Module if TORCH_AVAILABLE else object):
    """
    Physics-Informed Neural Network for FMB correction.

    Architecture: [n_features → 128 → 128 → 64 → 1]  with ReLU.

    Physics loss: enforce that when G_correction = 1 (perfect FMB),
    the predicted correction should equal 1 for low non-Darcy fraction.
    More specifically, penalise predictions that deviate from 1.0 when
    nondarcy_fraction_initial ≈ 0 (pure Darcy flow, FMB should be exact).
    """

    def __init__(self, n_features: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PINNRegressor.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_pinn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    physics_weight: float = 0.1,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple:
    """
    Train the PINN regressor.

    Loss
    ----
    L_data    = MSE(predicted, true_correction)
    L_physics = MSE of predicted values towards 1.0 for samples where
                nondarcy_fraction_initial < 0.01 (nearly pure Darcy)
    L_total   = L_data + physics_weight × L_physics

    Returns
    -------
    (scaler, model, train_losses, val_losses)
    """
    if not TORCH_AVAILABLE:
        warnings.warn("PyTorch not available — PINN training skipped.")
        return None, None, [], []

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_train.values.astype(np.float32))
    y_np = y_train.values.astype(np.float32)

    # Index of nondarcy_fraction_initial feature (for physics loss)
    nd_idx = (
        list(X_train.columns).index("nondarcy_fraction_initial")
        if "nondarcy_fraction_initial" in X_train.columns else None
    )

    X_t = torch.tensor(X_np)
    y_t = torch.tensor(y_np)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PINNRegressor(n_features=X_np.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            L_data = mse_loss(pred, yb)

            # Physics loss: for near-Darcy samples, correction should ≈ 1.0
            L_physics = torch.tensor(0.0)
            if nd_idx is not None:
                nd_vals = xb[:, nd_idx]   # scaled, so check roughly 0
                darcy_mask = (nd_vals < 0.0)   # below mean in scaled space ≈ low nd
                if darcy_mask.sum() > 0:
                    L_physics = mse_loss(pred[darcy_mask],
                                         torch.ones(darcy_mask.sum()))

            loss = L_data + physics_weight * L_physics
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        train_losses.append(epoch_loss / len(dataset))

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(scaler.transform(X_val.values.astype(np.float32)))
                y_val_t = torch.tensor(y_val.values.astype(np.float32))
                val_pred = model(X_val_t)
                val_loss = mse_loss(val_pred, y_val_t).item()
            val_losses.append(val_loss)

    return scaler, model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    scaler=None,
) -> dict:
    """
    Compute R², RMSE, MAE, MAPE for a fitted model.

    Handles tree models (no scaler), MLP + PINN (with scaler).
    """
    if model is None:
        return {"model_name": model_name, "R2": None, "RMSE": None, "MAE": None, "MAPE": None}

    X_eval = X_test.copy()

    if scaler is not None and not TORCH_AVAILABLE:
        # sklearn MLP
        X_eval_s = scaler.transform(X_eval)
        y_pred = model.predict(X_eval_s)
    elif scaler is not None and TORCH_AVAILABLE and isinstance(model, PINNRegressor):
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(scaler.transform(X_eval.values.astype(np.float32)))
            y_pred = model(X_t).numpy()
    elif scaler is not None:
        # MLP with StandardScaler
        X_eval_s = scaler.transform(X_eval)
        y_pred = model.predict(X_eval_s)
    else:
        y_pred = model.predict(X_eval)

    y_true = y_test.values
    R2 = r2_score(y_true, y_pred)
    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))
    MAE = mean_absolute_error(y_true, y_pred)
    # MAPE (avoid divide-by-zero)
    mask = y_true != 0
    MAPE = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

    return {
        "model_name": model_name,
        "R2": R2, "RMSE": RMSE, "MAE": MAE, "MAPE_pct": MAPE,
        "y_pred": y_pred, "y_true": y_true,
    }


def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Tabulate metrics for all trained models.

    Parameters
    ----------
    metrics_list : list of dicts from evaluate_model()

    Returns
    -------
    pd.DataFrame with columns: model_name, R2, RMSE, MAE, MAPE_pct
    """
    rows = [
        {
            "model_name": m["model_name"],
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "MAPE_pct": m["MAPE_pct"],
        }
        for m in metrics_list if m["R2"] is not None
    ]
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model,
    feature_names: list,
    model_type: str = "tree",
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    scaler=None,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Extract feature importance.

    For tree models: model.feature_importances_ (built-in impurity-based).
    For MLP/PINN: permutation importance on test set.

    Returns
    -------
    pd.DataFrame with columns: feature, importance, rank
    """
    if model_type == "tree":
        importances = model.feature_importances_
    else:
        # Permutation importance
        if X_test is None:
            raise ValueError("X_test required for permutation importance.")
        X_eval = X_test.copy()
        if scaler is not None:
            X_eval = pd.DataFrame(scaler.transform(X_eval), columns=feature_names)

        result = permutation_importance(
            model, X_eval, y_test, n_repeats=n_repeats,
            random_state=seed, n_jobs=-1
        )
        importances = result.importances_mean

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(
    metrics: dict,
    save_path: str = None,
) -> None:
    """Scatter plot of predicted vs actual G_correction with identity line."""
    if save_path is None:
        name = metrics["model_name"].replace(" ", "_")
        save_path = os.path.join(RESULTS_DIR, f"pred_vs_actual_{name}.png")

    y_pred = metrics["y_pred"]
    y_true = metrics["y_true"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Identity (perfect)")
    ax.set_xlabel("True G_correction", fontsize=11)
    ax.set_ylabel("Predicted G_correction", fontsize=11)
    ax.set_title(f"{metrics['model_name']} — Predicted vs Actual\n"
                 f"R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot saved] {save_path}")


def plot_feature_importance(
    fi_df: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    save_path: str = None,
) -> None:
    """Horizontal bar chart of top-N feature importances."""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"feature_importance_{model_name.replace(' ', '_')}.png")

    top = fi_df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue", alpha=0.85)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot saved] {save_path}")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str, model_type: str = "sklearn") -> None:
    """
    Save fitted model to disk.

    tree / sklearn : joblib.dump
    pinn           : torch.save(state_dict)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if model_type == "pinn" and TORCH_AVAILABLE:
        torch.save(model.state_dict(), path)
    else:
        joblib.dump(model, path)
    print(f"  [Model saved] {path}")


def load_model(path: str, model_type: str = "sklearn", n_features: int = None):
    """Load a previously saved model."""
    if model_type == "pinn" and TORCH_AVAILABLE:
        model = PINNRegressor(n_features=n_features)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Convenience: run full ML pipeline on a Monte Carlo DataFrame
# ---------------------------------------------------------------------------

def run_ml_pipeline(
    mc_df: pd.DataFrame,
    target: str = TARGET_COL,
    seed: int = 42,
    save_models: bool = True,
) -> dict:
    """
    End-to-end ML pipeline.

    Steps
    -----
    1. Feature engineering
    2. Train/test split (80/20)
    3. Train RF, XGBoost, MLP, PINN
    4. Evaluate all models
    5. Compare metrics
    6. Feature importance (RF + XGBoost)
    7. Save best model

    Returns
    -------
    dict with keys: comparison_df, feature_importance, metrics_list, best_model_name
    """
    print("\n--- ML PIPELINE START ---")

    X, y = prepare_features(mc_df, target=target)
    print(f"  Dataset: {len(X)} samples × {X.shape[1]} features")

    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)
    X_tr2, X_val, y_tr2, y_val = split_data(X_train, y_train, test_size=0.15, seed=seed)

    metrics_list = []

    # 1. Random Forest
    print("  Training Random Forest …")
    rf = train_random_forest(X_train, y_train, seed=seed)
    m_rf = evaluate_model(rf, X_test, y_test, "Random Forest")
    metrics_list.append(m_rf)
    plot_predicted_vs_actual(m_rf)

    # 2. XGBoost
    if XGB_AVAILABLE:
        print("  Training XGBoost …")
        xgb_m = train_xgboost(X_tr2, y_tr2, X_val, y_val, seed=seed)
        if xgb_m is not None:
            m_xgb = evaluate_model(xgb_m, X_test, y_test, "XGBoost")
            metrics_list.append(m_xgb)
            plot_predicted_vs_actual(m_xgb)
    else:
        xgb_m = None

    # 3. MLP
    print("  Training MLP …")
    mlp_scaler, mlp = train_mlp(X_train, y_train, seed=seed)
    m_mlp = evaluate_model(mlp, X_test, y_test, "MLP", scaler=mlp_scaler)
    metrics_list.append(m_mlp)
    plot_predicted_vs_actual(m_mlp)

    # 4. PINN
    pinn_scaler, pinn_model = None, None
    if TORCH_AVAILABLE:
        print("  Training PINN …")
        pinn_scaler, pinn_model, tr_losses, val_losses = train_pinn(
            X_tr2, y_tr2, X_val, y_val, epochs=150, seed=seed
        )
        if pinn_model is not None:
            m_pinn = evaluate_model(pinn_model, X_test, y_test, "PINN", scaler=pinn_scaler)
            metrics_list.append(m_pinn)
            plot_predicted_vs_actual(m_pinn)

    # Comparison table
    comp_df = compare_models(metrics_list)
    print("\n  MODEL COMPARISON:")
    print(comp_df.to_string(index=False))

    # Feature importance
    feature_names = list(X_train.columns)
    fi_rf = get_feature_importance(rf, feature_names, model_type="tree")
    plot_feature_importance(fi_rf, "Random Forest")

    fi_xgb = None
    if XGB_AVAILABLE and xgb_m is not None:
        fi_xgb = get_feature_importance(xgb_m, feature_names, model_type="tree")
        plot_feature_importance(fi_xgb, "XGBoost")

    # Save best model
    best_name = comp_df.iloc[0]["model_name"]
    if save_models:
        model_map = {
            "Random Forest": (rf, "sklearn"),
            "XGBoost": (xgb_m, "sklearn"),
            "MLP": (mlp, "sklearn"),
            "PINN": (pinn_model, "pinn"),
        }
        best_model_obj, best_type = model_map.get(best_name, (rf, "sklearn"))
        if best_model_obj is not None:
            save_model(
                best_model_obj,
                os.path.join(RESULTS_DIR, f"best_model_{best_name.replace(' ', '_')}.pkl"),
                model_type=best_type,
            )
            if best_name == "MLP":
                save_model(mlp_scaler, os.path.join(RESULTS_DIR, "mlp_scaler.pkl"))
            if best_name == "PINN" and pinn_scaler is not None:
                save_model(pinn_scaler, os.path.join(RESULTS_DIR, "pinn_scaler.pkl"))

    print("--- ML PIPELINE DONE ---\n")

    return {
        "comparison_df": comp_df,
        "metrics_list": metrics_list,
        "feature_importance_rf": fi_rf,
        "feature_importance_xgb": fi_xgb,
        "best_model_name": best_name,
        "X_test": X_test,
        "y_test": y_test,
    }
