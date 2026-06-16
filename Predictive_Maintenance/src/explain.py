"""
explain.py — SHAP explainability module
========================================
- Load the persisted best model and compute SHAP values.
- Global: summary bar plot saved to data/shap_summary.png
- Local:  explain_local_instance(sample) -> force-plot data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
import joblib

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# File paths
_MODEL_DIR = Path(__file__).resolve().parent.parent / "data"


# ──────────────────────────────────────────────────────────────
#  Model / scaler loader
# ──────────────────────────────────────────────────────────────

def load_model(model_name: str = "xgboost") -> Tuple[Any, Any]:
    """Load the persisted model and scaler from data/.

    Returns
    -------
    (model, scaler)
    """
    model_path = _MODEL_DIR / f"best_model_{model_name}.joblib"
    scaler_path = _MODEL_DIR / f"scaler_{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run src/model.py first to train and save."
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ──────────────────────────────────────────────────────────────
#  Feature names loader (reuses model module logic)
# ──────────────────────────────────────────────────────────────

def load_multilabel_models():
    models, scalers = {}, {}
    for t in ['twf', 'hdf', 'pwf', 'osf']:
        mp = _MODEL_DIR / f'best_model_{t}.joblib'
        sp = _MODEL_DIR / f'scaler_{t}.joblib'
        if not mp.exists():
            raise FileNotFoundError(f'Multi-label model not found: {mp}')
        models[t] = joblib.load(mp)
        scalers[t] = joblib.load(sp)
    return models, scalers


def get_feature_names() -> List[str]:
    """Return the list of feature column names used by the model."""
    from model import prepare_features
    from features import load_data, PhysicsFeatureEngineer

    data_path = _MODEL_DIR / "ai4i2020.csv"
    df = load_data(str(data_path))
    df = PhysicsFeatureEngineer(df).transform()
    _, _, names = prepare_features(df)
    return names


# ──────────────────────────────────────────────────────────────
#  Global SHAP summary plot  (saved to data/shap_summary.png)
# ──────────────────────────────────────────────────────────────

def generate_summary_plot(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None,
) -> str:
    """Compute SHAP values and save a global summary bar plot.

    The plot is saved to ``save_path`` (default: data/shap_summary.png).
    Returns the absolute path of the saved image.
    """
    if save_path is None:
        save_path = _MODEL_DIR / "shap_summary.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # TreeExplainer for tree-based models (XGBoost / LightGBM)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # fallback to KernelExplainer
        explainer = shap.KernelExplainer(
            model.predict_proba, shap.kmeans(X_background, min(50, len(X_background))),
        )

    shap_values = explainer.shap_values(X_explain)

    # Handle multi-class output -> only take class-1 (failure)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif shap_values.ndim == 3:
        sv = shap_values[..., 1] if shap_values.shape[2] > 1 else shap_values[..., 0]
    else:
        sv = shap_values

    # Bar plot
    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_names) * 0.3)))
    shap.summary_plot(sv, X_explain, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    ax.set_title("SHAP 特征重要性（故障类）", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"SHAP summary plot saved -> {save_path}")
    return str(save_path.resolve())


# ──────────────────────────────────────────────────────────────
#  Local instance explainer
# ──────────────────────────────────────────────────────────────

def explain_local_instance(
    model: Any,
    scaler: Any,
    sample: np.ndarray,
    feature_names: List[str],
    true_label: Optional[int] = None,
    background: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Explain a single sample's prediction with SHAP.

    Parameters
    ----------
    sample : 1-D array-like of shape (n_features,)

    Returns
    -------
    dict with keys:
        shap_values  : 1-D array of SHAP values per feature
        base_value   : model expected output
        prediction   : predicted probability of failure
        pred_class   : 0 or 1
        contributions: list of dicts {name, value, shap} for display
    """
    sample = np.asarray(sample, dtype=np.float64).reshape(1, -1)
    sample_scaled = scaler.transform(sample)

    # Prediction
    proba = model.predict_proba(sample_scaled)[0, 1]
    pred_class = int(model.predict(sample_scaled)[0])

    # SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        bg_data = scaler.transform(background) if background is not None else None
        if bg_data is None:
            bg_data = np.random.RandomState(42).randn(100, sample_scaled.shape[1])
        explainer = shap.KernelExplainer(
            model.predict_proba, bg_data,
        )

    shap_vals = explainer.shap_values(sample_scaled)

    # Extract the failure-class SHAP values
    if isinstance(shap_vals, list):
        sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    elif shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1] if shap_vals.shape[2] > 1 else shap_vals[0, :, 0]
    else:
        sv = shap_vals[0]

    # Build contribution list
    contributions = []
    for i, name in enumerate(feature_names):
        direction_str = ">> 推高故障" if sv[i] > 0 else "<< 拉低故障"
        contributions.append({
            "feature": name,
            "value": float(sample[0, i]),
            "shap": float(sv[i]),
            "direction": direction_str,
        })
    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "shap_values": sv.tolist(),
        "base_value": float(explainer.expected_value[1]
                            if isinstance(explainer.expected_value, list)
                            else explainer.expected_value),
        "prediction": float(proba),
        "pred_class": pred_class,
        "contributions": contributions,
    }


# ──────────────────────────────────────────────────────────────
#  Convenience: load model + generate everything
# ──────────────────────────────────────────────────────────────

def run_full_xai(model_name: str = "xgboost") -> Dict[str, Any]:
    """Load model, generate global SHAP plot, and return explanation objects."""
    from features import load_data, PhysicsFeatureEngineer
    from model import prepare_features

    print(f"Loading model '{model_name}' ...")
    model, scaler = load_model(model_name)

    print("Loading data ...")
    data_path = _MODEL_DIR / "ai4i2020.csv"
    df = load_data(str(data_path))
    df = PhysicsFeatureEngineer(df).transform()
    X, y, feature_names = prepare_features(df)

    # Background: use 200 random training samples
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X), min(200, len(X)), replace=False)
    X_bg = X[bg_idx]

    # SHAP summary plot (use failure samples + some normal)
    failure_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]
    explain_idx = np.concatenate([
        failure_idx[:min(50, len(failure_idx))],
        normal_idx[:min(50, len(normal_idx))],
    ])
    X_explain = X[explain_idx]

    summary_path = generate_summary_plot(
        model, scaler.transform(X_bg), scaler.transform(X_explain),
        feature_names,
    )

    # Local explanation for the first failure sample
    local_result = None
    normal_result = None
    if len(failure_idx) > 0:
        sample = X[failure_idx[0]]
        local_result = explain_local_instance(
            model, scaler, sample, feature_names, true_label=1,
            background=X_bg,
        )
    if len(normal_idx) > 0:
        sample_n = X[normal_idx[0]]
        normal_result = explain_local_instance(
            model, scaler, sample_n, feature_names, true_label=0,
            background=X_bg,
        )

    return {
        "summary_plot_path": summary_path,
        "local_explanation": local_result,
        "normal_explanation": normal_result,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }


# ──────────────────────────────────────────────────────────────
#  CLI quick-run
# ──────────────────────────────────────────────────────────────


def _quick_evaluate(model_name):
    """Load model and data, return recall score for model comparison."""
    from features import load_data, PhysicsFeatureEngineer
    from model import prepare_features
    from sklearn.metrics import recall_score
    from sklearn.model_selection import train_test_split

    print("  Evaluating " + model_name + " recall (on held-out 20%) ...")
    model, scaler = load_model(model_name)
    data_path = _MODEL_DIR / "ai4i2020.csv"
    df = load_data(str(data_path))
    df = PhysicsFeatureEngineer(df).transform()
    X, y, _ = prepare_features(df)
    # Split to get held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)
    return recall_score(y_test, y_pred, zero_division=0)


# ---------------------------------------------------------------------------
#  CLI quick-run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    best_name, best_recall = "", -1.0
    for name in ["xgboost", "lightgbm"]:
        if not (_MODEL_DIR / ("best_model_" + name + ".joblib")).exists():
            print("  [skip] " + name + " model file not found, skipped.")
            continue
        rec = _quick_evaluate(name)
        print("  " + name + " recall = " + str(round(rec, 4)))
        if rec > best_recall:
            best_recall = rec
            best_name = name

    if not best_name:
        print("\nNo model available! Run model.py first.")
        exit(1)

    print("\n>>> Selected " + best_name + " (highest recall = " + str(round(best_recall, 4)) + ")\n")
    result = run_full_xai(best_name)

    sep = "=" * 55
    print("\n" + sep)
    print("Model: " + best_name)
    print(sep)
    print("Global SHAP plot: " + result["summary_plot_path"] + "\n")

    for label, key in [("Failure sample", "local_explanation"), ("Normal sample", "normal_explanation")]:
        exp = result.get(key)
        if not exp:
            continue
        prob = exp["prediction"]
        cls = exp["pred_class"]
        print(">>> " + label + ":")
        print("    Prediction prob = " + str(round(prob, 4)) + "  (class " + str(cls) + ")")
        print("    Top-5 features:")
        header = "    {:28s} {:>8s}  {:>8s}  {:s}".format("Feature", "Value", "SHAP", "Direction")
        print(header)
        print("    " + "-" * 55)
        for c in exp["contributions"][:5]:
            line = "    {:28s} {:8.3f}  {:+8.4f}  {:s}".format(c["feature"], c["value"], c["shap"], c["direction"])
            print(line)
        print()

    print("Explanation complete!")
