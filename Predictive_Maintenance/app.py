"""
app.py — CNC machine health dashboard (Streamlit)
==================================================
Sidebar sliders for physical parameters, real-time physics feature
computation, model inference with risk gauge, and SHAP-based
fault diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent
SRC = ROOT / "src"
DATA = ROOT / "data"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import PhysicsFeatureEngineer
from explain import load_model, load_multilabel_models, explain_local_instance, get_feature_names
from model import prepare_features

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120


# ──────────────────────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CNC Machine Health Monitor",
    page_icon="⚙",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("⚙  CNC Machine Health Monitor")
st.markdown("Real-time fault risk assessment & SHAP-based diagnosis")


# ──────────────────────────────────────────────────────────────
#  Load model (cached)
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_assets():
    """Load model, scaler, and feature names (cached across reruns)."""
    try:
        model, scaler = load_model("xgboost")
    except (FileNotFoundError, Exception) as e:
        st.error(f"Model not found. Train it first:\n\n    cd {ROOT}\n    python src/model.py\n\n{e}")
        st.stop()
        return None, None, None, None, None

    try:
        feature_names = get_feature_names()
    except Exception as e:
        st.warning(f"Cannot read feature names: {e}. Using fallback.")
        feature_names = [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        ]
        # Load multi-label models
    try:
        ml_models, ml_scalers = load_multilabel_models()
    except (FileNotFoundError, Exception) as e:
        st.warning(f"Multi-label models not fully loaded: {e}")
        ml_models, ml_scalers = {}, {}

    return model, scaler, feature_names, ml_models, ml_scalers


model, scaler, FEATURE_NAMES, ML_MODELS, ML_SCALERS = load_assets()


# ──────────────────────────────────────────────────────────────
#  Sidebar — physical parameter sliders
# ──────────────────────────────────────────────────────────────

st.sidebar.header("Machine Parameters")

air_temp = st.sidebar.slider(
    "Air temperature [K]", 295.0, 305.0, 300.0, 0.1,
    help="Ambient air temperature (K). UCI dataset range: 295-305 K.",
)
proc_temp = st.sidebar.slider(
    "Process temperature [K]", air_temp + 0.1, 330.0, air_temp + 8.0, 0.1,
    help="Process temperature (K). Must be >= air temperature. Max: 330 K.",
)
rpm = st.sidebar.slider(
    "Rotational speed [rpm]", 1000, 3000, 1500, 10,
    help="Spindle rotational speed (RPM). Range: 1000-3000.",
)
torque = st.sidebar.slider(
    "Torque [Nm]", 3.0, 80.0, 30.0, 0.5,
    help="Cutting torque (Nm). Range: 3-80 Nm.",
)
tool_wear = st.sidebar.slider(
    "Tool wear [min]", 0, 250, 100, 1,
    help="Tool usage time (min). Range: 0-250 min.",
)


# ──────────────────────────────────────────────────────────────
#  Real-time physics feature computation
# ──────────────────────────────────────────────────────────────

st.subheader("Computed Physical Features")

raw_row = pd.DataFrame([{
    "Air temperature [K]":       air_temp,
    "Process temperature [K]":   proc_temp,
    "Rotational speed [rpm]":    rpm,
    "Torque [Nm]":               torque,
    "Tool wear [min]":           tool_wear,
}])

engineer = PhysicsFeatureEngineer(raw_row)
df_feat = engineer.transform()

P_m   = float(df_feat["P_m [W]"].iloc[0])
Delta_T = float(df_feat["Delta_T [K]"].iloc[0])
I_os  = float(df_feat["I_overstrain [min*Nm]"].iloc[0])

col1, col2, col3 = st.columns(3)
col1.metric("P\u2098 [W]", f"{P_m:.1f}",
            help="Mechanical cutting power = Torque x RPM x 2pi/60")
col2.metric("\u0394T [K]", f"{Delta_T:.2f}",
            help="Thermal dissipation = Process temp - Air temp")
col3.metric("I\u2092\u209b [min\u00b7Nm]", f"{I_os:.1f}",
            help="Overstrain index = Tool wear x Torque")


# ──────────────────────────────────────────────────────────────
#  Risk prediction
# ──────────────────────────────────────────────────────────────

st.subheader("Failure Risk Assessment")

# Build feature vector in correct order
feature_dict = {
    "Air temperature [K]":       air_temp,
    "Process temperature [K]":   proc_temp,
    "Rotational speed [rpm]":    rpm,
    "Torque [Nm]":               torque,
    "Tool wear [min]":           tool_wear,
    "P_m [W]":                   P_m,
    "Delta_T [K]":               Delta_T,
    "I_overstrain [min*Nm]":     I_os,

    "IQR_outlier":                False,
}

try:
    feature_vector = np.array([feature_dict[c] for c in FEATURE_NAMES],
                               dtype=np.float64).reshape(1, -1)
    feature_vector_scaled = scaler.transform(feature_vector)
    proba = float(model.predict_proba(feature_vector_scaled)[0, 1])
    pred_class = int(model.predict(feature_vector_scaled)[0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    proba = 0.0
    pred_class = 0
    feature_vector = np.array([feature_dict[c] for c in FEATURE_NAMES],
                               dtype=np.float64).reshape(1, -1)

# Risk gauge
risk_pct = proba * 100.0
if risk_pct < 10:
    color = "green"
    status = "Normal"
elif risk_pct < 50:
    color = "orange"
    status = "Caution"
else:
    color = "red"
    status = "!!! HIGH RISK !!!"

st.progress(int(risk_pct), text=f"{status}  —  Failure probability: {risk_pct:.1f}%")

col_status, col_prob = st.columns(2)
col_status.metric("Status", status,
                  delta=f"Class {'Failure' if pred_class else 'Normal'}")
col_prob.metric("Failure probability", f"{risk_pct:.2f} %",
                delta_color="off")


# ──────────────────────────────────────────────────────────────
#  SHAP local explanation
# ──────────────────────────────────────────────────────────────

# -------------------------------------------------------------------------------
#  Multi-label fault type prediction
# -------------------------------------------------------------------------------

st.subheader('Fault Type Analysis')

ml_probas = {}
if ML_MODELS:
    try:
        for ft in ['twf', 'hdf', 'pwf', 'osf']:
            if ft in ML_MODELS:
                ft_scaled = ML_SCALERS[ft].transform(feature_vector)
                ml_probas[ft] = float(ML_MODELS[ft].predict_proba(ft_scaled)[0, 1])
    except Exception as e:
        st.warning(f'Multi-label prediction failed: {e}')

if ml_probas:
    col_a, col_b, col_c, col_d = st.columns(4)
    ft_labels = {
        'twf': ('TWF', 'Tool Wear Failure'),
        'hdf': ('HDF', 'Heat Dissipation Failure'),
        'pwf': ('PWF', 'Power Failure'),
        'osf': ('OSF', 'Overstrain Failure'),
    }
    for col, ft in zip([col_a, col_b, col_c, col_d], ['twf', 'hdf', 'pwf', 'osf']):
        pct = ml_probas.get(ft, 0.0) * 100.0
        short, full = ft_labels[ft]
        col.metric(f'\u26a0\ufe0f {short}', f'{pct:.1f}%', help=full)
        if pct > 30:
            col.caption('\u26a0\ufe0f Elevated risk')
        else:
            col.caption('\u2705 Normal')

    # Model-based diagnosis
    diagnosis_lines = []
    if ml_probas.get('osf', 0) > 0.3:
        diagnosis_lines.append(
            '**High OSF risk** -- Overstrain Failure. **Action**: reduce cutting load.'
        )
    if ml_probas.get('hdf', 0) > 0.3:
        diagnosis_lines.append(
            '**High HDF risk** -- Heat Dissipation Failure. **Action**: check coolant system.'
        )
    if ml_probas.get('pwf', 0) > 0.3:
        diagnosis_lines.append(
            '**High PWF risk** -- Power Failure. **Action**: reduce spindle speed.'
        )
    if ml_probas.get('twf', 0) > 0.3:
        diagnosis_lines.append(
            '**High TWF risk** -- Tool Wear Failure. **Action**: replace cutting tool.'
        )

    if diagnosis_lines:
        st.error('### \u26a0\ufe0f  Maintenance Recommendations')
        for line in diagnosis_lines:
            st.write(f'- {line}')
    elif risk_pct > 50:
        st.info('No specific fault type exceeds threshold. Monitor trends.')
else:
    st.info('Multi-label models not loaded. Train with python src/model.py to enable fault type analysis.')# \u2500# \u2500# \u2500# \u2500# \u2500# \u2500# \u2500# \u2500# \u2500# \u2500\u2500\u2500\n#  Multi-label fault type prediction\n# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\nst.subheader("Fault Type Analysis")\n\nml_probas = {}\nif ML_MODELS:\n    try:\n        for ft in ["twf", "hdf", "pwf", "osf"]:\n            if ft in ML_MODELS:\n                ft_scaled = ML_SCALERS[ft].transform(feature_vector)\n                ml_probas[ft] = float(ML_MODELS[ft].predict_proba(ft_scaled)[0, 1])\n    except Exception as e:\n        st.warning(f"Multi-label prediction failed: {e}")\n\nif ml_probas:\n    col_a, col_b, col_c, col_d = st.columns(4)\n    ft_labels = {\n        "twf": ("TWF", "Tool Wear Failure"),\n        "hdf": ("HDF", "Heat Dissipation Failure"),\n        "pwf": ("PWF", "Power Failure"),\n        "osf": ("OSF", "Overstrain Failure"),\n    }\n    for col, ft in zip([col_a, col_b, col_c, col_d], ["twf", "hdf", "pwf", "osf"]):\n        pct = ml_probas.get(ft, 0.0) * 100.0\n        short, full = ft_labels[ft]\n        col.metric(f"\u26a0\ufe0f {short}", f"{pct:.1f}%", help=full)\n        if pct > 30:\n            col.caption("\u26a0\ufe0f Elevated risk")\n        else:\n            col.caption("\u2705 Normal")\n    \n    # Model-based diagnosis\n    diagnosis_lines = []\n    if ml_probas.get("osf", 0) > 0.3:\n        diagnosis_lines.append(\n            "**High OSF risk** -- Overstrain Failure. **Action**: reduce cutting load."\n        )\n    if ml_probas.get("hdf", 0) > 0.3:\n        diagnosis_lines.append(\n            "**High HDF risk** -- Heat Dissipation Failure. **Action**: check coolant system."\n        )\n    if ml_probas.get("pwf", 0) > 0.3:\n        diagnosis_lines.append(\n            "**High PWF risk** -- Power Failure. **Action**: reduce spindle speed."\n        )\n    if ml_probas.get("twf", 0) > 0.3:\n        diagnosis_lines.append(\n            "**High TWF risk** -- Tool Wear Failure. **Action**: replace cutting tool."\n        )\n    \n    if diagnosis_lines:\n        st.error("### \u26a0\ufe0f  Maintenance Recommendations")\n        for line in diagnosis_lines:\n            st.write(f"- {line}")\n    elif risk_pct > 50:\n        st.info("No specific fault type exceeds threshold. Monitor trends.")\nelse:\n    st.info("Multi-label models not loaded.")\n\n\nst.subheader("Fault Diagnosis — SHAP Local Explanation")

if risk_pct > 10.0:
    try:
        sample = feature_vector[0]
        explanation = explain_local_instance(
            model, scaler, sample, FEATURE_NAMES,
        )

        # Sort by absolute SHAP value for display
        contribs = sorted(
            explanation["contributions"], key=lambda x: abs(x["shap"]), reverse=True
        )

        st.markdown(
            f"**Base risk** (expected log-odds): {explanation['base_value']:.3f}  |  "
            f"**Predicted probability**: {explanation['prediction']:.4f}"
        )

        # Horizontal bar chart of SHAP contributions
        top_n = min(8, len(contribs))
        names = [c["feature"] for c in contribs[:top_n]][::-1]
        shap_vals = [c["shap"] for c in contribs[:top_n]][::-1]
        vals = [c["value"] for c in contribs[:top_n]][::-1]

        fig, ax = plt.subplots(figsize=(9, max(3, top_n * 0.4)))
        colors = ["#e74c3c" if v < 0 else "#27ae60" for v in shap_vals]
        ax.barh(range(top_n), shap_vals, color=colors, height=0.6)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("SHAP value (contribution to failure risk)")
        ax.set_title("Feature Contributions to Failure Risk", fontsize=11)

        # Annotate with actual values
        for i, (sv, val) in enumerate(zip(shap_vals, vals)):
            ha = "right" if sv < 0 else "left"
            offset = -0.05 if sv < 0 else 0.05
            ax.text(sv + offset, i, f"  {val:.1f}" if abs(val) < 100 else f" {val:.0f}",
                    va="center", ha=ha, fontsize=7, color="#555")

        fig.tight_layout()
        st.pyplot(fig)

        # Model-based diagnosis from multi-label predictions
        diagnosis_parts = []
        if ml_probas.get('osf', 0) > 0.3:
            diagnosis_parts.append('OSF (Overstrain Failure): reduce load, inspect tool.')
        if ml_probas.get('hdf', 0) > 0.3:
            diagnosis_parts.append('HDF (Heat Dissipation Failure): check coolant system.')
        if ml_probas.get('pwf', 0) > 0.3:
            diagnosis_parts.append('PWF (Power Failure): reduce spindle speed.')
        if ml_probas.get('twf', 0) > 0.3:
            diagnosis_parts.append('TWF (Tool Wear Failure): replace cutting tool.')

        if diagnosis_parts:
            st.error('### \u26a0\ufe0f  Maintenance Recommendations')
            for dp in diagnosis_parts:
                st.write(f'- {dp}')
        elif risk_pct > 50:
            st.warning('Risk elevated but no specific fault type exceeds threshold.')
        else:
            st.info('Moderate risk; monitor trends.')
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
else:
    st.info("Risk is low (below 10%). No SHAP diagnosis needed.")


# ──────────────────────────────────────────────────────────────
#  Footer
# ──────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption(
    "Based on UCI AI4I 2020 dataset | "
    "Model: XGBoost with scale_pos_weight"
)
