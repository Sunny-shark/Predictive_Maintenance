import sys
sys.path.insert(0, 'src')
# Write the patcher script
p = r'''
import re

# ─── model.py ───
with open('src/model.py', 'r', encoding='utf-8') as f:
    m = f.read()

# 1. Add MULTILABEL_TARGETS after _EXCLUDE_COLS
if 'MULTILABEL_TARGETS' not in m:
    m = m.replace(
        'def prepare_features',
        'MULTILABEL_TARGETS = ["TWF", "HDF", "PWF", "OSF"]\n\n\ndef prepare_features'
    )

# 2. Add train_multilabel function before if __name__
if 'def train_multilabel' not in m:
    train_fn = r'''
def train_multilabel(df):
    """Train separate classifiers for each failure type (TWF/HDF/PWF/OSF).
    For each target, runs GridSearchCV on XGBoost and LightGBM, selects best by test F1.
    """
    X, _, feature_cols = prepare_features(df)
    print("\\n" + "=" * 60)
    print("Multi-label fault type training")
    print("=" * 60)

    results = {}
    for target in MULTILABEL_TARGETS:
        y = df[target].values.astype(int)
        pos = y.sum()
        print(f"\\n--- {target}: {pos} positive / {len(y)} total ({100*pos/len(y):.2f}%) ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEED
        )

        best_f1, best_name, best_model, best_scaler = -1.0, "", None, None
        best_params_dict = {}

        for model_name in ["xgboost", "lightgbm"]:
            print(f"\\n  [{model_name}] Tuning...")
            params, cv_f1 = tune_hyperparameters(X_train, y_train, model_name)

            cv_result = train_cv(X_train, y_train, model_name=model_name,
                                 best_params=params, verbose=True)

            X_test_s = cv_result["scaler"].transform(X_test)
            y_pred = cv_result["final_model"].predict(X_test_s)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_prec = precision_score(y_test, y_pred, zero_division=0)

            print(f"  [{model_name}] Test: F1={test_f1:.4f}  recall={test_recall:.4f}  prec={test_prec:.4f}")

            if test_f1 > best_f1:
                best_f1 = test_f1
                best_name = model_name
                best_model = cv_result["final_model"]
                best_scaler = cv_result["scaler"]
                best_params_dict = params

        target_lower = target.lower()
        model_path = MODEL_DIR / f"best_model_{target_lower}.joblib"
        scaler_path = MODEL_DIR / f"scaler_{target_lower}.joblib"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)
        joblib.dump(best_scaler, scaler_path)
        print(f"  >>> Best: {best_name} (F1={best_f1:.4f}) -> {model_path.name}")

        # Confusion matrix
        X_test_s = best_scaler.transform(X_test)
        y_pred_final = best_model.predict(X_test_s)
        cm = confusion_matrix(y_test, y_pred_final)
        print(f"  Confusion matrix:\\n{cm}")

        results[target] = {"model_name": best_name, "test_f1": best_f1,
                           "best_params": best_params_dict}

    print("\\n" + "=" * 60)
    print("Multi-label training complete!")
    for target, r in results.items():
        print(f"  {target}: {r['model_name']} (F1={r['test_f1']:.4f})")
    return results


'''
    m = m.replace(
        "if __name__ == \"__main__\":",
        train_fn.strip() + "\n\n\nif __name__ == \"__main__\":"
    )

# 3. Add multilabel call at end of __main__
if 'train_multilabel(df)' not in m:
    m = m.replace(
        'print(f"Feature cols: {best_result['"'feature_cols'"]}")',
        'print(f"Feature cols: {best_result['"'feature_cols'"]}")\n\n    # Multi-label fault type training\n    print("\\n===== Multi-label fault type training =====")\n    train_multilabel(df)'
    )

with open('src/model.py', 'w', encoding='utf-8') as f:
    f.write(m)
print("model.py patched")

# ─── explain.py ───
with open('src/explain.py', 'r', encoding='utf-8') as f:
    e = f.read()

if 'def load_multilabel_models' not in e:
    loader = '''

def load_multilabel_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load all multi-label fault type models and their scalers.

    Returns
    -------
    (models_dict, scalers_dict)
        Keys are "twf", "hdf", "pwf", "osf".
    """
    models, scalers = {}, {}
    for t in ["twf", "hdf", "pwf", "osf"]:
        model_path = _MODEL_DIR / f"best_model_{t}.joblib"
        scaler_path = _MODEL_DIR / f"scaler_{t}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Multi-label model not found: {model_path}. "
                "Run src/model.py first to train and save."
            )
        models[t] = joblib.load(model_path)
        scalers[t] = joblib.load(scaler_path)
    return models, scalers
'''
    e = e.replace(
        'def get_feature_names',
        loader.strip() + '\n\n\ndef get_feature_names'
    )

with open('src/explain.py', 'w', encoding='utf-8') as f:
    f.write(e)
print("explain.py patched")

# ─── app.py ───
with open('app.py', 'r', encoding='utf-8') as f:
    a = f.read()

# 1. Update imports
if 'load_multilabel_models' not in a:
    a = a.replace(
        'from explain import load_model, explain_local_instance, get_feature_names',
        'from explain import load_model, load_multilabel_models, explain_local_instance, get_feature_names'
    )

# 2. Update load_assets to also load multilabel models
old_load = '''@st.cache_resource
def load_assets():
    """Load model, scaler, and feature names (cached across reruns)."""
    try:
        model, scaler = load_model("xgboost")
    except (FileNotFoundError, Exception) as e:
        st.error(f"Model not found. Train it first:\\n\\n    cd {ROOT}\\n    python src/model.py\\n\\n{e}")
        st.stop()
        return None, None, None

    try:
        feature_names = get_feature_names()
    except Exception as e:
        st.warning(f"Cannot read feature names: {e}. Using fallback.")
        feature_names = [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        ]
    return model, scaler, feature_names'''

new_load = '''@st.cache_resource
def load_assets():
    """Load model, scaler, and feature names (cached across reruns)."""
    try:
        model, scaler = load_model("xgboost")
    except (FileNotFoundError, Exception) as e:
        st.error(f"Model not found. Train it first:\\n\\n    cd {ROOT}\\n    python src/model.py\\n\\n{e}")
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
        st.warning(f"Multi-label models not available: {e}")
        ml_models, ml_scalers = {}, {}

    return model, scaler, feature_names, ml_models, ml_scalers'''

a = a.replace(old_load, new_load)

# 3. Update the unpacking
a = a.replace(
    'model, scaler, FEATURE_NAMES = load_assets()',
    'model, scaler, FEATURE_NAMES, ML_MODELS, ML_SCALERS = load_assets()'
)

# 4. Add multi-label prediction block after binary prediction
# Find the risk gauge section and insert after it
old_diag_start = '''st.subheader("Fault Diagnosis \\u2014 SHAP Local Explanation")'''

new_ml_section = '''
# \\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500
#  Multi-label fault type prediction
# \\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500

st.subheader("Fault Type Analysis")

ml_probas = {}
if ML_MODELS:
    try:
        for ft in ["twf", "hdf", "pwf", "osf"]:
            if ft in ML_MODELS:
                ft_scaled = ML_SCALERS[ft].transform(feature_vector)
                ml_probas[ft] = float(ML_MODELS[ft].predict_proba(ft_scaled)[0, 1])
    except Exception as e:
        st.warning(f"Multi-label prediction failed: {e}")

if ml_probas:
    col_a, col_b, col_c, col_d = st.columns(4)
    ft_labels = {
        "twf": ("TWF", "Tool Wear Failure"),
        "hdf": ("HDF", "Heat Dissipation Failure"),
        "pwf": ("PWF", "Power Failure"),
        "osf": ("OSF", "Overstrain Failure"),
    }
    for col, ft in zip([col_a, col_b, col_c, col_d], ["twf", "hdf", "pwf", "osf"]):
        pct = ml_probas.get(ft, 0.0) * 100.0
        short, full = ft_labels[ft]
        if pct < 10:
            fc = "normal"
        elif pct < 50:
            fc = "off"
        else:
            fc = "inverse"
        col.metric(f"\\u26a0\\ufe0f {short}", f"{pct:.1f}%",
                   help=full, delta_color=fc)
        if pct > 30:
            col.caption("\\ud83d\\udd25 Elevated risk")
        else:
            col.caption("\\u2705 Normal")

    # Diagnosis text replacing old heuristic rules
    diagnosis_lines = []
    if ml_probas.get("osf", 0) > 0.3:
        diagnosis_lines.append(
            "**High OSF risk** \\u2014 Overstrain Failure. The tool wear combined with "
            "cutting torque has exceeded the safe threshold. **Action**: reduce cutting "
            "load, inspect tool condition."
        )
    if ml_probas.get("hdf", 0) > 0.3:
        diagnosis_lines.append(
            "**High HDF risk** \\u2014 Heat Dissipation Failure. Poor thermal dissipation "
            "detected. **Action**: check coolant system, improve thermal management."
        )
    if ml_probas.get("pwf", 0) > 0.3:
        diagnosis_lines.append(
            "**High PWF risk** \\u2014 Power Failure. Excessive power draw detected. "
            "**Action**: reduce spindle speed, inspect motor drive."
        )
    if ml_probas.get("twf", 0) > 0.3:
        diagnosis_lines.append(
            "**High TWF risk** \\u2014 Tool Wear Failure. Tool wear has reached critical "
            "levels. **Action**: replace or regrind cutting tool immediately."
        )

    if diagnosis_lines:
        st.error("### \\u26a0\\ufe0f  Maintenance Recommendations")
        for line in diagnosis_lines:
            st.write(f"- {line}")
    elif risk_pct > 10:
        st.info("No specific fault type exceeds risk threshold. Monitor trends.")
else:
    st.info("Multi-label models not loaded. Train with python src/model.py to enable fault type analysis.")
'''

a = a.replace(old_diag_start, new_ml_section.strip() + '\\n\\n' + old_diag_start)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(a)
print("app.py patched")
'''

with open('patch_multilabel.py', 'w', encoding='utf-8') as f:
    f.write(p)
print("Patcher script written")
