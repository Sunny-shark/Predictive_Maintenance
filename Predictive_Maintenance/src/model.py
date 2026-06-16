"""
model.py - Robust model training pipeline
==========================================
StratifiedKFold cross-validation with scale_pos_weight for
imbalanced binary classification.  Evaluation prioritises
recall and PR-AUC over raw accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
TARGET_COL = "Machine failure"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data"

_EXCLUDE_COLS = {
    "UDI", "Product ID", "Type",
    "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
}

MULTILABEL_TARGETS = ["TWF", "HDF", "PWF", "OSF"]


def prepare_features(df):
    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
    X = df[feature_cols].values.astype(float)
    y = df[TARGET_COL].values.astype(int)
    return X, y, feature_cols


def _find_best_threshold(y_true, y_proba, n_thresholds=50, f1_tolerance=0.01):
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    candidates = []
    best_f1 = 0.0
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        candidates.append((th, rec, prec, f1))
        if f1 > best_f1:
            best_f1 = f1
    min_f1 = best_f1 * (1.0 - f1_tolerance)
    valid = [c for c in candidates if c[3] >= min_f1]
    return max(valid, key=lambda c: c[1])[0]


def _scale_pos_weight(y):
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return float(neg / pos) if pos else 1.0


def build_xgboost(y_train, **best_params):
    spw = _scale_pos_weight(y_train)
    params = dict(
        n_estimators=best_params.get("n_estimators", 300),
        max_depth=best_params.get("max_depth", 5),
        learning_rate=best_params.get("learning_rate", 0.08),
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", use_label_encoder=False,
        random_state=SEED, verbosity=0,
    )
    return xgb.XGBClassifier(**params)


def build_lightgbm(y_train, **best_params):
    spw = _scale_pos_weight(y_train)
    params = dict(
        n_estimators=best_params.get("n_estimators", 300),
        max_depth=best_params.get("max_depth", 5),
        learning_rate=best_params.get("learning_rate", 0.08),
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=SEED, verbose=-1,
    )
    return lgb.LGBMClassifier(**params)


def tune_hyperparameters(X, y, model_name="xgboost", cv_folds=5):
    spw = _scale_pos_weight(y)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    if model_name == "xgboost":
        model = xgb.XGBClassifier(scale_pos_weight=spw, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss",
            use_label_encoder=False, random_state=SEED, verbosity=0)
        param_grid = {"n_estimators": [100,200,300], "max_depth": [3,5,7], "learning_rate": [0.05,0.08,0.1]}
    else:
        model = lgb.LGBMClassifier(scale_pos_weight=spw, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, verbose=-1)
        param_grid = {"n_estimators": [100,200,300], "max_depth": [3,5,7], "learning_rate": [0.05,0.08,0.1]}
    print(f"  [tune] Searching {model_name} params: {3*3*3} combos x {cv_folds}-fold CV ...")
    grid = GridSearchCV(model, param_grid, cv=skf, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X, y)
    print(f"  [tune] Best F1 = {grid.best_score_:.4f}")
    print(f"  [tune] Best params = {grid.best_params_}")
    return grid.best_params_, grid.best_score_


def train_cv(X, y, model_name="xgboost", n_folds=5, verbose=True, best_params=None):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scaler = StandardScaler()
    fold_models, fold_metrics = [], []
    oof_proba = np.zeros(len(y))
    if verbose:
        print(f"\n=== {n_folds}-fold CV: {model_name} ===")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        if model_name == "xgboost":
            model = build_xgboost(y_tr, **(best_params or {}))
        else:
            model = build_lightgbm(y_tr, **(best_params or {}))
        model.fit(X_tr_s, y_tr)
        y_pred, y_proba = model.predict(X_val_s), model.predict_proba(X_val_s)[:, 1]
        rec = recall_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        ap, roc = average_precision_score(y_val, y_proba), roc_auc_score(y_val, y_proba)
        fold_metrics.append({"recall": rec, "precision": prec, "f1": f1, "avg_precision": ap, "roc_auc": roc})
        fold_models.append(model)
        oof_proba[val_idx] = y_proba
        if verbose:
            print(f"  Fold {fold+1}: recall={rec:.4f}  prec={prec:.4f}  F1={f1:.4f}  PR-AUC={ap:.4f}  ROC-AUC={roc:.4f}")
    best_idx = int(np.argmax([m["recall"] for m in fold_metrics]))
    if verbose:
        best = fold_metrics[best_idx]
        print(f"\n>>> Best fold (recall={best['recall']:.4f}): fold {best_idx+1}")
    best_threshold = _find_best_threshold(y, oof_proba)
    if verbose:
        print(f"  >> Best threshold (from OOF): {best_threshold:.2f}")
    X_scaled = scaler.fit_transform(X)
    if model_name == "xgboost":
        final_model = build_xgboost(y, **(best_params or {}))
    else:
        final_model = build_lightgbm(y, **(best_params or {}))
    final_model.fit(X_scaled, y)
    return {"model_name": model_name, "final_model": final_model, "scaler": scaler,
        "fold_metrics": fold_metrics, "fold_models": fold_models,
        "best_model_idx": best_idx, "best_threshold": best_threshold}


def evaluate(model, scaler, X_test, y_test, feature_names=None, threshold=0.5):
    X_test_s = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    print("\n" + "="*60 + "\nCLASSIFICATION REPORT\n" + "="*60)
    print(classification_report(y_test, y_pred, target_names=["Normal","Failure"], zero_division=0, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}")
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ap, roc = average_precision_score(y_test, y_proba), roc_auc_score(y_test, y_proba)
    print(f"\n  Threshold: {threshold:.2f}\n  Recall: {rec:.4f}\n  Precision: {prec:.4f}\n  F1: {f1:.4f}\n  PR-AUC: {ap:.4f}\n  ROC-AUC: {roc:.4f}")
    return {"recall": rec, "precision": prec, "f1": f1, "avg_precision": ap,
        "roc_auc": roc, "confusion_matrix": cm.tolist(), "y_pred": y_pred, "y_proba": y_proba}


def train_and_save(df, model_name="xgboost", tune=True):
    X, y, feature_cols = prepare_features(df)
    print(f"Feature matrix: {X.shape}  |  Positive ratio: {y.mean():.4f}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    best_params = {}
    if tune:
        print(f"\n===== Auto Tuning ({model_name}) =====")
        best_params, _ = tune_hyperparameters(X_train, y_train, model_name)
    cv_result = train_cv(X_train, y_train, model_name=model_name, best_params=best_params, verbose=True)
    print(f"\n=== Held-out test set evaluation ({model_name}) ===")
    metrics = evaluate(cv_result["final_model"], cv_result["scaler"], X_test, y_test, feature_cols,
        threshold=cv_result.get("best_threshold", 0.5))
    model_dir, model_path = MODEL_DIR, MODEL_DIR / f"best_model_{model_name}.joblib"
    scaler_path = MODEL_DIR / f"scaler_{model_name}.joblib"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cv_result["final_model"], model_path)
    joblib.dump(cv_result["scaler"], scaler_path)
    print(f"\nModel saved -> {model_path}\nScaler saved -> {scaler_path}")
    cv_result["test_metrics"] = metrics
    cv_result["feature_cols"] = feature_cols
    return cv_result





def train_multilabel(df):
    X, _, _ = prepare_features(df)
    print('\n' + '=' * 60)
    print('Multi-label fault type training')
    print('=' * 60)

    results = {}
    for target in MULTILABEL_TARGETS:
        y = df[target].values.astype(int)
        pos = y.sum()
        print(f'--- {target}: {pos} pos / {len(y)} total ({100*pos/len(y):.2f}%) ---')

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)

        best_f1, best_name, best_mod, best_scl = -1.0, '', None, None
        for mn in ['xgboost', 'lightgbm']:
            print(f'  [{mn}] Tuning...')
            params, _ = tune_hyperparameters(X_tr, y_tr, mn)
            cv_res = train_cv(X_tr, y_tr, model_name=mn, best_params=params, verbose=True)
            X_te_s = cv_res['scaler'].transform(X_te)
            y_p = cv_res['final_model'].predict(X_te_s)
            tf1 = f1_score(y_te, y_p, zero_division=0)
            trec = recall_score(y_te, y_p, zero_division=0)
            tpr = precision_score(y_te, y_p, zero_division=0)
            print(f'  [{mn}] Test: F1={tf1:.4f}  recall={trec:.4f}  prec={tpr:.4f}')
            if tf1 > best_f1:
                best_f1, best_name = tf1, mn
                best_mod = cv_res['final_model']
                best_scl = cv_res['scaler']

        target_lower = target.lower()
        mp = MODEL_DIR / f'best_model_{target_lower}.joblib'
        sp = MODEL_DIR / f'scaler_{target_lower}.joblib'
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_mod, mp)
        joblib.dump(best_scl, sp)
        print(f'  >>> Best: {best_name} (F1={best_f1:.4f}) -> {mp.name}')
        X_te_s = best_scl.transform(X_te)
        y_pf = best_mod.predict(X_te_s)
        cm = confusion_matrix(y_te, y_pf)
        print(f'  Confusion matrix:\n{cm}')
        results[target] = {'model_name': best_name, 'test_f1': best_f1}

    print('\n' + '=' * 60)
    print('Multi-label training complete!')
    for tgt, r in results.items():
        print(f'  {tgt}: {r["model_name"]} (F1={r["test_f1"]:.4f})')
    return results


if __name__ == "__main__":
    from features import load_data, PhysicsFeatureEngineer
    _DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    path = str(_DATA_DIR / "ai4i2020.csv")
    print(f"Loading data from {path} ...")
    raw = load_data(path)
    eng = PhysicsFeatureEngineer(raw)
    df = eng.transform()
    print(f"Data shape: {df.shape}  after engineering")
    best_result, best_name, best_recall = None, "", 0.0
    for model_name in ["xgboost", "lightgbm"]:
        result = train_and_save(df, model_name=model_name, tune=True)
        rec = result["test_metrics"]["recall"]
        if rec > best_recall:
            best_recall, best_result, best_name = rec, result, model_name
    print(f"\n{"="*50}\n>>> Best model: {best_name}  (test recall = {best_recall:.4f})")
    print(f"Feature cols: {best_result['feature_cols']}")
    # Multi-label fault type training
    print("\n===== Multi-label fault type training =====")
    train_multilabel(df)
