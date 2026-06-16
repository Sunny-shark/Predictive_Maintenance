"""
features.py — Physics-inspired feature engineering module
========================================================
PhysicsFeatureEngineer class for computing multi-modal physical features
from CNC machine sensor data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)



_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)


# ──────────────────────────────────────────────────────────────
#  PhysicsFeatureEngineer: three core composite physical features
# ──────────────────────────────────────────────────────────────

class PhysicsFeatureEngineer:
    """Compute physics-based composite features from raw CNC sensor data.

    The three core features are derived from first-principles physics:

      P_m  =  Torque x (RPM x 2pi/60)          [W]
      dT   =  Process temperature - Air temp    [K]
      I_os =  Tool wear x Torque                [min.Nm]
    """

    # Column name constants (UCI AI4I 2020 schema)
    _COL_TORQUE   = "Torque [Nm]"
    _COL_RPM      = "Rotational speed [rpm]"
    _COL_TEMP_PROC= "Process temperature [K]"
    _COL_TEMP_AIR = "Air temperature [K]"
    _COL_WEAR     = "Tool wear [min]"

    def __init__(self, df: pd.DataFrame):
        """Initialise with a copy of the raw sensor DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self._raw = df.copy()
        # 友好地检查是否缺少必要的列
        required = [
            self._COL_TORQUE,
            self._COL_RPM,
            self._COL_TEMP_PROC,
            self._COL_TEMP_AIR,
            self._COL_WEAR,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame 缺少 {len(missing)} 个必要列: {missing}.\n"
                f"现有列名: {list(df.columns)}"
            )

    # ── Public feature methods ──────────────────────────────

    def compute_mechanical_power(self) -> pd.Series:
        """Mechanical cutting power P_m (W).

        P_m = Torque [Nm] * (Rotational speed [rpm] * 2pi / 60)
        """
        torque = self._raw[self._COL_TORQUE]
        rpm    = self._raw[self._COL_RPM]
        return torque * (rpm * 2.0 * np.pi / 60.0)

    def compute_thermal_dissipation(self) -> pd.Series:
        """Thermal dissipation temperature difference dT (K).

        dT = Process temperature - Air temperature
        """
        return (self._raw[self._COL_TEMP_PROC]
                - self._raw[self._COL_TEMP_AIR])

    def compute_overstrain_index(self) -> pd.Series:
        """Overstrain wear index I_overstrain (min*Nm).

        I_overstrain = Tool wear [min] * Torque [Nm]
        """
        return self._raw[self._COL_WEAR] * self._raw[self._COL_TORQUE]

    def compute_all(self) -> pd.DataFrame:
        """Return a DataFrame with the three physical features appended.

        Returns
        -------
        pd.DataFrame with columns:
            'P_m [W]', 'Delta_T [K]', 'I_overstrain [min*Nm]'
        """
        return pd.DataFrame({
            "P_m [W]":              self.compute_mechanical_power(),
            "Delta_T [K]":          self.compute_thermal_dissipation(),
            "I_overstrain [min*Nm]": self.compute_overstrain_index(),
        })

    def transform(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Append physics features to a copy of the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame, optional
            If None, transforms the DataFrame passed at construction.

        Returns
        -------
        pd.DataFrame with original columns plus the three physics columns.
        """
        source = df.copy() if df is not None else self._raw
        # 直接计算特征，避免创建临时 PhysicsFeatureEngineer 对象
        torque = source[self._COL_TORQUE]
        rpm    = source[self._COL_RPM]
        features = pd.DataFrame({
            "P_m [W]":              torque * (rpm * 2.0 * np.pi / 60.0),
            "Delta_T [K]":          source[self._COL_TEMP_PROC] - source[self._COL_TEMP_AIR],
            "I_overstrain [min*Nm]": source[self._COL_WEAR] * torque,
        })
        result = pd.concat([source, features], axis=1)
        return result


# ──────────────────────────────────────────────────────────────
#  Data cleaning helper
# ──────────────────────────────────────────────────────────────

def clean_ai4i2020(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning for the UCI AI4I 2020 CNC dataset.

    - Drop rows with missing values
    - Ensure physical ranges are physically plausible
    - Cast categorical columns to the correct types
    - Flag statistical outliers via IQR (print warning, does NOT delete)
    """
    result = df.copy()

    # Drop duplicate UDI rows
    if "UDI" in result.columns:
        result = result.drop_duplicates(subset=["UDI"])

    # Drop rows with any NaN
    before = len(result)
    result = result.dropna()
    if len(result) < before:
        print(f"[clean] Removed {before - len(result)} rows with NaN.")

    # Constrain physically impossible values
    mask = (
        (result["Air temperature [K]"] >= 290.0)
        & (result["Air temperature [K]"] <= 310.0)
        & (result["Process temperature [K]"] >= result["Air temperature [K]"])
        & (result["Process temperature [K]"] <= 330.0)
        & (result["Rotational speed [rpm]"] >= 500)
        & (result["Rotational speed [rpm]"] <= 5000)
        & (result["Torque [Nm]"] >= 0.0)
        & (result["Torque [Nm]"] <= 200.0)
        & (result["Tool wear [min]"] >= 0)
    )
    result = result[mask]

    # Type column as categorical
    if "Type" in result.columns:
        result["Type"] = result["Type"].astype("category")

    # ── 统计异常值检测（IQR 方法） ──
    # 该数据集物理范围检查过于宽松，实际异常藏在统计异常值中。
    # 此处只标记不删除，因为异常值可能是故障前兆信号。
    _IQR_COLS = {
        "Rotational speed [rpm]": "转速",
        "Torque [Nm]": "扭矩",
    }
    outlier_flags = []
    for col, label in _IQR_COLS.items():
        if col in result.columns:
            Q1 = result[col].quantile(0.25)
            Q3 = result[col].quantile(0.75)
            IQR = Q3 - Q1
            lo = Q1 - 1.5 * IQR
            hi = Q3 + 1.5 * IQR
            is_out = (result[col] < lo) | (result[col] > hi)
            outlier_flags.append(is_out)
            n_out = is_out.sum()
            if n_out > 0:
                print(f"[clean] {label} ({col}) IQR 异常值: {n_out} 行 "
                      f"(范围 {lo:.1f}~{hi:.1f})")
    if outlier_flags:
        result["IQR_outlier"] = pd.concat(outlier_flags, axis=1).any(axis=1)
    else:
        result["IQR_outlier"] = False

    return result.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
#  Dataset loader
# ──────────────────────────────────────────────────────────────

def load_data(path: str, clean: bool = True) -> pd.DataFrame:
    """Load and optionally clean the UCI AI4I 2020 dataset."""
    df = pd.read_csv(path)
    if clean:
        df = clean_ai4i2020(df)
    return df


# ──────────────────────────────────────────────────────────────
#  Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = os.path.join(_DATA_DIR, "ai4i2020.csv")
    try:
        df = load_data(path)
        eng = PhysicsFeatureEngineer(df)
        feats = eng.transform()
        print(f"Physics features added. Shape: {feats.shape}")
        print(f"P_m mean:     {feats['P_m [W]'].mean():.1f} W")
        print(f"Delta_T mean: {feats['Delta_T [K]'].mean():.3f} K")
        print(f"I_os mean:    {feats['I_overstrain [min*Nm]'].mean():.1f}")

        # 统计 IQR 异常标记情况
        if "IQR_outlier" in feats.columns:
            n_flag = feats["IQR_outlier"].sum()
            print(f"IQR 标记异常行: {n_flag} / {len(feats)} ({100*n_flag/len(feats):.1f}%)")
    except FileNotFoundError:
        print("Dataset not found; run from the src/ directory.")
        raise
