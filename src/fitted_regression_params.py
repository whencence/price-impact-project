"""Load teammate fitted regression parameter files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.fitted_regression_features import META_COLUMNS, OW_TRANSIENT_FEATURES, infer_feature_columns_from_params


def _load_params(path: Path, expected_model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"pair_id", "train_month", "test_month", "model", "stock", "half_life_sec", "intercept"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"parameter file {path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(int)
    out["stock"] = out["stock"].astype(str)
    if expected_model:
        out = out[out["model"].astype(str).str.lower().eq(expected_model.lower())].copy()
    return out


def load_ow_transient_params(path: Path) -> pd.DataFrame:
    """Load OW_transient regression parameters."""

    return _load_params(path, "OW_transient")


def load_reduced_form_params(path: Path) -> pd.DataFrame:
    """Load reduced_form regression parameters."""

    return _load_params(path, "reduced_form")


def infer_feature_columns_from_teammate_params(params_df: pd.DataFrame, model_name: str) -> list[str]:
    """Infer feature columns by model."""

    if model_name == "OW_transient":
        return [col for col in OW_TRANSIENT_FEATURES if col in params_df.columns]
    return infer_feature_columns_from_params(params_df)


def get_params_for_pair_stock(
    params_df: pd.DataFrame,
    pair_id: int,
    stock: str,
    model_name: str,
    allow_missing: bool = False,
) -> dict[str, object] | None:
    """Return parameter dict for pair/stock/model."""

    rows = params_df[
        params_df["pair_id"].astype(int).eq(int(pair_id))
        & params_df["stock"].astype(str).eq(str(stock))
        & params_df["model"].astype(str).eq(model_name)
    ]
    if rows.empty:
        if allow_missing:
            return None
        raise KeyError(f"missing params for pair_id={pair_id}, stock={stock}, model={model_name}")
    row = rows.iloc[0]
    features = infer_feature_columns_from_teammate_params(params_df, model_name)
    coef = {"intercept": float(row.get("intercept", 0.0))}
    for col in features:
        value = row.get(col, np.nan)
        if pd.notna(value):
            coef[col] = float(value)
    return {
        "pair_id": int(row["pair_id"]),
        "stock": str(row["stock"]),
        "model": str(row["model"]),
        "half_life_sec": float(row["half_life_sec"]) if pd.notna(row["half_life_sec"]) else np.nan,
        "coef": coef,
        "feature_cols": [col for col in features if col in coef],
    }


def convert_teammate_half_life_to_my_half_life_minutes(half_life_sec: float) -> float:
    """Convert teammate exponential time constant tau to strict half-life minutes.

    Teammate features use exp(-dt / tau). My OW strategy uses
    exp(-ln(2) * dt / H). Equivalent H is tau * ln(2).
    """

    return float(half_life_sec) * float(np.log(2.0)) / 60.0
