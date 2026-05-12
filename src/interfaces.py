"""Shared input validation and schema helpers."""

from typing import Any, Mapping

import pandas as pd


REQUIRED_MARKET_COLUMNS = {"timestamp", "date", "ticker", "mid", "volume"}
REQUIRED_ALPHA_BASE_COLUMNS = {"timestamp", "date", "ticker", "mid"}


def validate_market_data(df: pd.DataFrame) -> None:
    """Validate the expected market-data schema.

    Required columns are timestamp, date, ticker, mid, and volume. Optional
    rolling-window compatibility columns, such as window_id, are allowed.
    """

    missing = REQUIRED_MARKET_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"market_data is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("market_data must not be empty")
    if df["mid"].isna().any():
        raise ValueError("market_data column 'mid' contains missing values")
    if (df["mid"] <= 0).any():
        raise ValueError("market_data column 'mid' must be strictly positive")


def validate_alpha_data(df: pd.DataFrame, alpha_col: str) -> None:
    """Validate data containing an alpha column usable by the strategy."""

    missing = REQUIRED_ALPHA_BASE_COLUMNS.union({alpha_col}).difference(df.columns)
    if missing:
        raise ValueError(f"alpha data is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("alpha data must not be empty")


def validate_impact_params(impact_params: Mapping[str, Any]) -> None:
    """Validate the expected impact-parameter dictionary format.

    Expected format:
        {"OW": {"AAPL": {"lambda": 1e-6, "rho": 0.1}}}

    A "__universal__" ticker entry is supported as fallback.
    """

    if not isinstance(impact_params, Mapping) or not impact_params:
        raise ValueError("impact_params must be a non-empty mapping")
    for model_name, model_params in impact_params.items():
        if not isinstance(model_name, str):
            raise ValueError("impact model names must be strings")
        if not isinstance(model_params, Mapping) or not model_params:
            raise ValueError(f"impact_params[{model_name!r}] must be a non-empty mapping")
        for ticker, params in model_params.items():
            if not isinstance(ticker, str):
                raise ValueError(f"ticker key under model {model_name!r} must be a string")
            if not isinstance(params, Mapping):
                raise ValueError(f"params for {model_name!r}/{ticker!r} must be a mapping")
            if "lambda" in params and params["lambda"] < 0:
                raise ValueError(f"lambda for {model_name!r}/{ticker!r} must be non-negative")

