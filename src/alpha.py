"""Synthetic alpha construction for project section 2.4."""

import numpy as np
import pandas as pd

from src.config import AlphaConfig
from src.interfaces import validate_market_data


def _sorted_market_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().sort_values(["ticker", "date", "timestamp"]).reset_index(drop=True)


def _clip_alpha(series: pd.Series, config: AlphaConfig) -> pd.Series:
    if config.alpha_clip is None:
        return series
    return series.clip(-config.alpha_clip, config.alpha_clip)


def compute_intraday_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute one-step intraday returns by ticker and date if needed.

    If ret_1m already exists, only missing entries are filled where possible.
    The first observation for each ticker-date has return 0.0.
    """

    validate_market_data(df)
    out = _sorted_market_data(df)
    computed = out.groupby(["ticker", "date"], sort=False)["mid"].pct_change().fillna(0.0)
    if "ret_1m" in out.columns:
        out["ret_1m"] = out["ret_1m"].fillna(computed).fillna(0.0)
    else:
        out["ret_1m"] = computed
    return out


def build_intraday_lookahead_alpha(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Create synthetic look-ahead intraday alpha from future returns.

    For each ticker-date, alpha uses mid[t+h] / mid[t] - 1. This intentionally
    uses future information because the coursework asks for synthetic
    look-ahead alphas; it should not be interpreted as a live-tradable signal.
    """

    if config.intraday_horizon_steps < 1:
        raise ValueError("intraday_horizon_steps must be at least 1")

    out = compute_intraday_returns(df)
    h = config.intraday_horizon_steps
    grouped = out.groupby(["ticker", "date"], sort=False)
    future_mid = grouped["mid"].shift(-h)
    future_return = future_mid / out["mid"] - 1.0

    # The decay is a simple horizon-level dampener: longer look-ahead horizons
    # can be made less aggressive without introducing extra state.
    decay_factor = float(np.exp(-config.intraday_decay * h))
    out["alpha_intraday"] = config.intraday_strength * decay_factor * future_return
    out["alpha_intraday"] = _clip_alpha(out["alpha_intraday"].fillna(0.0), config)
    return out


def build_overnight_alpha(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Create synthetic overnight alpha by ticker and date.

    Convention: for each ticker-day, compute next day open / current day close
    - 1 and broadcast that value across the current day. This keeps the alpha
    transparent for coursework experiments while avoiding cross-stock mixing.
    """

    validate_market_data(df)
    out = _sorted_market_data(df)
    day_prices = (
        out.groupby(["ticker", "date"], sort=False)
        .agg(open_mid=("mid", "first"), close_mid=("mid", "last"))
        .reset_index()
        .sort_values(["ticker", "date"])
    )
    day_prices["next_open_mid"] = day_prices.groupby("ticker", sort=False)["open_mid"].shift(-1)
    day_prices["overnight_return"] = day_prices["next_open_mid"] / day_prices["close_mid"] - 1.0
    out = out.merge(
        day_prices[["ticker", "date", "overnight_return"]],
        on=["ticker", "date"],
        how="left",
    )
    out["alpha_overnight"] = config.overnight_strength * out["overnight_return"]
    out["alpha_overnight"] = _clip_alpha(out["alpha_overnight"].fillna(0.0), config)
    return out.drop(columns=["overnight_return"])


def build_combined_alpha(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Combine intraday and overnight synthetic alphas into alpha_combined."""

    out = df.copy()
    if "alpha_intraday" not in out.columns:
        out = build_intraday_lookahead_alpha(out, config)
    if "alpha_overnight" not in out.columns:
        out = build_overnight_alpha(out, config)
    out["alpha_combined"] = (
        config.combined_intraday_weight * out["alpha_intraday"]
        + config.combined_overnight_weight * out["alpha_overnight"]
    )
    out["alpha_combined"] = _clip_alpha(out["alpha_combined"].fillna(0.0), config)
    return out


def build_all_alphas(df: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Run the full synthetic-alpha pipeline.

    Returns data with ret_1m, alpha_intraday, alpha_overnight, and
    alpha_combined columns.
    """

    out = build_intraday_lookahead_alpha(df, config)
    out = build_overnight_alpha(out, config)
    out = build_combined_alpha(out, config)
    for col in ["alpha_intraday", "alpha_overnight", "alpha_combined"]:
        out[col] = out[col].fillna(0.0)
    return out

