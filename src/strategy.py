"""Simplified impact-aware strategy logic for project section 2.5."""

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.config import StrategyConfig
from src.interfaces import validate_alpha_data, validate_impact_params

DEFAULT_LAMBDA = 1e-6
DEFAULT_PARAMS = {"lambda": 1e-6, "rho": 0.1}


def get_model_params_for_ticker(
    impact_params: Mapping[str, Any], model_name: str, ticker: str
) -> dict[str, float]:
    """Return ticker params, falling back to "__universal__" then safe defaults."""

    try:
        validate_impact_params(impact_params)
    except ValueError:
        return DEFAULT_PARAMS.copy()
    if model_name not in impact_params:
        return DEFAULT_PARAMS.copy()
    model_params = impact_params[model_name]
    params = model_params.get(ticker, model_params.get("__universal__"))
    if params is None:
        return DEFAULT_PARAMS.copy()
    return dict(params)


def estimate_linear_impact_cost(trade: float, params: Mapping[str, float]) -> float:
    """Estimate a simple quadratic temporary impact cost lambda * trade^2."""

    lam = float(params.get("lambda", DEFAULT_LAMBDA))
    return lam * float(trade) ** 2


def compute_target_position(
    alpha: float, params: Mapping[str, float], config: StrategyConfig
) -> float:
    """Compute a clipped impact-aware target position from an alpha value."""

    lam = float(params.get("lambda", DEFAULT_LAMBDA))
    denominator = config.risk_aversion + lam + config.position_penalty
    if denominator <= 0:
        raise ValueError("risk_aversion + lambda + position_penalty must be positive")
    target = config.target_position_scale * float(alpha) / denominator
    return float(np.clip(target, -config.max_position, config.max_position))


def generate_trades_for_alpha(
    df: pd.DataFrame,
    impact_params: Mapping[str, Any],
    model_name: str,
    alpha_col: str,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Generate target positions and trades for a chosen alpha column.

    The implementation is an approximate fallback strategy. TODO: replace the
    execution path with the teammate's final backtest engine once available.
    """

    validate_alpha_data(df, alpha_col)
    validate_impact_params(impact_params)
    data = df.copy().sort_values(["ticker", "date", "timestamp"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for (ticker, date), group in data.groupby(["ticker", "date"], sort=False):
        current_position = 0.0
        params = get_model_params_for_ticker(impact_params, model_name, str(ticker))
        last_index = group.index[-1]
        for idx, row in group.iterrows():
            if config.liquidation_at_close and idx == last_index:
                target_position = 0.0
            else:
                target_position = compute_target_position(row[alpha_col], params, config)
            raw_trade = target_position - current_position
            if config.max_trade_size is not None:
                raw_trade = float(np.clip(raw_trade, -config.max_trade_size, config.max_trade_size))
            new_position = current_position + raw_trade
            impact_cost = estimate_linear_impact_cost(raw_trade, params)
            if config.trade_penalty:
                impact_cost += config.trade_penalty * abs(raw_trade)
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "date": row["date"],
                    "ticker": ticker,
                    "mid": float(row["mid"]),
                    "alpha": float(row[alpha_col]) if pd.notna(row[alpha_col]) else 0.0,
                    "target_position": float(target_position),
                    "position": float(new_position),
                    "trade": float(raw_trade),
                    "impact_cost": float(impact_cost),
                    "model_name": model_name,
                }
            )
            current_position = new_position

    return pd.DataFrame(rows)


def simulate_strategy_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute approximate mark-to-market PnL for generated trades."""

    required = {"timestamp", "date", "ticker", "mid", "position", "trade", "impact_cost"}
    missing = required.difference(trades_df.columns)
    if missing:
        raise ValueError(f"trades_df is missing required columns: {sorted(missing)}")

    out = trades_df.copy().sort_values(["ticker", "date", "timestamp"]).reset_index(drop=True)
    grouped = out.groupby(["ticker", "date"], sort=False)
    out["price_change"] = grouped["mid"].diff().fillna(0.0)
    out["previous_position"] = grouped["position"].shift(1).fillna(0.0)
    out["trading_cashflow"] = -out["trade"] * out["mid"]
    out["inventory_pnl"] = out["previous_position"] * out["price_change"]
    # TODO: replace this approximate PnL calculation with teammate's
    # Waelbroeck simulator once available.
    out["pnl"] = out["inventory_pnl"] - out["impact_cost"]
    out["cumulative_pnl"] = out["pnl"].cumsum()
    return out


def run_strategy(
    df: pd.DataFrame,
    impact_params: Mapping[str, Any],
    model_name: str,
    alpha_col: str,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Generate trades and approximate PnL for one alpha/model pair."""

    trades = generate_trades_for_alpha(df, impact_params, model_name, alpha_col, config)
    return simulate_strategy_pnl(trades)
