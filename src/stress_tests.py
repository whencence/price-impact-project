"""Stress testing for project section 2.7."""

from typing import Any, Mapping

import pandas as pd

from src.config import StrategyConfig, StressTestConfig
from src.strategy import (
    estimate_linear_impact_cost,
    generate_trades_for_alpha,
    get_model_params_for_ticker,
    run_strategy,
    simulate_strategy_pnl,
)


def apply_signal_delay(df: pd.DataFrame, alpha_col: str, delay_steps: int) -> pd.DataFrame:
    """Delay an alpha column within each ticker-date group."""

    if delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")
    if alpha_col not in df.columns:
        raise ValueError(f"alpha column {alpha_col!r} not found")
    out = df.copy().sort_values(["ticker", "date", "timestamp"]).reset_index(drop=True)
    delayed_col = f"{alpha_col}_delay_{delay_steps}"
    out[delayed_col] = (
        out.groupby(["ticker", "date"], sort=False)[alpha_col].shift(delay_steps).fillna(0.0)
    )
    return out


def run_signal_delay_stress(
    df: pd.DataFrame,
    impact_params: Mapping[str, Any],
    model_name: str,
    alpha_col: str,
    strategy_config: StrategyConfig,
    delay_steps: int,
) -> pd.DataFrame:
    """Run the strategy using a delayed signal."""

    delayed = apply_signal_delay(df, alpha_col, delay_steps)
    delayed_col = f"{alpha_col}_delay_{delay_steps}"
    return run_strategy(delayed, impact_params, model_name, delayed_col, strategy_config)


def run_wrong_model_stress(
    df: pd.DataFrame,
    impact_params: Mapping[str, Any],
    assumed_model_name: str,
    true_model_name: str,
    alpha_col: str,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    """Optimize trades under one model and realize costs under another."""

    assumed = generate_trades_for_alpha(
        df, impact_params, assumed_model_name, alpha_col, strategy_config
    )
    out = assumed.copy()
    out = out.rename(columns={"impact_cost": "assumed_impact_cost"})
    true_costs = []
    for _, row in out.iterrows():
        params = get_model_params_for_ticker(impact_params, true_model_name, str(row["ticker"]))
        cost = estimate_linear_impact_cost(float(row["trade"]), params)
        if strategy_config.trade_penalty:
            cost += strategy_config.trade_penalty * abs(float(row["trade"]))
        true_costs.append(cost)
    out["true_impact_cost"] = true_costs
    out["impact_cost"] = out["true_impact_cost"]
    out["model_name_assumed"] = assumed_model_name
    out["model_name_true"] = true_model_name
    out["model_name"] = f"assumed={assumed_model_name},true={true_model_name}"
    return simulate_strategy_pnl(out)


def apply_forced_liquidation(
    trades_df: pd.DataFrame,
    liquidation_time: str,
    impact_params: Mapping[str, Any],
    true_model_name: str,
) -> pd.DataFrame:
    """Force positions to zero from the first timestamp at or after a time.

    Convention: the liquidation trade is applied in one block at the trigger
    row. Subsequent rows for that ticker-date stay flat and do not re-enter.
    """

    required = {"timestamp", "date", "ticker", "mid", "position", "trade"}
    missing = required.difference(trades_df.columns)
    if missing:
        raise ValueError(f"trades_df is missing required columns: {sorted(missing)}")

    out = trades_df.copy().sort_values(["ticker", "date", "timestamp"]).reset_index(drop=True)
    out["forced_liquidation"] = False
    threshold = pd.to_datetime(liquidation_time).time()

    for (_, _), group in out.groupby(["ticker", "date"], sort=False):
        trigger = group[group["timestamp"].dt.time >= threshold]
        if trigger.empty:
            continue
        trigger_idx = trigger.index[0]
        pre_position = (
            out.at[trigger_idx - 1, "position"] if trigger_idx > group.index[0] else 0.0
        )
        liquidation_trade = -float(pre_position)
        ticker = str(out.at[trigger_idx, "ticker"])
        params = get_model_params_for_ticker(impact_params, true_model_name, ticker)
        out.at[trigger_idx, "trade"] = liquidation_trade
        out.at[trigger_idx, "position"] = 0.0
        out.at[trigger_idx, "target_position"] = 0.0
        out.at[trigger_idx, "impact_cost"] = estimate_linear_impact_cost(liquidation_trade, params)
        out.at[trigger_idx, "forced_liquidation"] = True
        after_idx = group.index[group.index > trigger_idx]
        out.loc[after_idx, ["trade", "position", "target_position", "impact_cost"]] = 0.0

    return simulate_strategy_pnl(out)


def run_forced_liquidation_stress(
    df: pd.DataFrame,
    impact_params: Mapping[str, Any],
    model_name: str,
    alpha_col: str,
    strategy_config: StrategyConfig,
    liquidation_time: str,
) -> pd.DataFrame:
    """Run baseline strategy and then apply a forced intraday liquidation."""

    baseline = run_strategy(df, impact_params, model_name, alpha_col, strategy_config)
    return apply_forced_liquidation(baseline, liquidation_time, impact_params, model_name)


def run_all_stress_tests(
    df: pd.DataFrame,
    impact_params: Mapping[str, Any],
    alpha_col: str,
    strategy_config: StrategyConfig,
    stress_config: StressTestConfig,
) -> dict[str, pd.DataFrame]:
    """Run baseline, delayed signal, wrong-model, and forced-liquidation tests."""

    baseline_model = stress_config.wrong_model_assumed
    results = {
        "baseline": run_strategy(df, impact_params, baseline_model, alpha_col, strategy_config),
        "delayed_signal": run_signal_delay_stress(
            df,
            impact_params,
            baseline_model,
            alpha_col,
            strategy_config,
            stress_config.signal_delay_steps,
        ),
        "wrong_model": run_wrong_model_stress(
            df,
            impact_params,
            stress_config.wrong_model_assumed,
            stress_config.wrong_model_true,
            alpha_col,
            strategy_config,
        ),
    }
    if stress_config.forced_liquidation_time is not None:
        results["forced_liquidation"] = run_forced_liquidation_stress(
            df,
            impact_params,
            baseline_model,
            alpha_col,
            strategy_config,
            stress_config.forced_liquidation_time,
        )
    return results
