"""Evaluate strategy trades with teammate fitted regression models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fitted_regression_features import add_ow_state_by_stock, compute_training_scales, prepare_model_frame
from src.fitted_regression_params import get_params_for_pair_stock, infer_feature_columns_from_teammate_params


def predict_ols_from_coef(df: pd.DataFrame, coef: dict[str, float], feature_cols: list[str]) -> np.ndarray:
    """Predict OLS regression from coefficient dict."""

    pred = np.full(len(df), float(coef.get("intercept", 0.0)), dtype=float)
    for col in feature_cols:
        pred += float(coef.get(col, 0.0)) * pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0).to_numpy()
    return pred


def add_strategy_trade_to_orderflow(
    market_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    signed_volume_col: str = "signed_volume",
    stock_col: str = "stock",
    datetime_col: str = "datetime",
    trading_date_col: str = "trading_date",
) -> pd.DataFrame:
    """Add strategy signed volume to market orderFlow by exact stock/date/time bin."""

    out = market_df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    trades = trades_df.copy()
    if datetime_col not in trades.columns and "timestamp" in trades.columns:
        trades[datetime_col] = trades["timestamp"]
    if trading_date_col not in trades.columns:
        trades[trading_date_col] = pd.to_datetime(trades[datetime_col]).dt.date.astype(str)
    trades[datetime_col] = pd.to_datetime(trades[datetime_col], errors="coerce")
    if signed_volume_col not in trades.columns and "trade" in trades.columns:
        signed_volume_col = "trade"
    agg = (
        trades.groupby([stock_col, trading_date_col, datetime_col], as_index=False)[signed_volume_col]
        .sum()
        .rename(columns={signed_volume_col: "strategy_signed_volume"})
    )
    out = out.merge(agg, on=[stock_col, trading_date_col, datetime_col], how="left")
    out["strategy_signed_volume"] = out["strategy_signed_volume"].fillna(0.0)
    out["orderFlow_market"] = pd.to_numeric(out["orderFlow"], errors="coerce").fillna(0.0)
    out["orderFlow_scenario"] = out["orderFlow_market"] + out["strategy_signed_volume"]
    return out


def prepare_scenario_model_frame(
    market_raw_df: pd.DataFrame,
    train_scales: dict[str, object],
    use_scenario_orderflow: bool,
    add_strategy_to_trade_feature: bool = False,
) -> pd.DataFrame:
    """Prepare baseline or scenario feature frame."""

    data = market_raw_df.copy()
    if use_scenario_orderflow:
        data["orderFlow"] = data["orderFlow_scenario"]
        if add_strategy_to_trade_feature:
            data["trade"] = pd.to_numeric(data["trade"], errors="coerce").fillna(0.0) + data.get("strategy_signed_volume", 0.0)
    return prepare_model_frame(data, train_scales)


def predict_teammate_model_for_pair(
    model_frame: pd.DataFrame,
    params_df: pd.DataFrame,
    pair_id: int,
    model_name: str,
    allow_missing: bool = True,
) -> pd.DataFrame:
    """Predict teammate fitted model per stock for a pair."""

    frame = model_frame.copy()
    if model_name == "OW_transient":
        half_lives = {}
        for stock in frame["stock"].astype(str).unique():
            params = get_params_for_pair_stock(params_df, pair_id, stock, model_name, allow_missing=True)
            if params is not None and np.isfinite(params["half_life_sec"]):
                half_lives[stock] = params["half_life_sec"]
        frame = add_ow_state_by_stock(frame, half_lives)
    pred = np.full(len(frame), np.nan)
    available = np.zeros(len(frame), dtype=bool)
    feature_cols_default = infer_feature_columns_from_teammate_params(params_df, model_name)
    for stock, idx in frame.groupby("stock", sort=False).groups.items():
        params = get_params_for_pair_stock(params_df, pair_id, str(stock), model_name, allow_missing=allow_missing)
        if params is None:
            continue
        loc = list(idx)
        features = params.get("feature_cols") or feature_cols_default
        pred[loc] = predict_ols_from_coef(frame.loc[loc], params["coef"], list(features))
        available[loc] = True
    frame["pred_ret_bps"] = pred
    frame["model_name"] = model_name
    frame["pair_id"] = pair_id
    frame["param_available"] = available
    return frame


def evaluate_marginal_impact_from_strategy_trades(
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    params_df: pd.DataFrame,
    pair_id: int,
    model_name: str = "reduced_form",
) -> pd.DataFrame:
    """Evaluate fixed strategy trades under teammate fitted regression model."""

    scales = compute_training_scales(train_raw_df)
    market = test_raw_df.copy()
    market["datetime"] = pd.to_datetime(market["datetime"], errors="coerce")
    base_frame = prepare_scenario_model_frame(market, scales, use_scenario_orderflow=False)
    base_pred = predict_teammate_model_for_pair(base_frame, params_df, pair_id, model_name, allow_missing=True)
    scenario_raw = add_strategy_trade_to_orderflow(market, trades_df)
    scenario_frame = prepare_scenario_model_frame(scenario_raw, scales, use_scenario_orderflow=True)
    scenario_pred = predict_teammate_model_for_pair(scenario_frame, params_df, pair_id, model_name, allow_missing=True)

    keys = ["stock", "trading_date", "datetime"]
    base = base_pred[keys + ["pred_ret_bps"]].rename(columns={"pred_ret_bps": "pred_ret_bps_market"})
    scen_cols = keys + [
        "mid", "orderFlow_market", "orderFlow_scenario", "strategy_signed_volume",
        "pred_ret_bps", "param_available",
    ]
    out = scenario_pred[scen_cols].rename(columns={"pred_ret_bps": "pred_ret_bps_with_strategy"})
    out = out.merge(base, on=keys, how="left")

    trade_cols = [
        "stock", "timestamp", "signed_volume", "position_before", "position_after", "delta_mid",
        "gross_pnl", "net_pnl", "cumulative_wealth",
        "liquidation_mode", "is_forced_liquidation_start", "is_liquidation_trade",
        "liquidation_trade", "forced_liquidation_trade", "actual_liquidation_trade",
        "residual_inventory_active", "residual_position", "participation_rate_liquidation",
    ]
    available_trade_cols = [c for c in trade_cols if c in trades_df.columns]
    trades = trades_df[available_trade_cols].copy()
    if "timestamp" in trades.columns:
        trades["datetime"] = pd.to_datetime(trades["timestamp"], errors="coerce")
    trades = trades.drop(columns=["timestamp"], errors="ignore")
    out = out.merge(trades, on=["stock", "datetime"], how="left")
    out["signed_volume"] = out["signed_volume"].fillna(0.0)
    out["position_before"] = out["position_before"].fillna(0.0)
    out["position_after"] = out["position_after"].fillna(0.0)
    out["delta_mid"] = out["delta_mid"].fillna(0.0)
    out["marginal_impact_bps"] = out["pred_ret_bps_with_strategy"] - out["pred_ret_bps_market"]
    out["marginal_impact_return"] = out["marginal_impact_bps"] / 10000.0
    out["marginal_impact_price"] = out["mid"] * out["marginal_impact_return"]
    out["fitted_impact_cost_signed"] = out["signed_volume"] * out["marginal_impact_price"]
    out["fitted_impact_cost_abs"] = out["fitted_impact_cost_signed"].abs()
    out["trade_direction"] = np.sign(out["signed_volume"])
    out["impact_direction"] = np.sign(out["marginal_impact_bps"])
    out["same_direction_trade_impact"] = (
        (out["trade_direction"] != 0)
        & (out["impact_direction"] != 0)
        & (out["trade_direction"] == out["impact_direction"])
    )
    out["cost_is_adverse"] = out["fitted_impact_cost_signed"] > 0
    out["gross_pnl"] = out["position_before"] * out["delta_mid"]
    out["net_pnl_fitted_model"] = out["gross_pnl"] - out["fitted_impact_cost_signed"]
    out["cumulative_wealth_fitted_model"] = out["net_pnl_fitted_model"].cumsum()
    out["model_name"] = model_name
    out["pair_id"] = pair_id
    return out


def evaluate_ow_transient_regression(*args, **kwargs) -> pd.DataFrame:
    """Evaluate teammate OW_transient regression impact."""

    kwargs["model_name"] = "OW_transient"
    return evaluate_marginal_impact_from_strategy_trades(*args, **kwargs)
