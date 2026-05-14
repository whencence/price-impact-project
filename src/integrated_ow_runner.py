"""Run my OW strategy on teammate rolling pairs."""

import pandas as pd

from src.fitted_regression_params import convert_teammate_half_life_to_my_half_life_minutes
from src.integration_config import IntegratedRunConfig
from src.ow_strategy import attach_scaling_factors, run_ow_strategy
from src.scaling_factors import compute_daily_stock_info_from_bins
from src.strategy_config import OWStrategyConfig


def build_pair_scaling_from_train_test(train_raw_df: pd.DataFrame, test_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build simple train-derived sigma/ADV scaling for the pair test month."""

    train_for_scaling = train_raw_df.copy()
    if "datetime" in train_for_scaling.columns:
        train_for_scaling["date"] = pd.to_datetime(train_for_scaling["datetime"], errors="coerce").dt.strftime("%Y%m%d")
    elif "timestamp" in train_for_scaling.columns:
        train_for_scaling["date"] = pd.to_datetime(train_for_scaling["timestamp"], errors="coerce").dt.strftime("%Y%m%d")
    elif "trading_date" in train_for_scaling.columns:
        train_for_scaling["date"] = pd.to_datetime(train_for_scaling["trading_date"], errors="coerce").dt.strftime("%Y%m%d")
    daily = compute_daily_stock_info_from_bins(train_for_scaling)
    latest = (
        daily.groupby("stock", as_index=False)
        .agg(trailing_px_vol=("px_vol", "mean"), trailing_ADV=("daily_volume", "mean"))
    )
    dates = pd.to_datetime(test_raw_df["trading_date"]).drop_duplicates().sort_values()
    rows = []
    for date in dates:
        temp = latest.copy()
        temp["date"] = date
        rows.append(temp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stock", "date", "trailing_px_vol", "trailing_ADV"])


def build_per_stock_ow_params(
    teammate_ow_params_df: pd.DataFrame,
    pair_id: int,
    use_x_flow_lambda_proxy: bool = False,
) -> dict[str, dict[str, float]]:
    """Convert teammate OW time constants into my strict half-life convention."""

    rows = teammate_ow_params_df[teammate_ow_params_df["pair_id"].astype(int).eq(int(pair_id))]
    params: dict[str, dict[str, float]] = {}
    for _, row in rows.iterrows():
        stock = str(row["stock"])
        half_life = convert_teammate_half_life_to_my_half_life_minutes(float(row["half_life_sec"]))
        entry = {"impact_half_life_minutes": half_life}
        if use_x_flow_lambda_proxy:
            entry["impact_lambda"] = max(abs(float(row.get("x_flow", 1.0))), 1e-12)
        params[stock] = entry
    return params


def make_integrated_ow_config(
    pair_row: pd.Series,
    teammate_ow_params_df: pd.DataFrame,
    run_config: IntegratedRunConfig,
    alpha_col: str = "alpha_for_strategy",
) -> OWStrategyConfig:
    """Build the OW config used by integrated rolling runs.

    Teammate's half_life_sec is an exponential time constant in his regression
    feature construction. When using it inside my structural OW mechanics, it is
    converted to a strict half-life in minutes.
    """

    per_stock = build_per_stock_ow_params(
        teammate_ow_params_df,
        int(pair_row["pair_id"]),
        use_x_flow_lambda_proxy=run_config.use_x_flow_lambda_proxy,
    )
    return OWStrategyConfig(
        alpha_col=alpha_col,
        scaling_factors_path=None,
        use_scaling_factors=True,
        impact_lambda=1.0,
        impact_half_life_minutes=5.0,
        per_stock_impact_params=per_stock,
        max_participation_rate_per_trade=run_config.max_participation_rate_per_trade,
        max_abs_position_adv_fraction=run_config.max_abs_position_adv_fraction,
        max_abs_trade_adv_fraction=run_config.max_abs_trade_adv_fraction,
        target_impact_scale=run_config.target_impact_scale,
        alpha_scale=run_config.alpha_scale,
        liquidate_at_close=True,
    )


def prepare_integrated_ow_strategy_input(
    pair_row: pd.Series,
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    teammate_ow_params_df: pd.DataFrame,
    run_config: IntegratedRunConfig,
    alpha_col: str = "alpha_for_strategy",
) -> tuple[pd.DataFrame, OWStrategyConfig]:
    """Attach train-derived scaling and return the integrated OW config."""

    scaling = build_pair_scaling_from_train_test(train_raw_df, test_raw_df)
    cfg = make_integrated_ow_config(pair_row, teammate_ow_params_df, run_config, alpha_col=alpha_col)
    strategy_input = attach_scaling_factors(alpha_df, scaling, cfg)
    return strategy_input, cfg


def run_my_ow_strategy_on_pair(
    pair_row: pd.Series,
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    stocks: list[str],
    alpha_df: pd.DataFrame,
    teammate_ow_params_df: pd.DataFrame,
    run_config: IntegratedRunConfig,
) -> pd.DataFrame:
    """Run my OW strategy on a rolling pair using train-derived scaling."""

    strategy_input, cfg = prepare_integrated_ow_strategy_input(
        pair_row,
        train_raw_df,
        test_raw_df,
        alpha_df,
        teammate_ow_params_df,
        run_config,
    )
    trades = run_ow_strategy(strategy_input, cfg)
    trades["pair_id"] = int(pair_row["pair_id"])
    trades["train_month"] = str(pair_row["train_month"])
    trades["test_month"] = str(pair_row["test_month"])
    trades["strategy_model"] = "my_OW_theoretical"
    trades["lambda_source"] = "x_flow_proxy" if run_config.use_x_flow_lambda_proxy else "placeholder"
    trades["half_life_source"] = "teammate_time_constant_converted"
    return trades
