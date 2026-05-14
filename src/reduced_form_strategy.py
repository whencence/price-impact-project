"""Theoretical reduced-form AFS-style strategy for section 2.5.

This module implements a flexible reduced-form strategy using the course
formulae, while keeping fitted parameters external so calibrated model outputs
can be plugged in later.

Important integration caveat:
Teammate's fitted ``reduced_form`` output is a per-stock/per-rolling-pair OLS
regression for within-bin ``ret_bps``. It is not the dynamic AFS strategy
implemented here. For teammate-model integration and wrong-model stress, use
``src.reduced_form_regression_evaluator`` instead of this module.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.reduced_form_config import ReducedFormStrategyConfig


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def validate_reduced_form_input(df: pd.DataFrame, config: ReducedFormStrategyConfig) -> None:
    """Validate the input required by the reduced-form strategy."""

    required = [
        config.date_col,
        config.time_col,
        config.timestamp_col,
        config.stock_col,
        config.price_col,
        config.alpha_col,
    ]
    if df.empty:
        raise ValueError("reduced-form strategy input must not be empty")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"reduced-form input missing required columns: {missing}")
    timestamp = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    if timestamp.isna().any():
        raise ValueError("timestamp column contains unparseable values")
    mid = pd.to_numeric(df[config.price_col], errors="coerce")
    alpha = pd.to_numeric(df[config.alpha_col], errors="coerce")
    if (mid > 0).mean() < 0.95:
        _warn("less than 95% of mid values are positive")
    if alpha.isna().any():
        _warn(f"{int(alpha.isna().sum())} alpha values are missing and will be filled with 0")
    if config.volume_col not in df.columns:
        _warn(f"{config.volume_col} is missing; merge market volume from binSamples before running")
    if "sigma" not in df.columns or "ADV" not in df.columns:
        _warn("scaling factors are not attached yet")


def prepare_reduced_form_data(df: pd.DataFrame, config: ReducedFormStrategyConfig) -> pd.DataFrame:
    """Prepare timestamps, alpha, mu, time gaps, and price changes."""

    validate_reduced_form_input(df, config)
    out = df.copy()
    out[config.timestamp_col] = pd.to_datetime(out[config.timestamp_col], errors="coerce")
    out[config.price_col] = pd.to_numeric(out[config.price_col], errors="coerce")
    out[config.alpha_col] = pd.to_numeric(out[config.alpha_col], errors="coerce").fillna(0.0)
    if config.volume_col in out.columns:
        out[config.volume_col] = pd.to_numeric(out[config.volume_col], errors="coerce").fillna(0.0)
    else:
        out[config.volume_col] = 0.0
    out = out.sort_values([config.stock_col, config.date_col, config.timestamp_col]).reset_index(drop=True)
    grouped = out.groupby([config.stock_col, config.date_col], sort=False)
    out["dt_minutes"] = grouped[config.timestamp_col].diff().dt.total_seconds().div(60.0).fillna(0.0)
    out.loc[out["dt_minutes"] <= 0, "dt_minutes"] = 0.0
    out["delta_mid"] = grouped[config.price_col].diff().fillna(0.0)
    out["alpha_t"] = out[config.alpha_col]

    if config.mu_method == "backward_alpha_derivative":
        alpha_prev = grouped["alpha_t"].shift(1)
        out["mu_t"] = 0.0
        positive_dt = out["dt_minutes"] > 0
        out.loc[positive_dt, "mu_t"] = (
            (out.loc[positive_dt, "alpha_t"] - alpha_prev.loc[positive_dt].fillna(out.loc[positive_dt, "alpha_t"]))
            / out.loc[positive_dt, "dt_minutes"]
        )
        out.loc[grouped.cumcount() == 0, "mu_t"] = 0.0
    elif config.mu_method == "alpha_over_horizon":
        out["mu_t"] = out["alpha_t"] / config.mu_horizon_minutes
    else:
        if config.mu_col not in out.columns:
            raise ValueError(f"mu_col not found: {config.mu_col}")
        out["mu_t"] = pd.to_numeric(out[config.mu_col], errors="coerce").fillna(0.0)

    if config.winsorize_mu_quantile is not None and len(out):
        bound = float(out["mu_t"].abs().quantile(config.winsorize_mu_quantile))
        if np.isfinite(bound) and bound > 0:
            out["mu_t"] = out["mu_t"].clip(-bound, bound)
    return out


def load_scaling_factors(config: ReducedFormStrategyConfig, project_root: Path | None = None) -> pd.DataFrame:
    """Load trailing sigma/ADV scaling factors."""

    if not config.use_scaling_factors:
        return pd.DataFrame(columns=["stock", "date", "trailing_px_vol", "trailing_ADV"])
    if config.scaling_factors_path is None:
        raise ValueError("scaling_factors_path must be set when use_scaling_factors=True")
    path = Path(config.scaling_factors_path)
    if not path.is_absolute():
        path = (project_root or Path.cwd()) / path
    if not path.exists():
        raise FileNotFoundError(f"scaling factors file not found: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def attach_scaling_factors(
    strategy_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    config: ReducedFormStrategyConfig,
) -> pd.DataFrame:
    """Attach trailing sigma and ADV by stock/date."""

    out = strategy_df.copy()
    out["_merge_date"] = pd.to_datetime(out[config.date_col], errors="coerce").dt.normalize()
    if config.use_scaling_factors:
        required = {"stock", "date", "trailing_px_vol", "trailing_ADV"}
        missing = required.difference(scaling_df.columns)
        if missing:
            raise ValueError(f"scaling_df missing required columns: {sorted(missing)}")
        scaling = scaling_df.copy()
        scaling["_merge_date"] = pd.to_datetime(scaling["date"], errors="coerce").dt.normalize()
        scaling = scaling.rename(
            columns={"stock": config.stock_col, "trailing_px_vol": "sigma", "trailing_ADV": "ADV"}
        )
        extras = [c for c in ["scaling_window_days", "has_full_scaling_window"] if c in scaling.columns]
        out = out.merge(
            scaling[[config.stock_col, "_merge_date", "sigma", "ADV"] + extras],
            on=[config.stock_col, "_merge_date"],
            how="left",
        )
    else:
        out["sigma"] = config.fallback_sigma
        out["ADV"] = config.fallback_ADV

    out["sigma"] = pd.to_numeric(out.get("sigma"), errors="coerce")
    out["ADV"] = pd.to_numeric(out.get("ADV"), errors="coerce")
    missing = ~(out["sigma"].notna() & out["ADV"].notna() & (out["sigma"] > 0) & (out["ADV"] > 0))
    if config.fallback_sigma is not None:
        out.loc[out["sigma"].isna() | (out["sigma"] <= 0), "sigma"] = config.fallback_sigma
    if config.fallback_ADV is not None:
        out.loc[out["ADV"].isna() | (out["ADV"] <= 0), "ADV"] = config.fallback_ADV
    out["has_scaling"] = out["sigma"].notna() & out["ADV"].notna() & (out["sigma"] > 0) & (out["ADV"] > 0)
    if missing.any():
        _warn(f"{missing.mean() * 100:.2f}% of rows missing trailing sigma/ADV before fallback")
    return out.drop(columns=["_merge_date"])


def merge_market_volume_from_bins(
    alpha_df: pd.DataFrame,
    bin_dir: Path,
    config: ReducedFormStrategyConfig,
) -> pd.DataFrame:
    """Merge trade/orderFlow from binSamples into alpha rows by stock/date/time."""

    if config.volume_col in alpha_df.columns and "orderFlow" in alpha_df.columns:
        return alpha_df.copy()
    bin_dir = Path(bin_dir)
    if "source_file" in alpha_df.columns:
        files = [bin_dir / name for name in sorted(alpha_df["source_file"].dropna().astype(str).unique())]
        files = [p for p in files if p.exists()]
    else:
        files = sorted(bin_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no binSample files available under {bin_dir}")
    frames = []
    usecols = ["date", "time", "stock", "trade", "orderFlow"]
    needed_dates = set(alpha_df[config.date_col].astype(str).unique())
    for path in files:
        frame = pd.read_csv(path, usecols=lambda col: col in usecols, low_memory=False)
        if needed_dates:
            frame = frame[frame["date"].astype(str).isin(needed_dates)].copy()
        if len(frame):
            frames.append(frame)
    if not frames:
        _warn("no matching binSample rows found for alpha dates; volume will be zero")
        out = alpha_df.copy()
        out[config.volume_col] = 0.0
        return out
    bins = pd.concat(frames, ignore_index=True)
    merge_cols = ["_merge_date", config.time_col, config.stock_col]
    out = alpha_df.copy()
    out["_merge_date"] = pd.to_datetime(out[config.date_col], errors="coerce").dt.normalize().astype(str)
    bins["_merge_date"] = pd.to_datetime(bins[config.date_col].astype(str), format="%Y%m%d", errors="coerce")
    bins["_merge_date"] = bins["_merge_date"].fillna(pd.to_datetime(bins[config.date_col], errors="coerce"))
    bins["_merge_date"] = bins["_merge_date"].dt.normalize().astype(str)
    for col in [config.time_col, config.stock_col]:
        out[col] = out[col].astype(str)
        bins[col] = bins[col].astype(str)
    bins = bins.drop_duplicates(merge_cols)
    merged = out.merge(bins, on=merge_cols, how="left", suffixes=("", "_bin"))
    return merged.drop(columns=["_merge_date"])


def compute_market_volume_state_clock_time(
    df: pd.DataFrame,
    config: ReducedFormStrategyConfig,
) -> pd.DataFrame:
    """Compute rolling clock-time local market volume state v_t."""

    out = df.copy()
    out[config.volume_col] = pd.to_numeric(out[config.volume_col], errors="coerce").fillna(0.0)
    out["market_signed_volume"] = out[config.volume_col]
    out["market_abs_volume"] = out[config.volume_col].abs()
    window = f"{config.volume_window_minutes}min"
    pieces = []
    for _, group in out.groupby([config.stock_col, config.date_col], sort=False):
        g = group.sort_values(config.timestamp_col).copy()
        rolling = (
            g.set_index(config.timestamp_col)["market_abs_volume"]
            .rolling(window=window, min_periods=1)
            .sum()
            .to_numpy()
        )
        g["local_volume_state_v"] = np.maximum(rolling, config.min_volume_state)
        g["local_volume_state_v_lag"] = g["local_volume_state_v"].shift(1).fillna(config.min_volume_state)
        g["delta_abs_volume"] = g["market_abs_volume"].diff().fillna(0.0)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(
        [config.stock_col, config.date_col, config.timestamp_col]
    ).reset_index(drop=True)


def compute_liquidity_signals(df: pd.DataFrame, config: ReducedFormStrategyConfig) -> pd.DataFrame:
    """Compute lambda_t and gamma_prime_t liquidity signals."""

    out = df.copy()
    v = pd.to_numeric(out["local_volume_state_v"], errors="coerce").clip(lower=config.min_volume_state)
    out["lambda_t"] = config.lambda_base / np.sqrt(v)
    out["gamma_prime_t"] = 0.0
    positive_dt = out["dt_minutes"] > 0
    denom = 2.0 * out.loc[positive_dt, "dt_minutes"] * np.power(v.loc[positive_dt], 1.5)
    out.loc[positive_dt, "gamma_prime_t"] = config.lambda_base * out.loc[positive_dt, "delta_abs_volume"] / denom
    out["gamma_prime_t"] = out["gamma_prime_t"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def compute_reduced_form_target_impact(
    df: pd.DataFrame,
    config: ReducedFormStrategyConfig,
) -> pd.DataFrame:
    """Compute reduced-form target impact with full or heuristic formula."""

    out = df.copy()
    beta = config.effective_beta
    out["beta"] = beta
    out["target_formula_fallback"] = False
    out["target_formula_used"] = "slow_moving_liquidity"
    heuristic = 0.5 * out["alpha_t"] - out["mu_t"] / beta
    out["target_impact_heuristic"] = heuristic

    gamma = out["gamma_prime_t"]
    denominator = 2.0 * beta + gamma
    valid_full = denominator > 1e-12
    full = ((beta + gamma) / denominator) * out["alpha_t"] - out["mu_t"] / denominator
    out["target_impact_full_gamma"] = full.where(valid_full, heuristic)
    if config.use_full_gamma_formula:
        out["target_impact"] = out["target_impact_full_gamma"]
        out["target_formula_used"] = "full_gamma"
        out.loc[~valid_full, "target_formula_fallback"] = True
    else:
        out["target_impact"] = heuristic
    return out


def _apply_trade_constraints(
    signed_volume: float,
    position_before: float,
    adv: float,
    config: ReducedFormStrategyConfig,
) -> tuple[float, float]:
    trade = signed_volume
    if config.max_abs_trade is not None:
        trade = float(np.clip(trade, -config.max_abs_trade, config.max_abs_trade))
    if config.max_participation_rate is not None and np.isfinite(adv) and adv > 0:
        max_trade = config.max_participation_rate * adv
        trade = float(np.clip(trade, -max_trade, max_trade))
    position_after = position_before + trade
    if config.max_abs_position is not None:
        position_after = float(np.clip(position_after, -config.max_abs_position, config.max_abs_position))
        trade = position_after - position_before
    return trade, position_after


def _simulate_group(group: pd.DataFrame, config: ReducedFormStrategyConfig) -> list[dict]:
    beta = config.effective_beta
    position_prev = float(config.initial_position)
    impact_after_prev = float(config.initial_impact)
    cumulative_integral = 0.0
    previous_target = float(config.initial_impact)
    records: list[dict] = []
    last_idx = group.index[-1]

    for idx, row in group.iterrows():
        dt = float(row["dt_minutes"])
        decay = float(np.exp(-beta * dt))
        impact_before = decay * impact_after_prev
        lambda_t = float(row["lambda_t"])
        target = float(row["target_impact"])
        adv = float(row["ADV"]) if "ADV" in row and pd.notna(row["ADV"]) else np.nan
        has_scaling = bool(row.get("has_scaling", False))
        skipped = False

        if config.translation_method == "position_formula":
            cumulative_integral += beta * target * dt / lambda_t
            target_position = target / lambda_t + cumulative_integral
            desired_trade = target_position - position_prev
        else:
            d_impact = target - previous_target
            desired_trade = (beta * impact_before * dt + d_impact) / lambda_t
            target_position = position_prev + desired_trade

        if config.skip_missing_scaling and not has_scaling:
            desired_trade = 0.0
            skipped = True
        signed_trade, position_after = _apply_trade_constraints(desired_trade, position_prev, adv, config)

        liquidation_trade = 0.0
        is_liquidation = False
        if config.liquidate_at_close and idx == last_idx:
            liquidation_trade = -position_after
            signed_trade += liquidation_trade
            position_after = 0.0
            is_liquidation = True

        impact_after = impact_before + lambda_t * signed_trade
        gross_pnl = position_prev * float(row["delta_mid"])
        quadratic_cost = 0.5 * lambda_t * signed_trade**2
        signed_cost = impact_before * signed_trade + quadratic_cost
        net_pnl = gross_pnl - signed_cost
        participation = abs(signed_trade) / adv if np.isfinite(adv) and adv > 0 else np.nan

        rec = row.to_dict()
        rec.update(
            {
                "decay_factor": decay,
                "target_position": target_position,
                "position_before": position_prev,
                "signed_volume_trade": signed_trade,
                "trade": signed_trade,
                "position_after": position_after,
                "impact_before_trade": impact_before,
                "impact_after_trade": impact_after,
                "gross_pnl": gross_pnl,
                "quadratic_impact_cost": quadratic_cost,
                "signed_impact_cost": signed_cost,
                "net_pnl": net_pnl,
                "turnover_shares": abs(signed_trade),
                "turnover_notional": abs(signed_trade) * float(row[config.price_col]),
                "participation_rate": participation,
                "is_liquidation": is_liquidation,
                "liquidation_trade": liquidation_trade,
                "skipped_due_to_missing_scaling": skipped and not is_liquidation,
            }
        )
        records.append(rec)
        position_prev = position_after
        impact_after_prev = impact_after
        previous_target = target
    return records


def run_reduced_form_strategy(df: pd.DataFrame, config: ReducedFormStrategyConfig) -> pd.DataFrame:
    """Run the reduced-form strategy and return row-level trades/PnL."""

    data = prepare_reduced_form_data(df, config)
    if "sigma" not in data.columns or "ADV" not in data.columns or "has_scaling" not in data.columns:
        if config.use_scaling_factors:
            scaling = load_scaling_factors(config)
            data = attach_scaling_factors(data, scaling, config)
        else:
            data["sigma"] = config.fallback_sigma
            data["ADV"] = config.fallback_ADV
            data["has_scaling"] = True
    data = compute_market_volume_state_clock_time(data, config)
    data = compute_liquidity_signals(data, config)
    data = compute_reduced_form_target_impact(data, config)

    rows = []
    for _, group in data.groupby([config.stock_col, config.date_col], sort=False):
        rows.extend(_simulate_group(group, config))
    out = pd.DataFrame(rows)
    out = out.sort_values([config.stock_col, config.date_col, config.timestamp_col]).reset_index(drop=True)
    out["cumulative_wealth"] = out["net_pnl"].cumsum()
    out["cumulative_gross_pnl"] = out["gross_pnl"].cumsum()
    out["cumulative_signed_impact_cost"] = out["signed_impact_cost"].cumsum()
    return out


def run_reduced_form_strategy_from_csv(
    input_path: Path,
    output_dir: Path,
    config: ReducedFormStrategyConfig,
    scaling_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load alpha input CSV, run reduced-form strategy, save trades, and return output."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"strategy input file not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    if scaling_df is not None:
        data = attach_scaling_factors(data, scaling_df, config)
    trades = run_reduced_form_strategy(data, config)
    trades.to_csv(output_dir / "reduced_form_strategy_trades.csv", index=False)
    return trades
