"""OW optimal trading strategy mechanics for section 2.5.

The course OW convention separates real signed volume q from normalized trade
qtilde. Impact dynamics are driven by qtilde, while position is tracked in
signed volume units.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy_config import OWStrategyConfig


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def validate_strategy_input(df: pd.DataFrame, config: OWStrategyConfig) -> None:
    """Validate the alpha input required by the OW strategy."""

    required = [
        config.date_col,
        config.time_col,
        config.timestamp_col,
        config.stock_col,
        config.price_col,
        config.alpha_col,
    ]
    if df.empty:
        raise ValueError("strategy input must not be empty")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"strategy input missing required columns: {missing}")
    mid = pd.to_numeric(df[config.price_col], errors="coerce")
    alpha = pd.to_numeric(df[config.alpha_col], errors="coerce")
    if mid.notna().mean() < 0.99:
        _warn("more than 1% of mid values are missing/non-numeric")
    if (mid <= 0).fillna(False).mean() > 0.01:
        _warn("more than 1% of mid values are non-positive")
    if alpha.isna().any():
        _warn(f"{int(alpha.isna().sum())} alpha values missing; they will be set to 0")
    timestamp = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    if timestamp.isna().any():
        raise ValueError("timestamp column contains unparseable values")
    duplicates = int(df.duplicated([config.stock_col, config.date_col, config.timestamp_col]).sum())
    if duplicates:
        _warn(f"{duplicates} duplicate stock/date/timestamp keys")


def prepare_strategy_data(df: pd.DataFrame, config: OWStrategyConfig) -> pd.DataFrame:
    """Prepare alpha, price, time gaps, price changes, and backward alpha derivative."""

    validate_strategy_input(df, config)
    out = df.copy()
    out[config.timestamp_col] = pd.to_datetime(out[config.timestamp_col], errors="coerce")
    out[config.price_col] = pd.to_numeric(out[config.price_col], errors="coerce")
    out[config.alpha_col] = pd.to_numeric(out[config.alpha_col], errors="coerce").fillna(0.0)
    out["usable_mid"] = out[config.price_col].notna() & (out[config.price_col] > 0)
    out = out.sort_values([config.stock_col, config.date_col, config.timestamp_col]).reset_index(drop=True)
    grouped = out.groupby([config.stock_col, config.date_col], sort=False)
    out["dt_minutes"] = grouped[config.timestamp_col].diff().dt.total_seconds().div(60.0).fillna(0.0)
    out.loc[out["dt_minutes"] <= 0, "dt_minutes"] = 0.0
    out["delta_mid"] = grouped[config.price_col].diff().fillna(0.0)
    alpha_prev = grouped[config.alpha_col].shift(1)
    out["alpha"] = out[config.alpha_col] * config.alpha_scale
    out["alpha_dot"] = 0.0
    positive_dt = out["dt_minutes"] > 0
    out.loc[positive_dt, "alpha_dot"] = (
        (out.loc[positive_dt, "alpha"] - alpha_prev.loc[positive_dt].fillna(out.loc[positive_dt, "alpha"]))
        / out.loc[positive_dt, "dt_minutes"]
    )
    out.loc[grouped.cumcount() == 0, "alpha_dot"] = 0.0
    if config.winsorize_alpha_dot_quantile is not None and len(out):
        bound = float(out["alpha_dot"].abs().quantile(config.winsorize_alpha_dot_quantile))
        if np.isfinite(bound) and bound > 0:
            out["alpha_dot"] = out["alpha_dot"].clip(-bound, bound)
    return out


def load_scaling_for_strategy(config: OWStrategyConfig, project_root: Path) -> pd.DataFrame:
    """Load trailing sigma/ADV scaling factors for the strategy."""

    if not config.use_scaling_factors:
        return pd.DataFrame(columns=["stock", "date", "trailing_px_vol", "trailing_ADV"])
    if config.scaling_factors_path is None:
        raise ValueError("scaling_factors_path must be set when use_scaling_factors=True")
    path = Path(config.scaling_factors_path)
    if not path.is_absolute():
        path = Path(project_root) / path
    if not path.exists():
        raise FileNotFoundError(f"scaling factors file not found: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def attach_scaling_factors(
    strategy_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    config: OWStrategyConfig,
) -> pd.DataFrame:
    """Merge trailing volatility and ADV scaling factors into strategy rows."""

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
            columns={
                "stock": config.stock_col,
                "trailing_px_vol": "sigma",
                "trailing_ADV": "ADV",
            }
        )
        keep = [config.stock_col, "_merge_date", "sigma", "ADV"]
        extras = [
            col
            for col in ["scaling_window_days", "has_full_scaling_window", "px_vol_current_day", "daily_volume_current_day"]
            if col in scaling.columns
        ]
        out = out.merge(scaling[keep + extras], on=[config.stock_col, "_merge_date"], how="left")
    else:
        out["sigma"] = config.fallback_sigma
        out["ADV"] = config.fallback_ADV
        out["has_full_scaling_window"] = True

    out["sigma"] = pd.to_numeric(out.get("sigma"), errors="coerce")
    out["ADV"] = pd.to_numeric(out.get("ADV"), errors="coerce")
    missing_before_fallback = ~(out["sigma"].notna() & out["ADV"].notna() & (out["sigma"] > 0) & (out["ADV"] > 0))
    if config.fallback_sigma is not None:
        out.loc[out["sigma"].isna() | (out["sigma"] <= 0), "sigma"] = config.fallback_sigma
    if config.fallback_ADV is not None:
        out.loc[out["ADV"].isna() | (out["ADV"] <= 0), "ADV"] = config.fallback_ADV
    out["has_scaling"] = out["sigma"].notna() & out["ADV"].notna() & (out["sigma"] > 0) & (out["ADV"] > 0)
    if missing_before_fallback.any():
        pct = float(missing_before_fallback.mean() * 100.0)
        missing_keys = out.loc[missing_before_fallback, [config.stock_col, config.date_col]].drop_duplicates().shape[0]
        _warn(f"{pct:.2f}% of rows were missing trailing sigma/ADV before fallback; missing stock-days={missing_keys}")
    out = out.drop(columns=["_merge_date"])
    return out


def normalize_signed_volume(
    q: float | np.ndarray | pd.Series,
    sigma: float | np.ndarray | pd.Series,
    ADV: float | np.ndarray | pd.Series,
    model_type: str,
) -> float | np.ndarray | pd.Series:
    """Convert signed volume q into normalized OW trade qtilde."""

    q_arr = np.asarray(q, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    adv_arr = np.asarray(ADV, dtype=float)
    q_b, sigma_b, adv_b = np.broadcast_arrays(q_arr, sigma_arr, adv_arr)
    valid = (sigma_b > 0) & (adv_b > 0)
    out = np.full(q_b.shape, np.nan, dtype=float)
    if model_type == "linear":
        out[valid] = sigma_b[valid] * q_b[valid] / adv_b[valid]
    elif model_type == "sqrt":
        out[valid] = sigma_b[valid] * np.sign(q_b[valid]) * np.sqrt(np.abs(q_b[valid]) / adv_b[valid])
    else:
        raise ValueError("model_type must be 'linear' or 'sqrt'")
    if np.isscalar(q) and np.isscalar(sigma) and np.isscalar(ADV):
        return float(out)
    return out


def invert_normalized_trade(
    qtilde: float,
    sigma: float,
    ADV: float,
    model_type: str,
) -> float:
    """Convert normalized OW trade qtilde back to real signed volume q."""

    if not np.isfinite(qtilde) or qtilde == 0:
        return 0.0
    if not np.isfinite(sigma) or not np.isfinite(ADV) or sigma <= 0 or ADV <= 0:
        return np.nan
    if model_type == "linear":
        return float(qtilde * ADV / sigma)
    if model_type == "sqrt":
        return float(np.sign(qtilde) * ADV * (abs(qtilde) / sigma) ** 2)
    raise ValueError("model_type must be 'linear' or 'sqrt'")


def _stock_param(config: OWStrategyConfig, stock: str, key: str, default: float) -> float:
    if config.per_stock_impact_params and stock in config.per_stock_impact_params:
        return float(config.per_stock_impact_params[stock].get(key, default))
    return float(default)


def get_effective_impact_lambda(config: OWStrategyConfig, stock: str | None = None) -> float:
    """Return the impact lambda used for a stock, falling back to the global config."""

    if stock is None:
        return float(config.impact_lambda)
    return _stock_param(config, stock, "impact_lambda", config.impact_lambda)


def compute_impact_beta(config: OWStrategyConfig, stock: str | None = None) -> float:
    """Return beta_I = ln(2) / impact_half_life_minutes."""

    half_life = config.impact_half_life_minutes
    if stock is not None:
        half_life = _stock_param(config, stock, "impact_half_life_minutes", half_life)
    return float(np.log(2.0) / half_life)


def compute_target_impact(df: pd.DataFrame, config: OWStrategyConfig) -> pd.DataFrame:
    """Add OW target impact I_target = 0.5 alpha - alpha_dot / beta."""

    out = df.copy()
    out["impact_beta"] = out[config.stock_col].astype(str).map(lambda s: compute_impact_beta(config, s))
    out["target_impact"] = (0.5 * out["alpha"] - out["alpha_dot"] / out["impact_beta"]) * config.target_impact_scale
    if config.max_abs_target_impact is not None:
        out["target_impact"] = out["target_impact"].clip(
            -config.max_abs_target_impact, config.max_abs_target_impact
        )
    return out


def _row_records_for_group(group: pd.DataFrame, config: OWStrategyConfig) -> list[dict]:
    stock = str(group[config.stock_col].iloc[0])
    beta = compute_impact_beta(config, stock)
    impact_lambda = get_effective_impact_lambda(config, stock)
    position_prev = float(config.initial_position)
    impact_after_prev = float(config.initial_impact)
    records: list[dict] = []
    last_idx = group.index[-1]

    for idx, row in group.iterrows():
        dt = float(row["dt_minutes"])
        decay = float(np.exp(-beta * dt))
        impact_before = decay * impact_after_prev
        target = float(row["target_impact"])
        sigma = float(row["sigma"]) if "sigma" in row and pd.notna(row["sigma"]) else np.nan
        adv = float(row["ADV"]) if "ADV" in row and pd.notna(row["ADV"]) else np.nan
        has_scaling = bool(row.get("has_scaling", False))

        required_qtilde = (target - impact_before) / impact_lambda
        skipped = False
        raw_signed_volume = 0.0
        signed_volume_before_clipping = 0.0
        signed_volume = 0.0
        normalized_trade = 0.0

        if has_scaling:
            raw_signed_volume = invert_normalized_trade(
                required_qtilde, sigma, adv, config.impact_model_type
            )
            if not np.isfinite(raw_signed_volume):
                raw_signed_volume = 0.0
                skipped = True
        else:
            skipped = True

        signed_volume = raw_signed_volume
        if config.max_abs_trade is not None:
            signed_volume = float(np.clip(signed_volume, -config.max_abs_trade, config.max_abs_trade))
        max_trade_allowed = np.inf
        if np.isfinite(adv) and adv > 0:
            trade_caps = []
            if config.max_abs_trade_adv_fraction is not None:
                trade_caps.append(config.max_abs_trade_adv_fraction * adv)
            if config.max_participation_rate_per_trade is not None:
                trade_caps.append(config.max_participation_rate_per_trade * adv)
            if trade_caps:
                max_trade_allowed = float(min(trade_caps))
                signed_volume = float(np.clip(signed_volume, -max_trade_allowed, max_trade_allowed))
        signed_volume_before_clipping = raw_signed_volume

        position_before = position_prev
        position_after = position_before + signed_volume
        max_position_allowed = np.inf
        if config.max_abs_position is not None:
            clipped_position = float(
                np.clip(position_after, -config.max_abs_position, config.max_abs_position)
            )
            signed_volume = clipped_position - position_before
            position_after = clipped_position
        if np.isfinite(adv) and adv > 0 and config.max_abs_position_adv_fraction is not None:
            max_position_allowed = float(config.max_abs_position_adv_fraction * adv)
            clipped_position = float(np.clip(position_after, -max_position_allowed, max_position_allowed))
            signed_volume = clipped_position - position_before
            position_after = clipped_position
        if np.isfinite(max_trade_allowed):
            signed_volume = float(np.clip(signed_volume, -max_trade_allowed, max_trade_allowed))
            position_after = position_before + signed_volume

        liquidation_trade = 0.0
        is_liquidation = False
        if config.liquidate_at_close and idx == last_idx:
            signed_volume_before_liquidation = signed_volume
            liquidation_trade_raw = -position_after
            liquidation_trade = liquidation_trade_raw
            if np.isfinite(max_trade_allowed):
                liquidation_trade = float(np.clip(liquidation_trade, -max_trade_allowed, max_trade_allowed))
            signed_volume += liquidation_trade
            position_after = position_before + signed_volume
            if np.isfinite(max_position_allowed):
                clipped_position = float(np.clip(position_after, -max_position_allowed, max_position_allowed))
                signed_volume = clipped_position - position_before
                position_after = clipped_position
            if np.isfinite(max_trade_allowed):
                signed_volume = float(np.clip(signed_volume, -max_trade_allowed, max_trade_allowed))
                position_after = position_before + signed_volume
            liquidation_trade = signed_volume - signed_volume_before_liquidation
            is_liquidation = True

        if has_scaling:
            normalized_trade = float(
                normalize_signed_volume(signed_volume, sigma, adv, config.impact_model_type)
            )
            if not np.isfinite(normalized_trade):
                normalized_trade = 0.0
                skipped = True
        else:
            normalized_trade = 0.0

        impact_after = impact_before + impact_lambda * normalized_trade
        gross_pnl = position_before * float(row["delta_mid"])
        quadratic_cost = 0.5 * impact_lambda * normalized_trade**2
        signed_cost = (
            impact_before * normalized_trade
            + quadratic_cost
            + config.liquidation_penalty * abs(liquidation_trade)
        )
        net_pnl = gross_pnl - signed_cost
        participation = abs(signed_volume) / adv if np.isfinite(adv) and adv > 0 else np.nan
        trade_clipped = bool(abs(raw_signed_volume - signed_volume) > 1e-9)
        position_clipped = bool(np.isfinite(max_position_allowed) and abs(position_after) >= max_position_allowed - 1e-9)

        rec = row.to_dict()
        rec.update(
            {
                "impact_model_type": config.impact_model_type,
                "impact_lambda": impact_lambda,
                "decay_factor": decay,
                "impact_before_trade": impact_before,
                "required_normalized_trade": required_qtilde,
                "normalized_trade": normalized_trade,
                "raw_signed_volume": raw_signed_volume,
                "signed_volume_before_clipping": signed_volume_before_clipping,
                "signed_volume_after_clipping": signed_volume,
                "trade_clipped": trade_clipped,
                "position_clipped": position_clipped,
                "max_trade_allowed": max_trade_allowed,
                "max_position_allowed": max_position_allowed,
                "signed_volume": signed_volume,
                "trade": signed_volume,
                "position_before": position_before,
                "position_after": position_after,
                "impact_after_trade": impact_after,
                "participation_rate": participation,
                "abs_signed_volume": abs(signed_volume),
                "abs_normalized_trade": abs(normalized_trade),
                "signed_volume_notional": signed_volume * float(row[config.price_col]),
                "skipped_due_to_missing_scaling": skipped and not is_liquidation,
                "gross_pnl": gross_pnl,
                "quadratic_impact_cost_normalized": quadratic_cost,
                "signed_impact_cost_normalized": signed_cost,
                "quadratic_impact_cost": quadratic_cost,
                "signed_impact_cost": signed_cost,
                "net_pnl": net_pnl,
                "is_liquidation": is_liquidation,
                "liquidation_trade": liquidation_trade,
            }
        )
        records.append(rec)
        position_prev = position_after
        impact_after_prev = impact_after
    return records


def run_ow_strategy(df: pd.DataFrame, config: OWStrategyConfig) -> pd.DataFrame:
    """Run the discrete OW strategy using sigma/ADV-normalized signed volume."""

    prepared = prepare_strategy_data(df, config)
    if "sigma" not in prepared.columns or "ADV" not in prepared.columns or "has_scaling" not in prepared.columns:
        if config.use_scaling_factors:
            raise ValueError("strategy data must include sigma, ADV, and has_scaling; call attach_scaling_factors first")
        prepared["sigma"] = config.fallback_sigma
        prepared["ADV"] = config.fallback_ADV
        prepared["has_scaling"] = True
    prepared = compute_target_impact(prepared, config)
    rows = []
    for _, group in prepared.groupby([config.stock_col, config.date_col], sort=False):
        rows.extend(_row_records_for_group(group, config))
    out = pd.DataFrame(rows)
    out = out.sort_values([config.stock_col, config.date_col, config.timestamp_col]).reset_index(drop=True)
    out["cumulative_wealth"] = out["net_pnl"].cumsum()
    out["cumulative_gross_pnl"] = out["gross_pnl"].cumsum()
    out["cumulative_signed_impact_cost_normalized"] = out["signed_impact_cost_normalized"].cumsum()
    out["cumulative_wealth_by_stock"] = out.groupby(config.stock_col, sort=False)["net_pnl"].cumsum()
    keep_front = [
        config.date_col,
        config.time_col,
        config.timestamp_col,
        config.stock_col,
        config.price_col,
        "alpha",
        "alpha_dot",
        "target_impact",
        "impact_before_trade",
        "impact_after_trade",
        "impact_beta",
        "impact_lambda",
        "dt_minutes",
        "decay_factor",
        "sigma",
        "ADV",
        "has_scaling",
        "impact_model_type",
        "required_normalized_trade",
        "normalized_trade",
        "raw_signed_volume",
        "signed_volume_before_clipping",
        "signed_volume_after_clipping",
        "trade_clipped",
        "position_clipped",
        "max_trade_allowed",
        "max_position_allowed",
        "signed_volume",
        "trade",
        "position_before",
        "position_after",
        "participation_rate",
        "skipped_due_to_missing_scaling",
        "is_liquidation",
        "liquidation_trade",
        "delta_mid",
        "gross_pnl",
        "quadratic_impact_cost_normalized",
        "signed_impact_cost_normalized",
        "quadratic_impact_cost",
        "signed_impact_cost",
        "net_pnl",
        "cumulative_wealth",
    ]
    remaining = [col for col in out.columns if col not in keep_front]
    return out[[col for col in keep_front if col in out.columns] + remaining]


def run_ow_strategy_from_csv(
    input_path: Path,
    output_dir: Path,
    config: OWStrategyConfig,
    scaling_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load alpha input CSV, attach scaling if provided, run OW strategy, and save trades."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"strategy input file not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    if scaling_df is not None:
        data = attach_scaling_factors(data, scaling_df, config)
    trades = run_ow_strategy(data, config)
    trades.to_csv(output_dir / "ow_strategy_trades.csv", index=False)
    return trades
