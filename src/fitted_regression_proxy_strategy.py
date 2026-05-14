"""Local myopic proxy strategy induced by teammate fitted regressions.

This is not a closed-form dynamic optimal strategy. Teammate's fitted
``reduced_form`` model is an OLS regression for within-bin ``ret_bps`` rather
than a structural impact SDE. The strategy below uses the regression's local
marginal impact slope with respect to additional strategy order flow and solves
a one-step quadratic-cost proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.fitted_regression_features import compute_training_scales, prepare_model_frame
from src.fitted_regression_params import get_params_for_pair_stock


@dataclass(frozen=True)
class FittedRegressionProxyConfig:
    """Configuration for the fitted-regression local myopic proxy strategy."""

    alpha_col: str = "alpha_for_strategy"
    add_strategy_to_trade_feature: bool = False
    inventory_penalty: float = 1e-8
    min_impact_slope_price_per_share: float = 1e-10
    max_participation_rate_per_trade: float | None = 0.01
    max_abs_position_adv_fraction: float | None = 0.05
    max_abs_trade_adv_fraction: float | None = 0.01
    liquidate_at_close: bool = True

    def __post_init__(self) -> None:
        """Validate fitted proxy settings."""

        if self.inventory_penalty < 0:
            raise ValueError("inventory_penalty must be non-negative")
        if self.min_impact_slope_price_per_share <= 0:
            raise ValueError("min_impact_slope_price_per_share must be positive")
        for name in ["max_participation_rate_per_trade", "max_abs_position_adv_fraction", "max_abs_trade_adv_fraction"]:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be None or positive")


def _cap_trade_and_position(
    raw_trade: float,
    position_before: float,
    adv: float,
    config: FittedRegressionProxyConfig,
) -> tuple[float, float, float, float, bool, bool]:
    """Apply ADV-based trade and position caps."""

    trade = raw_trade
    max_trade_allowed = np.inf
    max_position_allowed = np.inf
    if np.isfinite(adv) and adv > 0:
        trade_caps = []
        if config.max_abs_trade_adv_fraction is not None:
            trade_caps.append(config.max_abs_trade_adv_fraction * adv)
        if config.max_participation_rate_per_trade is not None:
            trade_caps.append(config.max_participation_rate_per_trade * adv)
        if trade_caps:
            max_trade_allowed = float(min(trade_caps))
            trade = float(np.clip(trade, -max_trade_allowed, max_trade_allowed))
        position_after = position_before + trade
        if config.max_abs_position_adv_fraction is not None:
            max_position_allowed = float(config.max_abs_position_adv_fraction * adv)
            clipped_position = float(np.clip(position_after, -max_position_allowed, max_position_allowed))
            trade = clipped_position - position_before
            if np.isfinite(max_trade_allowed):
                trade = float(np.clip(trade, -max_trade_allowed, max_trade_allowed))
            position_after = position_before + trade
    else:
        position_after = position_before + trade
    trade_clipped = abs(trade - raw_trade) > 1e-9
    position_clipped = np.isfinite(max_position_allowed) and abs(position_after) >= max_position_allowed - 1e-9
    return trade, position_after, max_trade_allowed, max_position_allowed, trade_clipped, position_clipped


def build_proxy_strategy_frame(
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    reduced_form_params_df: pd.DataFrame,
    pair_id: int,
    config: FittedRegressionProxyConfig,
) -> pd.DataFrame:
    """Build row-level frame with alpha and local fitted-regression slopes."""

    scales = compute_training_scales(train_raw_df)
    model_frame = prepare_model_frame(test_raw_df, scales)
    alpha = alpha_df.copy()
    if "datetime" not in alpha.columns:
        alpha["datetime"] = pd.to_datetime(alpha.get("timestamp"), errors="coerce")
    alpha["datetime"] = pd.to_datetime(alpha["datetime"], errors="coerce")
    alpha["trading_date"] = alpha.get("trading_date", alpha.get("date")).astype(str)
    alpha_keep = ["stock", "trading_date", "datetime", config.alpha_col, "future_return_h", "valid_future_return"]
    alpha_keep = [col for col in alpha_keep if col in alpha.columns]
    frame = model_frame.merge(alpha[alpha_keep], on=["stock", "trading_date", "datetime"], how="left")
    frame[config.alpha_col] = pd.to_numeric(frame[config.alpha_col], errors="coerce").fillna(0.0)

    slopes = np.full(len(frame), np.nan)
    for stock, idx in frame.groupby("stock", sort=False, observed=True).groups.items():
        params = get_params_for_pair_stock(
            reduced_form_params_df,
            pair_id,
            str(stock),
            "reduced_form",
            allow_missing=True,
        )
        loc = list(idx)
        if params is None:
            continue
        coef = params["coef"]
        flow_scale = pd.to_numeric(frame.loc[loc, "flow_scale"], errors="coerce").replace(0, np.nan)
        depth = pd.to_numeric(frame.loc[loc, "depth"], errors="coerce").replace(0, np.nan)
        slope = float(coef.get("x_flow", 0.0)) / flow_scale
        if "x_flow_depth" in coef:
            slope = slope + float(coef.get("x_flow_depth", 0.0)) / depth
        if config.add_strategy_to_trade_feature and "x_trade" in coef:
            slope = slope + float(coef.get("x_trade", 0.0)) / flow_scale
        slopes[loc] = pd.to_numeric(slope, errors="coerce")

    frame["impact_slope_bps_per_share_raw"] = slopes
    frame["impact_slope_bps_per_share"] = pd.to_numeric(frame["impact_slope_bps_per_share_raw"], errors="coerce")
    # Negative local slopes would imply the fitted model rewards our own impact.
    # For a cost proxy, use a conservative positive floor.
    frame["impact_slope_bps_per_share"] = frame["impact_slope_bps_per_share"].where(
        frame["impact_slope_bps_per_share"] > 0,
        config.min_impact_slope_price_per_share * 10000.0 / frame["mid"].clip(lower=1e-12),
    )
    frame["impact_slope_price_per_share"] = (
        frame["mid"] * frame["impact_slope_bps_per_share"] / 10000.0
    ).clip(lower=config.min_impact_slope_price_per_share)
    frame["ADV"] = frame["flow_scale"]
    frame["alpha"] = frame[config.alpha_col]
    frame["alpha_price"] = frame["mid"] * frame["alpha"]
    frame["pair_id"] = pair_id
    return frame.sort_values(["stock", "trading_date", "datetime"]).reset_index(drop=True)


def run_fitted_regression_proxy_strategy(
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    reduced_form_params_df: pd.DataFrame,
    pair_id: int,
    config: FittedRegressionProxyConfig | None = None,
) -> pd.DataFrame:
    """Run the local myopic quadratic-cost fitted-regression proxy strategy."""

    cfg = config or FittedRegressionProxyConfig()
    frame = build_proxy_strategy_frame(train_raw_df, test_raw_df, alpha_df, reduced_form_params_df, pair_id, cfg)
    rows = []
    for _, group in frame.groupby(["stock", "trading_date"], sort=False, observed=True):
        position_prev = 0.0
        prev_mid = None
        last_idx = group.index[-1]
        for idx, row in group.iterrows():
            mid = float(row["mid"])
            delta_mid = 0.0 if prev_mid is None else mid - float(prev_mid)
            slope = float(row["impact_slope_price_per_share"])
            alpha_price = float(row["alpha_price"])
            adv = float(row["ADV"]) if pd.notna(row["ADV"]) else np.nan
            denom = 2.0 * slope + 2.0 * cfg.inventory_penalty
            raw_trade = (alpha_price - 2.0 * cfg.inventory_penalty * position_prev) / denom if denom > 0 else 0.0
            trade, position_after, max_trade_allowed, max_position_allowed, trade_clipped, position_clipped = (
                _cap_trade_and_position(raw_trade, position_prev, adv, cfg)
            )
            liquidation_trade = 0.0
            if cfg.liquidate_at_close and idx == last_idx:
                before_liq = trade
                liquidation_raw = -(position_prev + trade)
                liq_trade, position_after, max_trade_allowed, max_position_allowed, liq_clipped, pos_liq_clipped = (
                    _cap_trade_and_position(liquidation_raw, position_prev + trade, adv, cfg)
                )
                trade = trade + liq_trade
                position_after = position_prev + trade
                liquidation_trade = liq_trade
                trade_clipped = trade_clipped or liq_clipped or abs(trade - raw_trade) > 1e-9
                position_clipped = position_clipped or pos_liq_clipped
                if np.isfinite(max_trade_allowed):
                    trade = float(np.clip(trade, -max_trade_allowed, max_trade_allowed))
                    position_after = position_prev + trade
                    liquidation_trade = trade - before_liq
            gross_pnl = position_prev * delta_mid
            fitted_cost = slope * trade**2
            net_pnl = gross_pnl - fitted_cost
            rec = row.to_dict()
            rec.update(
                {
                    "delta_mid": delta_mid,
                    "raw_trade": raw_trade,
                    "signed_volume": trade,
                    "position_before": position_prev,
                    "position_after": position_after,
                    "participation_rate": abs(trade) / adv if np.isfinite(adv) and adv > 0 else np.nan,
                    "max_trade_allowed": max_trade_allowed,
                    "max_position_allowed": max_position_allowed,
                    "trade_clipped": trade_clipped,
                    "position_clipped": position_clipped,
                    "liquidation_trade": liquidation_trade,
                    "gross_pnl": gross_pnl,
                    "fitted_impact_cost": fitted_cost,
                    "net_pnl": net_pnl,
                }
            )
            rows.append(rec)
            position_prev = position_after
            prev_mid = mid
    out = pd.DataFrame(rows).sort_values(["stock", "trading_date", "datetime"]).reset_index(drop=True)
    out["cumulative_wealth"] = out["net_pnl"].cumsum()
    out["turnover"] = out["signed_volume"].abs()
    out["notional_turnover"] = out["turnover"] * out["mid"]
    return out


def compute_fitted_proxy_summary(trades: pd.DataFrame) -> dict[str, float]:
    """Compute summary metrics for fitted proxy strategy."""

    daily = trades.groupby("trading_date", sort=True)["net_pnl"].sum()
    sharpe = daily.mean() / daily.std(ddof=1) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    wealth = trades["cumulative_wealth"]
    drawdown = wealth - wealth.cummax()
    return {
        "total_gross_pnl_fitted_proxy": float(trades["gross_pnl"].sum()),
        "total_fitted_proxy_cost": float(trades["fitted_impact_cost"].sum()),
        "total_net_pnl_fitted_proxy": float(trades["net_pnl"].sum()),
        "total_turnover_fitted_proxy": float(trades["turnover"].sum()),
        "total_notional_turnover_fitted_proxy": float(trades["notional_turnover"].sum()),
        "max_drawdown_fitted_proxy": float(drawdown.min()) if len(drawdown) else np.nan,
        "daily_sharpe_fitted_proxy": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_participation_rate_fitted_proxy": float(trades["participation_rate"].max(skipna=True)),
        "mean_participation_rate_fitted_proxy": float(trades["participation_rate"].mean(skipna=True)),
        "mean_abs_trade_fitted_proxy": float(trades["signed_volume"].abs().mean()),
        "number_of_trades_fitted_proxy": int((trades["signed_volume"].abs() > 0).sum()),
    }


def validate_fitted_proxy_strategy(trades: pd.DataFrame, config: FittedRegressionProxyConfig) -> list[dict[str, str]]:
    """Validate fitted proxy strategy mechanics."""

    checks: list[dict[str, str]] = []
    if trades.empty:
        return [{"check": "fitted_proxy_nonempty", "status": "FAIL", "message": "trades empty"}]
    pos_error = (trades["position_after"] - (trades["position_before"] + trades["signed_volume"])).abs().max()
    gross_error = (trades["gross_pnl"] - trades["position_before"] * trades["delta_mid"]).abs().max()
    cost_error = (trades["fitted_impact_cost"] - trades["impact_slope_price_per_share"] * trades["signed_volume"] ** 2).abs().max()
    critical = ["signed_volume", "position_after", "impact_slope_price_per_share", "fitted_impact_cost", "net_pnl"]
    nan_count = int(trades[critical].isna().sum().sum())
    max_part = float(trades["participation_rate"].max(skipna=True))
    cap_ok = config.max_participation_rate_per_trade is None or max_part <= config.max_participation_rate_per_trade + 1e-9
    nonzero = trades["signed_volume"].abs() > 0
    sign_align = (
        np.sign(trades.loc[nonzero, "signed_volume"]) == np.sign(trades.loc[nonzero, "alpha"])
    ).mean() if nonzero.any() else np.nan
    checks.extend(
        [
            {"check": "fitted_proxy_position_dynamics", "status": "PASS" if pos_error < 1e-8 else "FAIL", "message": f"max_error={pos_error:.3g}"},
            {"check": "fitted_proxy_gross_pnl_formula", "status": "PASS" if gross_error < 1e-8 else "FAIL", "message": f"max_error={gross_error:.3g}"},
            {"check": "fitted_proxy_cost_formula", "status": "PASS" if cost_error < 1e-8 else "FAIL", "message": f"max_error={cost_error:.3g}"},
            {"check": "fitted_proxy_participation_cap", "status": "PASS" if cap_ok else "FAIL", "message": f"max={max_part:.6g}; cap={config.max_participation_rate_per_trade}"},
            {"check": "fitted_proxy_no_critical_nans", "status": "PASS" if nan_count == 0 else "FAIL", "message": f"nan_count={nan_count}"},
            {"check": "fitted_proxy_sign_alignment", "status": "PASS" if np.isnan(sign_align) or sign_align >= 0.5 else "WARN", "message": f"share_trade_same_sign_as_alpha={sign_align:.2%}"},
        ]
    )
    return checks


def _portfolio_timeseries(
    df: pd.DataFrame,
    timestamp_col: str,
    cols: list[str],
) -> pd.DataFrame:
    """Aggregate fitted proxy rows across stocks by timestamp before cumulating."""

    out = df[[timestamp_col, *cols]].copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    grouped = out.groupby(timestamp_col, as_index=False, sort=True)[cols].sum()
    for col in cols:
        grouped[f"cumulative_{col}"] = grouped[col].cumsum()
    return grouped


def save_fitted_proxy_plots(proxy_trades: pd.DataFrame, ow_trades: pd.DataFrame, output_dir: Path) -> None:
    """Save fitted proxy strategy plots."""

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    proxy_port = _portfolio_timeseries(proxy_trades, "datetime", ["net_pnl", "gross_pnl", "fitted_impact_cost"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(proxy_port["datetime"], proxy_port["cumulative_net_pnl"], label="net PnL")
    ax.plot(proxy_port["datetime"], proxy_port["cumulative_gross_pnl"], label="gross PnL", alpha=0.75)
    ax.plot(proxy_port["datetime"], proxy_port["cumulative_fitted_impact_cost"], label="fitted cost", alpha=0.75)
    ax.set_title("Fitted-regression proxy strategy: portfolio cumulative wealth")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "fitted_proxy_cumulative_wealth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    if {"timestamp", "net_pnl"}.issubset(ow_trades.columns):
        ow_port = _portfolio_timeseries(ow_trades, "timestamp", ["net_pnl"])
        ax.plot(ow_port["timestamp"], ow_port["cumulative_net_pnl"], label="OW target-impact")
    ax.plot(proxy_port["datetime"], proxy_port["cumulative_net_pnl"], label="fitted proxy")
    ax.set_title("OW vs Fitted Proxy Wealth")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "ow_vs_fitted_proxy_wealth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for col, filename, title in [
        ("signed_volume", "fitted_proxy_trade_histogram.png", "Fitted Proxy Trade Histogram"),
        ("impact_slope_price_per_share", "fitted_proxy_impact_slope_histogram.png", "Fitted Proxy Impact Slope Histogram"),
        ("participation_rate", "fitted_proxy_participation_rate_histogram.png", "Fitted Proxy Participation Rate Histogram"),
    ]:
        values = pd.to_numeric(proxy_trades[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        lo, hi = values.quantile(0.005), values.quantile(0.995)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values.clip(lo, hi), bins=80)
        ax.set_title(title)
        ax.set_xlabel(col)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
