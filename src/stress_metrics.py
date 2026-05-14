"""Metrics, comparisons, and plots for section 2.7 stress tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def safe_divide(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    """Return numerator / denominator, or NaN when denominator is too small."""

    return float(numerator / denominator) if np.isfinite(denominator) and abs(denominator) > eps else np.nan


def compute_drawdown_from_trades(trades_df: pd.DataFrame) -> pd.Series:
    """Compute drawdown from cumulative wealth or cumulative net PnL."""

    if "cumulative_wealth" in trades_df.columns:
        wealth = pd.to_numeric(trades_df["cumulative_wealth"], errors="coerce").fillna(0.0)
    else:
        wealth = pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0.0).cumsum()
    return wealth - wealth.cummax()


def extract_strategy_summary(
    trades_df: pd.DataFrame,
    scenario_name: str,
    scenario_type: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract robust summary metrics from a strategy trades DataFrame."""

    metadata = metadata or {}
    df = trades_df.copy()
    date_col = "date"
    daily = df.groupby(date_col, sort=True).agg(
        net_pnl=("net_pnl", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        turnover=("trade", lambda x: float(np.abs(pd.to_numeric(x, errors="coerce")).sum())),
    )
    mean_daily = float(daily["net_pnl"].mean()) if len(daily) else np.nan
    std_daily = float(daily["net_pnl"].std(ddof=1)) if len(daily) > 1 else np.nan
    daily_sharpe = safe_divide(mean_daily, std_daily)
    drawdown = compute_drawdown_from_trades(df)
    signed_cost_col = "signed_impact_cost_normalized" if "signed_impact_cost_normalized" in df.columns else "signed_impact_cost"
    quad_cost_col = "quadratic_impact_cost_normalized" if "quadratic_impact_cost_normalized" in df.columns else "quadratic_impact_cost"
    turnover_col = "signed_volume" if "signed_volume" in df.columns else "trade"
    summary = {
        "scenario_name": scenario_name,
        "scenario_type": scenario_type,
        "total_gross_pnl": float(df.get("gross_pnl", pd.Series(dtype=float)).sum()),
        "total_net_pnl": float(df.get("net_pnl", pd.Series(dtype=float)).sum()),
        "total_signed_impact_cost": float(df.get(signed_cost_col, pd.Series(dtype=float)).sum()),
        "total_quadratic_impact_cost": float(df.get(quad_cost_col, pd.Series(dtype=float)).sum()),
        "mean_daily_net_pnl": mean_daily,
        "std_daily_net_pnl": std_daily,
        "daily_sharpe": daily_sharpe,
        "annualized_sharpe": np.sqrt(252.0) * daily_sharpe if np.isfinite(daily_sharpe) else np.nan,
        "total_turnover": float(df.get(turnover_col, pd.Series(dtype=float)).abs().sum()),
        "total_notional_turnover": float(df.get("signed_volume_notional", pd.Series(dtype=float)).abs().sum()),
        "average_daily_turnover": float(daily["turnover"].mean()) if len(daily) else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "max_daily_drawdown": float((daily["net_pnl"].cumsum() - daily["net_pnl"].cumsum().cummax()).min()) if len(daily) else np.nan,
        "max_abs_position": float(df.get("position_after", pd.Series(dtype=float)).abs().max()),
        "max_abs_impact": float(df.get("impact_after_trade", pd.Series(dtype=float)).abs().max()),
        "max_participation_rate": float(df.get("participation_rate", pd.Series(dtype=float)).max(skipna=True)),
        "mean_participation_rate": float(df.get("participation_rate", pd.Series(dtype=float)).mean(skipna=True)),
        "n_trades": int((df.get(turnover_col, pd.Series(dtype=float)).abs() > 0).sum()),
        "n_rows": int(len(df)),
        "n_stock_days": int(df[["stock", "date"]].drop_duplicates().shape[0]) if {"stock", "date"}.issubset(df.columns) else np.nan,
        "n_dates": int(df["date"].nunique()) if "date" in df.columns else np.nan,
        "n_stocks": int(df["stock"].nunique()) if "stock" in df.columns else np.nan,
    }
    summary.update(metadata)
    return summary


def compare_to_baseline(
    summary_df: pd.DataFrame,
    baseline_scenario_name: str = "baseline_OW",
) -> pd.DataFrame:
    """Add baseline-relative comparison columns."""

    out = summary_df.copy()
    base_rows = out[out["scenario_name"] == baseline_scenario_name]
    if base_rows.empty:
        return out
    base = base_rows.iloc[0]
    out["baseline_total_net_pnl"] = base["total_net_pnl"]
    out["delta_net_pnl_vs_baseline"] = out["total_net_pnl"] - base["total_net_pnl"]
    out["pct_net_pnl_degradation_vs_baseline"] = -out["delta_net_pnl_vs_baseline"].apply(
        lambda x: safe_divide(x, base["total_net_pnl"])
    )
    out["delta_sharpe_vs_baseline"] = out["daily_sharpe"] - base["daily_sharpe"]
    out["delta_turnover_vs_baseline"] = out["total_turnover"] - base["total_turnover"]
    out["delta_impact_cost_vs_baseline"] = out["total_signed_impact_cost"] - base["total_signed_impact_cost"]
    out["delta_max_drawdown_vs_baseline"] = out["max_drawdown"] - base["max_drawdown"]
    out["delta_max_abs_impact_vs_baseline"] = out["max_abs_impact"] - base["max_abs_impact"]
    return out


def save_stress_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Save stress summary CSV."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(Path(output_dir) / "stress_summary.csv", index=False)


def save_sensitivity_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Save sensitivity summary CSV."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(Path(output_dir) / "sensitivity_summary.csv", index=False)


def _plot_bar(df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    if df.empty or value_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_df = df.sort_values(value_col)
    ax.bar(plot_df["scenario_name"].astype(str), plot_df[value_col])
    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=80)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary_bar_plots(all_summary: pd.DataFrame, fig_dir: Path) -> None:
    """Save report-level scenario comparison bar plots."""

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_bar(all_summary, "total_net_pnl", "Scenario total net PnL", fig_dir / "scenario_net_pnl_bar.png")
    _plot_bar(all_summary, "daily_sharpe", "Scenario daily Sharpe", fig_dir / "scenario_sharpe_bar.png")
    _plot_bar(all_summary, "max_drawdown", "Scenario max drawdown", fig_dir / "scenario_drawdown_bar.png")


def save_cumulative_wealth_plot(
    series_map: dict[str, pd.DataFrame],
    fig_path: Path,
    title: str,
) -> None:
    """Save cumulative wealth comparison plot."""

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in series_map.items():
        if "timestamp" in df.columns and "cumulative_wealth" in df.columns:
            ax.plot(pd.to_datetime(df["timestamp"]), df["cumulative_wealth"], label=label)
    ax.set_title(title)
    ax.set_ylabel("cumulative wealth")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
