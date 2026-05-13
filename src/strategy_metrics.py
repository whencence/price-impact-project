"""Metrics and plots for OW strategy outputs."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def compute_drawdown(series: pd.Series) -> pd.Series:
    """Return drawdown relative to the running maximum of a wealth series."""

    wealth = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return wealth - wealth.cummax()


def _sum_abs(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").abs().sum())


def compute_daily_strategy_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily strategy metrics across all stocks."""

    grouped = trades_df.groupby("date", sort=True)
    daily = grouped.agg(
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        signed_impact_cost_normalized=("signed_impact_cost_normalized", "sum"),
        quadratic_impact_cost_normalized=("quadratic_impact_cost_normalized", "sum"),
        turnover_shares=("signed_volume", _sum_abs),
        turnover_notional=("signed_volume_notional", lambda x: float(np.abs(x).sum())),
        normalized_turnover=("normalized_trade", _sum_abs),
        max_abs_position=("position_after", lambda x: float(np.abs(x).max())),
        max_abs_impact=("impact_after_trade", lambda x: float(np.abs(x).max())),
        max_participation_rate=("participation_rate", "max"),
        mean_participation_rate=("participation_rate", "mean"),
        n_trades=("signed_volume", lambda x: int((np.abs(x) > 0).sum())),
        n_stock_days=("stock", "nunique"),
        rows_with_scaling=("has_scaling", "sum"),
        rows_skipped=("skipped_due_to_missing_scaling", "sum"),
    ).reset_index()
    daily["cumulative_daily_wealth"] = daily["net_pnl"].cumsum()
    daily["daily_drawdown"] = compute_drawdown(daily["cumulative_daily_wealth"])
    return daily


def compute_overall_strategy_summary(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall OW strategy summary metrics."""

    daily = compute_daily_strategy_metrics(trades_df)
    std_daily = float(daily["net_pnl"].std(ddof=1)) if len(daily) > 1 else np.nan
    mean_daily = float(daily["net_pnl"].mean()) if len(daily) else np.nan
    daily_sharpe = mean_daily / std_daily if std_daily and std_daily > 0 else np.nan
    drawdown = compute_drawdown(trades_df["cumulative_wealth"])
    abs_trade = trades_df["signed_volume"].abs()
    return {
        "total_gross_pnl": float(trades_df["gross_pnl"].sum()),
        "total_net_pnl": float(trades_df["net_pnl"].sum()),
        "total_signed_impact_cost_normalized": float(trades_df["signed_impact_cost_normalized"].sum()),
        "total_quadratic_impact_cost_normalized": float(trades_df["quadratic_impact_cost_normalized"].sum()),
        "mean_daily_net_pnl": mean_daily,
        "std_daily_net_pnl": std_daily,
        "daily_sharpe": daily_sharpe,
        "annualized_sharpe": np.sqrt(252.0) * daily_sharpe if np.isfinite(daily_sharpe) else np.nan,
        "total_signed_volume_turnover": float(abs_trade.sum()),
        "total_notional_turnover": float(trades_df["signed_volume_notional"].abs().sum()),
        "average_daily_turnover_shares": float(daily["turnover_shares"].mean()) if len(daily) else np.nan,
        "average_daily_turnover_notional": float(daily["turnover_notional"].mean()) if len(daily) else np.nan,
        "total_normalized_turnover": float(trades_df["normalized_trade"].abs().sum()),
        "max_participation_rate": float(trades_df["participation_rate"].max(skipna=True)),
        "mean_participation_rate": float(trades_df["participation_rate"].mean(skipna=True)),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "max_daily_drawdown": float(daily["daily_drawdown"].min()) if len(daily) else np.nan,
        "max_abs_position": float(trades_df["position_after"].abs().max()),
        "max_abs_impact": float(trades_df["impact_after_trade"].abs().max()),
        "number_of_trades": int((abs_trade > 0).sum()),
        "number_of_stock_days": int(trades_df[["stock", "date"]].drop_duplicates().shape[0]),
        "mean_abs_trade": float(abs_trade.mean()),
        "median_abs_trade": float(abs_trade.median()),
        "rows": int(len(trades_df)),
        "rows_with_scaling": int(trades_df["has_scaling"].sum()),
        "rows_skipped_due_to_missing_scaling": int(trades_df["skipped_due_to_missing_scaling"].sum()),
    }


def save_strategy_metrics(trades_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Save daily and summary strategy metrics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = compute_daily_strategy_metrics(trades_df)
    summary = compute_overall_strategy_summary(trades_df)
    daily.to_csv(output_dir / "ow_daily_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "ow_summary_metrics.csv", index=False)
    return daily, summary


def _hist_plot(series: pd.Series, title: str, xlabel: str, fig_path: Path) -> None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return
    lo, hi = values.quantile(0.005), values.quantile(0.995)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values.clip(lo, hi), bins=100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def save_strategy_plots(trades_df: pd.DataFrame, daily_metrics: pd.DataFrame, output_dir: Path) -> None:
    """Save standard OW strategy plots with matplotlib."""

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t = pd.to_datetime(trades_df["timestamp"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, trades_df["cumulative_wealth"])
    ax.set_title("OW cumulative wealth")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("wealth")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "ow_cumulative_wealth.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(daily_metrics["date"].astype(str), daily_metrics["net_pnl"])
    ax.set_title("OW daily net PnL")
    ax.set_xlabel("date")
    ax.set_ylabel("net PnL")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "ow_daily_net_pnl.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, trades_df["cumulative_gross_pnl"], label="gross")
    ax.plot(t, trades_df["cumulative_wealth"], label="net")
    ax.set_title("OW cumulative gross vs net PnL")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "ow_gross_vs_net_pnl.png", dpi=150)
    plt.close(fig)

    if len(trades_df):
        sample_key = trades_df.groupby(["stock", "date"], sort=False).size().idxmax()
        sample = trades_df.loc[(trades_df["stock"] == sample_key[0]) & (trades_df["date"] == sample_key[1])].head(500)
        fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(pd.to_datetime(sample["timestamp"]), sample["alpha"], label="alpha")
        axes[1].plot(pd.to_datetime(sample["timestamp"]), sample["target_impact"], label="target")
        axes[1].plot(pd.to_datetime(sample["timestamp"]), sample["impact_after_trade"], label="impact")
        axes[2].plot(pd.to_datetime(sample["timestamp"]), sample["signed_volume"], label="signed volume")
        axes[3].plot(pd.to_datetime(sample["timestamp"]), sample["position_after"], label="position")
        axes[4].plot(pd.to_datetime(sample["timestamp"]), sample["mid"] / sample["mid"].iloc[0], label="mid normalized")
        for ax in axes:
            ax.legend(fontsize=8)
        axes[0].set_title("OW sample strategy path")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "ow_sample_strategy_path.png", dpi=150)
        plt.close(fig)

    _hist_plot(trades_df["signed_volume"], "OW signed volume histogram", "signed_volume", fig_dir / "ow_signed_volume_histogram.png")
    _hist_plot(trades_df["normalized_trade"], "OW normalized trade histogram", "normalized_trade", fig_dir / "ow_normalized_trade_histogram.png")
    _hist_plot(trades_df["impact_after_trade"], "OW impact histogram", "impact_after_trade", fig_dir / "ow_impact_histogram.png")
    _hist_plot(trades_df["participation_rate"], "OW participation rate histogram", "participation_rate", fig_dir / "ow_participation_rate_histogram.png")
    _hist_plot(trades_df["trade"], "OW trade histogram", "trade = signed_volume", fig_dir / "ow_trade_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(daily_metrics["date"].astype(str), daily_metrics["turnover_shares"])
    ax.set_title("OW daily signed-volume turnover")
    ax.set_ylabel("shares")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "ow_daily_turnover.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, compute_drawdown(trades_df["cumulative_wealth"]))
    ax.set_title("OW drawdown")
    ax.set_ylabel("drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "ow_drawdown.png", dpi=150)
    plt.close(fig)
