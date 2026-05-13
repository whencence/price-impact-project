"""Metrics and plots for reduced-form strategy outputs."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def compute_drawdown(series: pd.Series) -> pd.Series:
    """Return drawdown relative to running maximum."""

    wealth = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return wealth - wealth.cummax()


def compute_daily_reduced_form_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily metrics for the reduced-form strategy."""

    daily = (
        trades_df.groupby("date", sort=True)
        .agg(
            gross_pnl=("gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
            signed_impact_cost=("signed_impact_cost", "sum"),
            quadratic_impact_cost=("quadratic_impact_cost", "sum"),
            turnover_shares=("turnover_shares", "sum"),
            turnover_notional=("turnover_notional", "sum"),
            max_abs_position=("position_after", lambda x: float(np.abs(x).max())),
            max_abs_impact=("impact_after_trade", lambda x: float(np.abs(x).max())),
            max_participation_rate=("participation_rate", "max"),
            mean_participation_rate=("participation_rate", "mean"),
            n_trades=("signed_volume_trade", lambda x: int((np.abs(x) > 0).sum())),
            n_stock_days=("stock", "nunique"),
            rows_with_scaling=("has_scaling", "sum"),
            rows_skipped=("skipped_due_to_missing_scaling", "sum"),
        )
        .reset_index()
    )
    daily["cumulative_daily_wealth"] = daily["net_pnl"].cumsum()
    daily["daily_drawdown"] = compute_drawdown(daily["cumulative_daily_wealth"])
    return daily


def compute_reduced_form_summary(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall reduced-form strategy metrics."""

    daily = compute_daily_reduced_form_metrics(trades_df)
    mean_daily = float(daily["net_pnl"].mean()) if len(daily) else np.nan
    std_daily = float(daily["net_pnl"].std(ddof=1)) if len(daily) > 1 else np.nan
    daily_sharpe = mean_daily / std_daily if std_daily and std_daily > 0 else np.nan
    drawdown = compute_drawdown(trades_df["cumulative_wealth"])
    return {
        "total_gross_pnl": float(trades_df["gross_pnl"].sum()),
        "total_net_pnl": float(trades_df["net_pnl"].sum()),
        "total_signed_impact_cost": float(trades_df["signed_impact_cost"].sum()),
        "total_quadratic_impact_cost": float(trades_df["quadratic_impact_cost"].sum()),
        "mean_daily_net_pnl": mean_daily,
        "std_daily_net_pnl": std_daily,
        "daily_sharpe": daily_sharpe,
        "annualized_sharpe": np.sqrt(252.0) * daily_sharpe if np.isfinite(daily_sharpe) else np.nan,
        "total_turnover_shares": float(trades_df["turnover_shares"].sum()),
        "total_turnover_notional": float(trades_df["turnover_notional"].sum()),
        "average_daily_turnover_shares": float(daily["turnover_shares"].mean()) if len(daily) else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "max_daily_drawdown": float(daily["daily_drawdown"].min()) if len(daily) else np.nan,
        "max_abs_position": float(trades_df["position_after"].abs().max()),
        "max_abs_impact": float(trades_df["impact_after_trade"].abs().max()),
        "max_participation_rate": float(trades_df["participation_rate"].max(skipna=True)),
        "mean_participation_rate": float(trades_df["participation_rate"].mean(skipna=True)),
        "n_trades": int((trades_df["signed_volume_trade"].abs() > 0).sum()),
        "n_stock_days": int(trades_df[["stock", "date"]].drop_duplicates().shape[0]),
        "rows": int(len(trades_df)),
        "rows_with_scaling": int(trades_df["has_scaling"].sum()),
        "rows_skipped_due_to_missing_scaling": int(trades_df["skipped_due_to_missing_scaling"].sum()),
    }


def save_reduced_form_metrics(
    trades_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Save daily and summary reduced-form strategy metrics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = compute_daily_reduced_form_metrics(trades_df)
    summary = compute_reduced_form_summary(trades_df)
    daily.to_csv(output_dir / "reduced_form_daily_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "reduced_form_summary_metrics.csv", index=False)
    return daily, summary


def _hist(series: pd.Series, title: str, xlabel: str, path: Path) -> None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return
    lo, hi = values.quantile(0.005), values.quantile(0.995)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values.clip(lo, hi), bins=100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_reduced_form_plots(trades_df: pd.DataFrame, daily_metrics: pd.DataFrame, output_dir: Path) -> None:
    """Save reduced-form strategy plots using matplotlib."""

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t = pd.to_datetime(trades_df["timestamp"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, trades_df["cumulative_wealth"])
    ax.set_title("Reduced-form cumulative wealth")
    ax.set_ylabel("wealth")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "reduced_form_cumulative_wealth.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(daily_metrics["date"].astype(str), daily_metrics["net_pnl"])
    ax.set_title("Reduced-form daily net PnL")
    ax.set_ylabel("net PnL")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "reduced_form_daily_net_pnl.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, trades_df["cumulative_gross_pnl"], label="gross")
    ax.plot(t, trades_df["cumulative_wealth"], label="net")
    ax.set_title("Reduced-form gross vs net cumulative PnL")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "reduced_form_gross_vs_net_pnl.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, compute_drawdown(trades_df["cumulative_wealth"]))
    ax.set_title("Reduced-form drawdown")
    ax.set_ylabel("drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "reduced_form_drawdown.png", dpi=150)
    plt.close(fig)

    if len(trades_df):
        key = trades_df.groupby(["stock", "date"], sort=False).size().idxmax()
        sample = trades_df[(trades_df["stock"] == key[0]) & (trades_df["date"] == key[1])].head(500)
        x = pd.to_datetime(sample["timestamp"])
        fig, axes = plt.subplots(8, 1, figsize=(12, 14), sharex=True)
        for ax, col in zip(
            axes,
            ["alpha_t", "mu_t", "target_impact", "impact_after_trade", "lambda_t", "local_volume_state_v", "position_after"],
            strict=False,
        ):
            ax.plot(x, sample[col])
            ax.set_ylabel(col)
        axes[-1].plot(x, sample["mid"] / sample["mid"].iloc[0])
        axes[-1].set_ylabel("mid norm.")
        axes[0].set_title("Reduced-form sample stock/day path")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "reduced_form_sample_strategy_path.png", dpi=150)
        plt.close(fig)

        valid = sample[["target_impact", "impact_after_trade"]].dropna()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(valid["target_impact"], valid["impact_after_trade"], s=8, alpha=0.5)
        ax.set_title("Reduced-form target impact vs realized impact")
        ax.set_xlabel("target impact")
        ax.set_ylabel("impact after trade")
        fig.tight_layout()
        fig.savefig(fig_dir / "reduced_form_target_vs_impact.png", dpi=150)
        plt.close(fig)

    _hist(trades_df["lambda_t"], "Reduced-form lambda_t histogram", "lambda_t", fig_dir / "reduced_form_lambda_t_histogram.png")
    _hist(trades_df["local_volume_state_v"], "Reduced-form local volume state histogram", "v_t", fig_dir / "reduced_form_volume_state_histogram.png")
    _hist(trades_df["signed_volume_trade"], "Reduced-form signed volume histogram", "signed_volume_trade", fig_dir / "reduced_form_signed_volume_histogram.png")
    _hist(trades_df["participation_rate"], "Reduced-form participation rate histogram", "participation_rate", fig_dir / "reduced_form_participation_rate_histogram.png")
