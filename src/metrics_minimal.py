"""Minimal performance metrics for local testing."""

from typing import Any

import numpy as np
import pandas as pd


def summarize_performance(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact summary of strategy performance."""

    required = {"date", "pnl", "impact_cost", "trade", "position"}
    missing = required.difference(trades_df.columns)
    if missing:
        raise ValueError(f"trades_df is missing required columns: {sorted(missing)}")

    daily_pnl = trades_df.groupby("date", sort=False)["pnl"].sum()
    cumulative = trades_df["pnl"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    std_daily = float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0
    mean_daily = float(daily_pnl.mean()) if len(daily_pnl) else 0.0
    daily_sharpe = float(np.sqrt(252) * mean_daily / std_daily) if std_daily > 0 else 0.0
    return {
        "total_pnl": float(trades_df["pnl"].sum()),
        "mean_daily_pnl": mean_daily,
        "std_daily_pnl": std_daily,
        "daily_sharpe": daily_sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "total_impact_cost": float(trades_df["impact_cost"].sum()),
        "total_turnover": float(trades_df["trade"].abs().sum()),
        "max_abs_position": float(trades_df["position"].abs().max()),
    }


def compare_stress_results(results_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare minimal metrics across baseline and stress scenarios."""

    rows = []
    for name, trades in results_dict.items():
        metrics = summarize_performance(trades)
        metrics["scenario"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("scenario")

