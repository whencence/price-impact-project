"""Reporting and plots for integrated rolling simulations."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def clean_figure_dir(fig_dir: Path, patterns: tuple[str, ...] = ("*.png", "*.pdf")) -> list[Path]:
    """Delete generated figure files from a figure directory only."""

    fig_dir = Path(fig_dir)
    removed: list[Path] = []
    if not fig_dir.exists():
        return removed
    for pattern in patterns:
        for path in fig_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def archive_figure_dir(source_dir: Path, archive_dir: Path, patterns: tuple[str, ...] = ("*.png", "*.pdf")) -> list[Path]:
    """Copy latest figures into a run-specific archive directory."""

    source_dir = Path(source_dir)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    if not source_dir.exists():
        return copied
    for pattern in patterns:
        for path in source_dir.glob(pattern):
            if path.is_file():
                target = archive_dir / path.name
                shutil.copy2(path, target)
                copied.append(target)
    return copied


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "nan"
    if abs(v) >= 1e5 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.{digits}e}"
    return f"{v:.{digits}f}"


def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _hist(series: pd.Series, title: str, xlabel: str, path: Path) -> None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return
    lo, hi = values.quantile(0.005), values.quantile(0.995)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values.clip(lo, hi), bins=80)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_portfolio_timeseries(
    df: pd.DataFrame,
    timestamp_col: str = "datetime",
    pnl_cols: list[str] | tuple[str, ...] = ("net_pnl",),
) -> pd.DataFrame:
    """Aggregate row-level stock data to one portfolio row per timestamp.

    Portfolio figures must sum PnL/costs across stocks first and only then take
    cumulative sums. Plotting per-stock cumulative paths on one axis makes the
    report figures look like many superposed horizontal lines.
    """

    if df.empty:
        return pd.DataFrame(columns=[timestamp_col, *pnl_cols])
    missing = [col for col in [timestamp_col, *pnl_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"make_portfolio_timeseries missing columns: {missing}")
    out = df[[timestamp_col, *pnl_cols]].copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    for col in pnl_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    grouped = out.groupby(timestamp_col, as_index=False, sort=True)[list(pnl_cols)].sum()
    for col in pnl_cols:
        grouped[f"cumulative_{col}"] = grouped[col].cumsum()
    if grouped[timestamp_col].duplicated().any():
        raise ValueError("portfolio timeseries contains duplicate timestamps after aggregation")
    return grouped


def save_pair_plots(pair_dir: Path, trades: pd.DataFrame, ow_eval: pd.DataFrame, rf_eval: pd.DataFrame) -> None:
    """Save key pair-level plots."""

    fig_dir = Path(pair_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if {"timestamp", "net_pnl"}.issubset(trades.columns):
        portfolio = make_portfolio_timeseries(trades, "timestamp", ["net_pnl"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(portfolio["timestamp"], portfolio["cumulative_net_pnl"], label="my OW")
        ax.set_title("My OW strategy portfolio cumulative wealth")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "pair_cumulative_wealth_my_ow.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    if "participation_rate" in trades.columns:
        _hist(
            trades["participation_rate"],
            "Participation Rate Per Trade",
            "abs(signed_volume) / ADV",
            fig_dir / "participation_rate_histogram.png",
        )
    if {"position_after", "ADV"}.issubset(trades.columns):
        pos_over_adv = trades["position_after"].abs() / trades["ADV"].replace(0, np.nan)
        _hist(pos_over_adv, "Absolute Position Over ADV", "abs(position) / ADV", fig_dir / "position_over_ADV_histogram.png")
    if {"date", "trade_clipped"}.issubset(trades.columns):
        clipped = trades.groupby("date", sort=True)["trade_clipped"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(clipped["date"].astype(str), clipped["trade_clipped"])
        ax.set_title("Share Of Trades Clipped By Day")
        ax.set_ylabel("clipped share")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "clipped_trade_share_by_day.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    sensitivity_path = Path(pair_dir) / "sizing_sensitivity_summary.csv"
    if sensitivity_path.exists():
        sens = pd.read_csv(sensitivity_path)
        if len(sens):
            fig, ax = plt.subplots(figsize=(9, 5))
            for cap, group in sens.groupby("max_participation_rate", sort=True):
                ax.plot(group["target_impact_scale"], group["total_net_pnl"], marker="o", label=f"cap={cap:g}")
            ax.set_title("Sizing Sensitivity: Net PnL")
            ax.set_xlabel("target_impact_scale")
            ax.set_ylabel("total_net_pnl")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "sizing_sensitivity_net_pnl.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 5))
            for cap, group in sens.groupby("max_participation_rate", sort=True):
                ax.plot(group["target_impact_scale"], group["total_turnover"], marker="o", label=f"cap={cap:g}")
            ax.set_title("Sizing Sensitivity: Turnover")
            ax.set_xlabel("target_impact_scale")
            ax.set_ylabel("total_turnover")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "sizing_sensitivity_turnover.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    n_lines = 0
    if len(ow_eval) and {"datetime", "net_pnl_fitted_model"}.issubset(ow_eval.columns):
        ow_port = make_portfolio_timeseries(ow_eval, "datetime", ["net_pnl_fitted_model"])
        ax.plot(ow_port["datetime"], ow_port["cumulative_net_pnl_fitted_model"], label="OW_transient eval")
        n_lines += 1
    if len(rf_eval) and {"datetime", "net_pnl_fitted_model"}.issubset(rf_eval.columns):
        rf_port = make_portfolio_timeseries(rf_eval, "datetime", ["net_pnl_fitted_model"])
        ax.plot(rf_port["datetime"], rf_port["cumulative_net_pnl_fitted_model"], label="reduced_form eval")
        n_lines += 1
    ax.set_title("Portfolio fitted regression evaluator wealth")
    if n_lines:
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "pair_fitted_evaluator_wealth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if len(ow_eval) and len(rf_eval):
        merged = ow_eval[["stock", "datetime", "marginal_impact_bps"]].merge(
            rf_eval[["stock", "datetime", "marginal_impact_bps"]],
            on=["stock", "datetime"],
            suffixes=("_ow", "_rf"),
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(merged["marginal_impact_bps_ow"], merged["marginal_impact_bps_rf"], s=4, alpha=0.3)
        ax.set_xlabel("OW_transient marginal impact bps")
        ax.set_ylabel("reduced_form marginal impact bps")
        ax.set_title("Marginal impact comparison")
        fig.tight_layout()
        fig.savefig(fig_dir / "pair_marginal_impact_ow_vs_reduced_form.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if {"timestamp", "net_pnl", "gross_pnl"}.issubset(trades.columns) and len(ow_eval) and len(rf_eval):
        internal_port = make_portfolio_timeseries(trades, "timestamp", ["net_pnl", "gross_pnl"])
        ow_port = make_portfolio_timeseries(ow_eval, "datetime", ["net_pnl_fitted_model", "fitted_impact_cost_signed", "gross_pnl"])
        rf_port = make_portfolio_timeseries(rf_eval, "datetime", ["net_pnl_fitted_model", "fitted_impact_cost_signed"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(internal_port["timestamp"], internal_port["cumulative_net_pnl"], label="internal OW")
        ax.plot(ow_port["datetime"], ow_port["cumulative_net_pnl_fitted_model"], label="OW regression eval")
        ax.plot(rf_port["datetime"], rf_port["cumulative_net_pnl_fitted_model"], label="reduced-form eval")
        ax.set_title("Portfolio cumulative wealth: internal OW vs fitted evaluators")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "cumulative_wealth_internal_vs_fitted.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ow_port["datetime"], ow_port["cumulative_gross_pnl"], label="cumulative gross PnL")
        ax.plot(ow_port["datetime"], ow_port["cumulative_fitted_impact_cost_signed"], label="OW fitted cost")
        ax.plot(rf_port["datetime"], rf_port["cumulative_fitted_impact_cost_signed"], label="RF fitted cost")
        ax.set_title("Portfolio cumulative gross PnL vs fitted impact costs")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "gross_pnl_vs_fitted_cost_cumulative.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ow_eval["marginal_impact_bps"].clip(ow_eval["marginal_impact_bps"].quantile(0.005), ow_eval["marginal_impact_bps"].quantile(0.995)), bins=80, alpha=0.55, label="OW")
        ax.hist(rf_eval["marginal_impact_bps"].clip(rf_eval["marginal_impact_bps"].quantile(0.005), rf_eval["marginal_impact_bps"].quantile(0.995)), bins=80, alpha=0.55, label="RF")
        ax.set_title("Marginal Impact Bps Histogram")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "marginal_impact_bps_histogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        merged_cost = ow_eval[["gross_pnl", "fitted_impact_cost_signed"]].rename(columns={"fitted_impact_cost_signed": "cost_ow"})
        merged_cost["cost_rf"] = rf_eval["fitted_impact_cost_signed"].to_numpy()[: len(merged_cost)]
        fig, ax = plt.subplots(figsize=(7, 6))
        sample = merged_cost.sample(min(len(merged_cost), 10000), random_state=1) if len(merged_cost) else merged_cost
        ax.scatter(sample["gross_pnl"], sample["cost_rf"], s=4, alpha=0.25, label="RF cost")
        ax.scatter(sample["gross_pnl"], sample["cost_ow"], s=4, alpha=0.25, label="OW cost")
        ax.set_xlabel("row gross_pnl")
        ax.set_ylabel("row fitted impact cost")
        ax.set_title("Fitted Cost vs Gross PnL")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "fitted_cost_vs_gross_pnl_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        internal_wealth = internal_port["cumulative_net_pnl"]
        ow_wealth = ow_port["cumulative_net_pnl_fitted_model"]
        rf_wealth = rf_port["cumulative_net_pnl_fitted_model"]
        internal_dd = internal_wealth - internal_wealth.cummax()
        ow_dd = ow_wealth - ow_wealth.cummax()
        rf_dd = rf_wealth - rf_wealth.cummax()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(internal_port["timestamp"], internal_dd, label="internal")
        ax.plot(ow_port["datetime"], ow_dd, label="OW eval")
        ax.plot(rf_port["datetime"], rf_dd, label="RF eval")
        ax.set_title("Portfolio drawdown: internal OW vs fitted evaluators")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "drawdown_internal_vs_fitted.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        sample_key = trades.groupby(["stock", "date"], sort=False)["signed_volume"].apply(lambda x: x.abs().sum()).idxmax()
        sample_trades = trades.loc[(trades["stock"] == sample_key[0]) & (trades["date"].astype(str) == str(sample_key[1]))].copy()
        sample_ow = ow_eval.loc[(ow_eval["stock"] == sample_key[0]) & (ow_eval["trading_date"].astype(str) == str(sample_key[1]))].copy()
        sample_rf = rf_eval.loc[(rf_eval["stock"] == sample_key[0]) & (rf_eval["trading_date"].astype(str) == str(sample_key[1]))].copy()
        if len(sample_trades):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(pd.to_datetime(sample_trades["timestamp"]), sample_trades["alpha"], label="alpha")
            ax.plot(pd.to_datetime(sample_trades["timestamp"]), sample_trades["position_after"] / sample_trades["position_after"].abs().max(), label="position scaled")
            ax.plot(pd.to_datetime(sample_trades["timestamp"]), sample_trades["mid"] / sample_trades["mid"].iloc[0] - 1, label="mid normalized return")
            ax.set_title("Alpha / Position Alignment Sample")
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(fig_dir / "alpha_position_alignment_sample.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        if len(sample_ow) and len(sample_rf):
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(pd.to_datetime(sample_ow["datetime"]), sample_ow["marginal_impact_bps"], label="OW impact bps")
            ax1.plot(pd.to_datetime(sample_rf["datetime"]), sample_rf["marginal_impact_bps"], label="RF impact bps")
            ax2 = ax1.twinx()
            ax2.plot(pd.to_datetime(sample_ow["datetime"]), sample_ow["signed_volume"], color="gray", alpha=0.4, label="signed volume")
            ax1.set_title("Marginal Impact Bps Timeseries Sample")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(fig_dir / "marginal_impact_bps_timeseries_sample.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    stress_dir = Path(pair_dir) / "stress"
    hard_eval_path = stress_dir / "forced_liq_hard_block_fitted_evaluator.csv"
    capped_eval_path = stress_dir / "forced_liq_capped_residual_fitted_evaluator.csv"
    hard_events_path = stress_dir / "forced_liq_hard_block_events.csv"
    capped_events_path = stress_dir / "forced_liq_capped_residual_events.csv"
    if hard_eval_path.exists() and capped_eval_path.exists() and len(rf_eval):
        hard_eval = pd.read_csv(hard_eval_path)
        capped_eval = pd.read_csv(capped_eval_path)
        baseline_port = make_portfolio_timeseries(rf_eval, "datetime", ["net_pnl_fitted_model"])
        hard_port = make_portfolio_timeseries(hard_eval, "datetime", ["net_pnl_fitted_model_rf"])
        capped_port = make_portfolio_timeseries(capped_eval, "datetime", ["net_pnl_fitted_model_rf"])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(baseline_port["datetime"], baseline_port["cumulative_net_pnl_fitted_model"], label="baseline RF eval")
        ax.plot(hard_port["datetime"], hard_port["cumulative_net_pnl_fitted_model_rf"], label="hard block")
        ax.plot(capped_port["datetime"], capped_port["cumulative_net_pnl_fitted_model_rf"], label="capped residual")
        ax.set_title("Forced liquidation: hard block vs capped residual wealth")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / "forced_liq_hard_vs_capped_wealth.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if hard_events_path.exists() and capped_events_path.exists():
        hard_events = pd.read_csv(hard_events_path)
        capped_events = pd.read_csv(capped_events_path)
        if len(hard_events) or len(capped_events):
            hard_stock = hard_events.groupby("stock", as_index=False)["fitted_liquidation_cost_rf"].sum() if "fitted_liquidation_cost_rf" in hard_events else pd.DataFrame(columns=["stock", "fitted_liquidation_cost_rf"])
            capped_stock = capped_events.groupby("stock", as_index=False)["fitted_liquidation_cost_rf"].sum() if "fitted_liquidation_cost_rf" in capped_events else pd.DataFrame(columns=["stock", "fitted_liquidation_cost_rf"])
            cost = hard_stock.merge(capped_stock, on="stock", how="outer", suffixes=("_hard", "_capped")).fillna(0.0)
            if len(cost):
                x = np.arange(len(cost))
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.bar(x - 0.2, cost["fitted_liquidation_cost_rf_hard"], width=0.4, label="hard block")
                ax.bar(x + 0.2, cost["fitted_liquidation_cost_rf_capped"], width=0.4, label="capped residual")
                ax.set_xticks(x)
                ax.set_xticklabels(cost["stock"], rotation=90)
                ax.set_title("Forced liquidation fitted RF costs by stock")
                ax.legend()
                fig.tight_layout()
                fig.savefig(fig_dir / "forced_liq_event_costs_by_stock.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

            residual = capped_events[["stock", "residual_position_after_first_liquidation", "final_residual_position"]].copy() if len(capped_events) else pd.DataFrame()
            if len(residual):
                fig, ax = plt.subplots(figsize=(12, 5))
                x = np.arange(len(residual))
                ax.bar(x - 0.2, residual["residual_position_after_first_liquidation"], width=0.4, label="after first liquidation")
                ax.bar(x + 0.2, residual["final_residual_position"], width=0.4, label="final residual")
                ax.set_xticks(x)
                ax.set_xticklabels(residual["stock"], rotation=90)
                ax.set_title("Forced liquidation residual inventory by stock")
                ax.legend()
                fig.tight_layout()
                fig.savefig(fig_dir / "forced_liq_residual_inventory_by_stock.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

            if "time_to_liquidate_minutes" in capped_events and capped_events["time_to_liquidate_minutes"].notna().any():
                _hist(capped_events["time_to_liquidate_minutes"], "Capped Residual Time To Liquidate", "minutes", fig_dir / "forced_liq_time_to_liquidate.png")

            fig, ax = plt.subplots(figsize=(8, 5))
            if "max_liquidation_participation_rate" in hard_events:
                ax.hist(hard_events["max_liquidation_participation_rate"].dropna(), bins=30, alpha=0.55, label="hard block")
            if "max_liquidation_participation_rate" in capped_events:
                ax.hist(capped_events["max_liquidation_participation_rate"].dropna(), bins=30, alpha=0.55, label="capped residual")
            ax.set_title("Forced liquidation participation rates")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "forced_liq_participation_rates.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    hard_trades_path = stress_dir / "forced_liq_hard_block_trades.csv"
    if hard_trades_path.exists():
        hard_trades = pd.read_csv(hard_trades_path)
        liq_rows = hard_trades.loc[hard_trades.get("is_liquidation_trade", pd.Series(False, index=hard_trades.index)).fillna(False).astype(bool)]
        if len(liq_rows):
            sample_row = liq_rows.loc[liq_rows["signed_volume"].abs().idxmax()]
            date_col = "date" if "date" in hard_trades.columns else "trading_date"
            sample = hard_trades.loc[(hard_trades["stock"] == sample_row["stock"]) & (hard_trades[date_col].astype(str) == str(sample_row[date_col]))].copy()
            sample["timestamp"] = pd.to_datetime(sample["timestamp"], errors="coerce")
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(sample["timestamp"], sample["position_after"], label="position")
            ax1.bar(sample["timestamp"], sample["signed_volume"], alpha=0.25, label="signed volume")
            ax2 = ax1.twinx()
            ax2.plot(sample["timestamp"], sample["alpha"], color="green", label="alpha")
            ax2.plot(sample["timestamp"], sample["mid"] / sample["mid"].iloc[0] - 1.0, color="gray", label="mid return")
            ax1.set_title("Forced liquidation sample path")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(fig_dir / "forced_liq_sample_path.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def save_debug_strategy_path_sample(pair_dir: Path, trades: pd.DataFrame, ow_eval: pd.DataFrame, rf_eval: pd.DataFrame) -> Path | None:
    """Save row-level debug table for highest-turnover stock/day."""

    if trades.empty:
        return None
    pair_dir = Path(pair_dir)
    debug_dir = pair_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    key = trades.groupby(["stock", "date"], sort=False)["signed_volume"].apply(lambda x: x.abs().sum()).idxmax()
    sample = trades.loc[(trades["stock"] == key[0]) & (trades["date"].astype(str) == str(key[1]))].copy()
    sample["datetime"] = pd.to_datetime(sample["timestamp"])
    ow_cols = ["stock", "datetime", "orderFlow_market", "orderFlow_scenario", "pred_ret_bps_market", "pred_ret_bps_with_strategy", "marginal_impact_bps", "fitted_impact_cost_signed"]
    rf_cols = ow_cols
    if len(ow_eval):
        ow = ow_eval[[c for c in ow_cols if c in ow_eval.columns]].rename(columns={
            "pred_ret_bps_market": "pred_ret_bps_market_ow",
            "pred_ret_bps_with_strategy": "pred_ret_bps_with_trade_ow",
            "marginal_impact_bps": "marginal_impact_ow_bps",
            "fitted_impact_cost_signed": "fitted_cost_ow",
        })
        sample = sample.merge(ow, on=["stock", "datetime"], how="left")
    if len(rf_eval):
        rf = rf_eval[[c for c in rf_cols if c in rf_eval.columns]].rename(columns={
            "orderFlow_market": "orderFlow_market_rf",
            "orderFlow_scenario": "orderFlow_scenario_rf",
            "pred_ret_bps_market": "pred_ret_bps_market_rf",
            "pred_ret_bps_with_strategy": "pred_ret_bps_with_trade_rf",
            "marginal_impact_bps": "marginal_impact_rf_bps",
            "fitted_impact_cost_signed": "fitted_cost_rf",
        })
        sample = sample.merge(rf, on=["stock", "datetime"], how="left")
    for col in ["orderFlow_market", "orderFlow_scenario"]:
        alt = f"{col}_rf"
        if col not in sample.columns and alt in sample.columns:
            sample[col] = sample[alt]
    keep = [
        "timestamp", "stock", "date", "mid", "delta_mid", "alpha", "future_return_h",
        "target_impact", "impact_before_trade", "impact_after_trade", "signed_volume",
        "position_before", "position_after", "gross_pnl", "net_pnl", "orderFlow_market",
        "orderFlow_scenario", "pred_ret_bps_market_ow", "pred_ret_bps_with_trade_ow",
        "marginal_impact_ow_bps", "pred_ret_bps_market_rf", "pred_ret_bps_with_trade_rf",
        "marginal_impact_rf_bps", "fitted_cost_ow", "fitted_cost_rf",
    ]
    out = sample[[c for c in keep if c in sample.columns]].copy()
    path = debug_dir / "debug_strategy_path_sample.csv"
    out.to_csv(path, index=False)
    return path


def save_global_plots(output_dir: Path) -> None:
    """Save aggregate rolling plots."""

    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = output_dir / "all_pairs_strategy_summary.csv"
    wrong_path = output_dir / "all_pairs_wrong_model_summary.csv"
    proxy_path = output_dir / "all_pairs_fitted_proxy_summary.csv"
    if strategy_path.exists():
        df = pd.read_csv(strategy_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df["pair_id"].astype(str), df["total_net_pnl"])
        ax.set_title("Total net PnL by rolling pair")
        ax.set_xlabel("pair_id")
        fig.tight_layout()
        fig.savefig(fig_dir / "total_net_pnl_by_pair.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    if wrong_path.exists():
        df = pd.read_csv(wrong_path)
        if "degradation_rf_vs_ow" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df["pair_id"].astype(str), df["degradation_rf_vs_ow"])
            ax.set_title("Wrong-model degradation by pair")
            fig.tight_layout()
            fig.savefig(fig_dir / "wrong_model_degradation_by_pair.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    if strategy_path.exists() and proxy_path.exists():
        strategy = pd.read_csv(strategy_path)
        proxy = pd.read_csv(proxy_path)
        if len(strategy) and len(proxy):
            merged = strategy[["pair_id", "total_net_pnl"]].merge(
                proxy[["pair_id", "total_net_pnl_fitted_proxy"]],
                on="pair_id",
                how="inner",
            )
            if len(merged):
                x = np.arange(len(merged))
                width = 0.38
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(x - width / 2, merged["total_net_pnl"], width=width, label="OW target-impact")
                ax.bar(x + width / 2, merged["total_net_pnl_fitted_proxy"], width=width, label="fitted proxy")
                ax.set_xticks(x)
                ax.set_xticklabels(merged["pair_id"].astype(str))
                ax.set_title("OW vs Fitted Proxy Net PnL by Pair")
                ax.set_xlabel("pair_id")
                ax.legend()
                fig.tight_layout()
                fig.savefig(fig_dir / "ow_vs_fitted_proxy_net_pnl_by_pair.png", dpi=150, bbox_inches="tight")
                plt.close(fig)


def write_pair_report(
    pair_dir: Path,
    pair_id: int,
    pair_row: pd.Series,
    summary: dict,
    run_id: str | None = None,
    figures_cleaned: bool | None = None,
) -> None:
    """Write pair integration report."""

    proxy = _safe_read(Path(pair_dir) / "fitted_proxy_strategy_trades.csv")
    proxy_lines = ["No fitted proxy trades saved."]
    if len(proxy):
        proxy_lines = [
            "Fitted-regression proxy strategy:",
            "- This is a local myopic quadratic-cost proxy induced by teammate reduced_form regression slopes.",
            "- It is not a closed-form dynamic optimal strategy under a structural impact model.",
            f"- total_gross_pnl_fitted_proxy: {proxy['gross_pnl'].sum()}",
            f"- total_fitted_proxy_cost: {proxy['fitted_impact_cost'].sum()}",
            f"- total_net_pnl_fitted_proxy: {proxy['net_pnl'].sum()}",
            f"- total_turnover_fitted_proxy: {proxy['signed_volume'].abs().sum()}",
            f"- max_participation_rate_fitted_proxy: {proxy['participation_rate'].max()}",
        ]
    lines = [
        f"# Pair {pair_id} Integration Report",
        "",
        f"- run_id: {run_id}",
        f"- figures_cleaned: {figures_cleaned}",
        f"- train_month: {pair_row['train_month']}",
        f"- test_month: {pair_row['test_month']}",
        f"- stocks: {summary.get('n_stocks')}",
        f"- rows: {summary.get('n_rows')}",
        f"- total_net_pnl_my_ow: {summary.get('total_net_pnl')}",
        f"- max_participation_rate: {summary.get('max_participation_rate')}",
        f"- mean_participation_rate: {summary.get('mean_participation_rate')}",
        f"- max_abs_position: {summary.get('max_abs_position')}",
        "",
        "Sizing controls:",
        "- Previous unconstrained strategy variants can produce unrealistic trade sizes.",
        "- The reported integrated strategy uses participation, trade, and inventory caps before wealth is interpreted.",
        "- Fitted-model cost diagnostics should be read only after these caps are applied.",
        "",
        "Teammate reduced_form is treated as a fitted regression evaluator, not dynamic AFS.",
        "Wrong-model stress uses orderFlow_scenario = orderFlow_market + q_strategy.",
        "",
        *proxy_lines,
    ]
    Path(pair_dir).mkdir(parents=True, exist_ok=True)
    (Path(pair_dir) / f"pair_{pair_id}_integration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _alpha_diagnostics(trades: pd.DataFrame) -> dict[str, float]:
    valid = trades.get("valid_future_return", pd.Series(True, index=trades.index)).astype(bool)
    return {
        "alpha_mean_bps": float(trades["alpha"].mean() * 10000) if "alpha" in trades else np.nan,
        "alpha_std_bps": float(trades["alpha"].std(ddof=1) * 10000) if "alpha" in trades else np.nan,
        "corr_alpha_future_return": float(trades.loc[valid, "alpha"].corr(trades.loc[valid, "future_return_h"])) if {"alpha", "future_return_h"}.issubset(trades.columns) else np.nan,
        "corr_position_future_return": float(trades.loc[valid, "position_before"].corr(trades.loc[valid, "future_return_h"])) if {"position_before", "future_return_h"}.issubset(trades.columns) else np.nan,
        "corr_trade_alpha": float(trades["signed_volume"].corr(trades["alpha"])) if {"signed_volume", "alpha"}.issubset(trades.columns) else np.nan,
        "corr_position_alpha": float(trades["position_after"].corr(trades["alpha"])) if {"position_after", "alpha"}.issubset(trades.columns) else np.nan,
        "share_sign_position_matches_alpha": float((np.sign(trades.loc[trades["position_before"].abs() > 0, "position_before"]) == np.sign(trades.loc[trades["position_before"].abs() > 0, "alpha"])).mean()) if {"position_before", "alpha"}.issubset(trades.columns) and (trades["position_before"].abs() > 0).any() else np.nan,
        "share_sign_trade_matches_alpha": float((np.sign(trades.loc[trades["signed_volume"].abs() > 0, "signed_volume"]) == np.sign(trades.loc[trades["signed_volume"].abs() > 0, "alpha"])).mean()) if {"signed_volume", "alpha"}.issubset(trades.columns) and (trades["signed_volume"].abs() > 0).any() else np.nan,
        "gross_alpha_capture": float(trades["gross_pnl"].sum()) if "gross_pnl" in trades else np.nan,
    }


def _sizing_diagnostics(trades: pd.DataFrame) -> dict[str, float]:
    pos_over_adv = trades["position_after"].abs() / trades["ADV"].replace(0, np.nan) if {"position_after", "ADV"}.issubset(trades.columns) else pd.Series(dtype=float)
    return {
        "mean_abs_trade": float(trades["signed_volume"].abs().mean()),
        "median_abs_trade": float(trades["signed_volume"].abs().median()),
        "max_abs_trade": float(trades["signed_volume"].abs().max()),
        "max_abs_position": float(trades["position_after"].abs().max()),
        "total_signed_volume_turnover": float(trades["signed_volume"].abs().sum()),
        "total_notional_turnover": float(trades["signed_volume_notional"].abs().sum()) if "signed_volume_notional" in trades else np.nan,
        "max_participation_rate": float(trades["participation_rate"].max(skipna=True)),
        "mean_participation_rate": float(trades["participation_rate"].mean(skipna=True)),
        "share_trade_clipped": float(trades.get("trade_clipped", pd.Series(False, index=trades.index)).astype(bool).mean()),
        "share_position_clipped": float(trades.get("position_clipped", pd.Series(False, index=trades.index)).astype(bool).mean()),
        "max_position_over_ADV": float(pos_over_adv.max(skipna=True)) if len(pos_over_adv) else np.nan,
    }


def _eval_summary(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    if df.empty:
        return {}
    traded = df["signed_volume"].abs() > 0
    return {
        f"total_fitted_cost_{prefix}": float(df["fitted_impact_cost_signed"].sum()),
        f"total_abs_fitted_cost_{prefix}": float(df["fitted_impact_cost_abs"].sum()),
        f"net_pnl_under_{prefix}_eval": float(df["net_pnl_fitted_model"].sum()),
        f"mean_abs_marginal_impact_{prefix}_bps": float(df["marginal_impact_bps"].abs().mean()),
        f"share_same_direction_trade_impact_{prefix}": float((np.sign(df.loc[traded, "signed_volume"]) == np.sign(df.loc[traded, "marginal_impact_bps"])).mean()) if traded.any() else np.nan,
        f"share_positive_signed_cost_{prefix}": float((df.loc[traded, "fitted_impact_cost_signed"] > 0).mean()) if traded.any() else np.nan,
    }


def write_global_report(
    output_dir: Path,
    pair_ids: list[int],
    run_config: object | None = None,
    run_id: str | None = None,
    figures_cleaned: bool | None = None,
    run_archive_dir: Path | None = None,
) -> None:
    """Write deep global integrated rolling report."""

    output_dir = Path(output_dir)
    strategy = pd.read_csv(output_dir / "all_pairs_strategy_summary.csv") if (output_dir / "all_pairs_strategy_summary.csv").exists() else pd.DataFrame()
    wrong = pd.read_csv(output_dir / "all_pairs_wrong_model_summary.csv") if (output_dir / "all_pairs_wrong_model_summary.csv").exists() else pd.DataFrame()
    proxy_summary = _safe_read(output_dir / "all_pairs_fitted_proxy_summary.csv")
    first_pair = pair_ids[0] if pair_ids else None
    pair_dir = output_dir / f"pair_{first_pair}" if first_pair is not None else output_dir / "pair_unknown"
    trades = _safe_read(pair_dir / "my_ow_trades.csv")
    ow_eval = _safe_read(pair_dir / "ow_transient_regression_evaluator.csv")
    rf_eval = _safe_read(pair_dir / "reduced_form_regression_evaluator.csv")
    proxy_trades = _safe_read(pair_dir / "fitted_proxy_strategy_trades.csv")
    stress = _safe_read(output_dir / "all_pairs_stress_summary.csv")
    sensitivity = _safe_read(output_dir / "all_pairs_sensitivity_summary.csv")
    validation = _safe_read(output_dir / "integrated_validation_checks.csv")
    alpha_diag = _alpha_diagnostics(trades) if len(trades) else {}
    sizing = _sizing_diagnostics(trades) if len(trades) else {}
    ow_summary = _eval_summary(ow_eval, "ow_regression")
    rf_summary = _eval_summary(rf_eval, "reduced_form")
    strategy_first = strategy.iloc[0].to_dict() if len(strategy) else {}
    rows_report = int(strategy_first.get("n_rows", strategy_first.get("rows", len(trades))) or 0)
    stocks_report = int(strategy_first.get("n_stocks", trades["stock"].nunique() if "stock" in trades else 0) or 0)
    n_stock_days = int(strategy_first.get("number_of_stock_days", 0) or 0)
    if "number_of_unique_dates" in strategy_first and pd.notna(strategy_first["number_of_unique_dates"]):
        n_dates = int(strategy_first["number_of_unique_dates"])
    elif stocks_report > 0 and n_stock_days > 0:
        n_dates = int(round(n_stock_days / stocks_report))
    else:
        n_dates = int(trades["date"].nunique()) if "date" in trades else 0
    truncated = getattr(run_config, "max_rows_per_pair", None) is not None if run_config is not None else False
    save_trades = bool(getattr(run_config, "save_pair_level_trades", False)) if run_config is not None else len(trades) > 0
    max_part_cap = getattr(run_config, "max_participation_rate_per_trade", None) if run_config is not None else None
    gross = float(trades["gross_pnl"].sum()) if len(trades) and "gross_pnl" in trades else float(strategy_first.get("total_gross_pnl", np.nan))
    internal_cost = float(trades["signed_impact_cost_normalized"].sum()) if len(trades) and "signed_impact_cost_normalized" in trades else np.nan
    internal_ratio = abs(internal_cost) / abs(gross) if np.isfinite(gross) and gross != 0 and np.isfinite(internal_cost) else np.nan
    ow_net = ow_summary.get("net_pnl_under_ow_regression_eval", np.nan)
    rf_net = rf_summary.get("net_pnl_under_reduced_form_eval", np.nan)
    proxy_first = proxy_summary.iloc[0].to_dict() if len(proxy_summary) else {}

    max_part_observed = sizing.get("max_participation_rate", strategy_first.get("max_participation_rate", np.inf))
    sizing_status = "PASS" if max_part_cap is None or max_part_observed <= max_part_cap + 1e-9 else "WARN"
    fail_count = int((validation["status"] == "FAIL").sum()) if len(validation) and "status" in validation else 0
    pnl_status = "PASS" if len(validation) and fail_count == 0 else ("WARN" if len(validation) else "WARN")
    evaluator_status = "PASS" if len(ow_eval) and len(rf_eval) else "WARN"
    integration_status = "PASS" if len(strategy) else "WARN"
    std_daily = float(strategy_first.get("std_daily_net_pnl", np.nan))
    daily_sharpe_meaningful = (not truncated) and n_dates >= 2 and np.isfinite(std_daily) and std_daily > 0
    gross_msg = "The strategy captures alpha before fitted impact costs." if gross > 0 else "The strategy may not be aligned with alpha or timing may be wrong."
    fitted_msg = ""
    if gross > 0 and (ow_net < 0 or rf_net < 0):
        fitted_msg = "The alpha strategy is profitable before fitted impact costs, but fitted regression impact costs more than offset gross PnL on this sample."
    rf_vs_ow = ""
    if np.isfinite(rf_net) and np.isfinite(ow_net):
        rf_vs_ow = "Reduced-form evaluator is less punitive than OW_transient on this sample." if rf_net > ow_net else "Reduced-form evaluator is more punitive than OW_transient on this sample."
    figure_paths = sorted((pair_dir / "figures").glob("*.png")) + sorted((output_dir / "figures").glob("*.png"))
    sizing_report = {
        "mean_abs_trade": sizing.get("mean_abs_trade", strategy_first.get("mean_abs_trade")),
        "median_abs_trade": sizing.get("median_abs_trade", strategy_first.get("median_abs_trade")),
        "max_abs_trade": sizing.get("max_abs_trade", strategy_first.get("max_abs_trade")),
        "max_abs_position": sizing.get("max_abs_position", strategy_first.get("max_abs_position")),
        "total_signed_volume_turnover": sizing.get("total_signed_volume_turnover", strategy_first.get("total_signed_volume_turnover")),
        "total_notional_turnover": sizing.get("total_notional_turnover", strategy_first.get("total_notional_turnover")),
        "max_participation_rate": sizing.get("max_participation_rate", strategy_first.get("max_participation_rate")),
        "mean_participation_rate": sizing.get("mean_participation_rate", strategy_first.get("mean_participation_rate")),
        "share_trade_clipped": sizing.get("share_trade_clipped", np.nan),
        "share_position_clipped": sizing.get("share_position_clipped", np.nan),
        "max_position_over_ADV": sizing.get("max_position_over_ADV", np.nan),
    }
    wrong_model_rows = wrong.copy() if len(wrong) else pd.DataFrame()
    if wrong_model_rows.empty and len(stress) and "scenario_name" in stress.columns:
        wrong_model_rows = stress.loc[
            stress["scenario_name"].astype(str).str.contains("wrong_model|wrong_impact", na=False)
        ].copy()
    signal_delay_rows = (
        stress.loc[stress["scenario_name"].astype(str).str.contains("signal_delay", na=False)].copy()
        if len(stress) and "scenario_name" in stress.columns
        else pd.DataFrame()
    )
    forced_liq_rows = (
        stress.loc[stress["scenario_name"].astype(str).str.contains("forced_liq", na=False)].copy()
        if len(stress) and "scenario_name" in stress.columns
        else pd.DataFrame()
    )
    show_proxy = len(proxy_summary) > 0 and any(pd.notna(proxy_first.get(col)) for col in [
        "total_net_pnl_fitted_proxy",
        "total_fitted_proxy_cost",
        "total_turnover_fitted_proxy",
    ])

    lines = [
        "# Integrated Rolling Simulation Report",
        "",
        "## 1. Executive Summary",
        f"- Run ID: {run_id}",
        f"- Figures cleaned before run: {figures_cleaned}",
        f"- Run-specific archive: {run_archive_dir if run_archive_dir is not None else 'not created'}",
        f"- Pair IDs: {pair_ids}",
        f"- Truncated debug run: {truncated}",
        f"- Row-level trade files saved: {save_trades}",
        f"- Rows: {rows_report}",
        f"- Stocks: {stocks_report}",
        f"- Dates: {n_dates}",
        f"- Stock-days: {n_stock_days}",
        f"- Daily Sharpe meaningful: {'YES' if daily_sharpe_meaningful else 'NO, only one date, zero daily variance, or truncated debug sample.'}",
        f"- Integration mechanics: {integration_status}",
        f"- Sizing plausibility: {sizing_status}",
        f"- PnL formula checks: {pnl_status}",
        f"- Fitted evaluator checks: {evaluator_status}",
        "",
        "## 2. Data and Sample",
        "- Train/test months and stock universes are read from teammate rolling_pair_summary.csv.",
        "- Row-level trade files were not saved. Portfolio path reconciliation checks that require row-level trades are skipped, while summary-level metrics are still reported." if not save_trades else "- Row-level trade files were saved; portfolio reconciliation checks use saved row-level files.",
        strategy.to_string(index=False) if len(strategy) else "No strategy summary.",
        "- Daily Sharpe is meaningful for this run." if daily_sharpe_meaningful else "- Warning: daily Sharpe is undefined or not meaningful when only one date is present, daily variance is zero, or max_rows_per_pair truncates the run.",
        "",
        "## 3. Alpha Diagnostics",
        "- Baseline synthetic alpha: h=5m, rho=0.10, H_alpha=5m unless configured otherwise.",
        f"- alpha_mean_bps: {_fmt(alpha_diag.get('alpha_mean_bps'))}",
        f"- alpha_std_bps: {_fmt(alpha_diag.get('alpha_std_bps'))}",
        f"- corr(alpha, future_return_h): {_fmt(alpha_diag.get('corr_alpha_future_return'))}",
        f"- corr(position_before, future_return_h): {_fmt(alpha_diag.get('corr_position_future_return'))}",
        f"- corr(trade, alpha): {_fmt(alpha_diag.get('corr_trade_alpha'))}",
        f"- corr(position_after, alpha): {_fmt(alpha_diag.get('corr_position_alpha'))}",
        f"- share_sign_position_matches_alpha: {_fmt(alpha_diag.get('share_sign_position_matches_alpha'))}",
        f"- share_sign_trade_matches_alpha: {_fmt(alpha_diag.get('share_sign_trade_matches_alpha'))}",
        f"- gross alpha capture: {_fmt(alpha_diag.get('gross_alpha_capture'))}",
        f"- Interpretation: {gross_msg}",
        "",
        "## 4. Strategy Sizing Diagnostics",
        f"- mean_abs_trade: {_fmt(sizing_report.get('mean_abs_trade'))}",
        f"- median_abs_trade: {_fmt(sizing_report.get('median_abs_trade'))}",
        f"- max_abs_trade: {_fmt(sizing_report.get('max_abs_trade'))}",
        f"- max_abs_position: {_fmt(sizing_report.get('max_abs_position'))}",
        f"- total_signed_volume_turnover: {_fmt(sizing_report.get('total_signed_volume_turnover'))}",
        f"- total_notional_turnover: {_fmt(sizing_report.get('total_notional_turnover'))}",
        f"- max_participation_rate: {_fmt(sizing_report.get('max_participation_rate'))}",
        f"- mean_participation_rate: {_fmt(sizing_report.get('mean_participation_rate'))}",
        f"- share_trade_clipped: {_fmt(sizing_report.get('share_trade_clipped'))}",
        f"- share_position_clipped: {_fmt(sizing_report.get('share_position_clipped'))}",
        f"- max_position_over_ADV: {_fmt(sizing_report.get('max_position_over_ADV'))}",
        "- Previous unconstrained runs were unrealistic. The capped runs are the reportable ones.",
        "",
        "## 5. OW Internal Wealth",
        f"- total_gross_pnl: {_fmt(gross)}",
        f"- total_internal_impact_cost_normalized: {_fmt(internal_cost)}",
        f"- total_net_pnl_internal: {_fmt(trades['net_pnl'].sum() if len(trades) else np.nan)}",
        f"- internal_impact_cost_to_gross_pnl_ratio: {_fmt(internal_ratio)}",
        f"- max_drawdown: {_fmt(strategy['max_drawdown'].iloc[0] if len(strategy) and 'max_drawdown' in strategy else np.nan)}",
        "- Internal OW costs are normalized diagnostics and are not on the same economic scale as fitted-regression price-unit costs.",
        "- Warning: internal normalized impact costs are tiny relative to gross PnL." if np.isfinite(internal_ratio) and internal_ratio < 1e-4 else "- Internal cost scale warning not triggered.",
        "",
        "## 6. Fitted Regression Evaluator Wealth",
        f"- total_fitted_cost_ow_regression: {_fmt(ow_summary.get('total_fitted_cost_ow_regression'))}",
        f"- total_abs_fitted_cost_ow_regression: {_fmt(ow_summary.get('total_abs_fitted_cost_ow_regression'))}",
        f"- net_pnl_under_ow_regression_eval: {_fmt(ow_summary.get('net_pnl_under_ow_regression_eval'))}",
        f"- mean_abs_marginal_impact_ow_regression_bps: {_fmt(ow_summary.get('mean_abs_marginal_impact_ow_regression_bps'))}",
        f"- total_fitted_cost_reduced_form: {_fmt(rf_summary.get('total_fitted_cost_reduced_form'))}",
        f"- total_abs_fitted_cost_reduced_form: {_fmt(rf_summary.get('total_abs_fitted_cost_reduced_form'))}",
        f"- net_pnl_under_reduced_form_eval: {_fmt(rf_summary.get('net_pnl_under_reduced_form_eval'))}",
        f"- mean_abs_marginal_impact_reduced_form_bps: {_fmt(rf_summary.get('mean_abs_marginal_impact_reduced_form_bps'))}",
        f"- Interpretation: {fitted_msg or 'Fitted impact costs do not overturn gross PnL on this sample.'}",
        f"- Model comparison: {rf_vs_ow}",
        "",
    ]
    if show_proxy:
        lines.extend([
        "## 7. Fitted-Regression Proxy Strategy",
        "- Because teammate's fitted reduced_form model is an OLS regression, not a structural impact dynamics, it does not provide a closed-form dynamic optimal strategy.",
        "- I therefore implement a local myopic quadratic-cost proxy using the regression marginal slope with respect to strategy order flow.",
        "- The proxy uses orderFlow_scenario = orderFlow_market + q_strategy and estimates d(ret_bps)/dq from x_flow, x_flow_depth, and optionally x_trade.",
        "- Objective: maximize q * alpha_price - impact_slope_price_per_share * q^2 - inventory_penalty * (Q_prev + q)^2.",
        "- This is reportable as a fitted-regression-aware proxy strategy, not as the course structural AFS optimum.",
        f"- total_gross_pnl_fitted_proxy: {_fmt(proxy_first.get('total_gross_pnl_fitted_proxy'))}",
        f"- total_fitted_proxy_cost: {_fmt(proxy_first.get('total_fitted_proxy_cost'))}",
        f"- total_net_pnl_fitted_proxy: {_fmt(proxy_first.get('total_net_pnl_fitted_proxy'))}",
        f"- total_turnover_fitted_proxy: {_fmt(proxy_first.get('total_turnover_fitted_proxy'))}",
        f"- max_drawdown_fitted_proxy: {_fmt(proxy_first.get('max_drawdown_fitted_proxy'))}",
        f"- max_participation_rate_fitted_proxy: {_fmt(proxy_first.get('max_participation_rate_fitted_proxy'))}",
        f"- mean_participation_rate_fitted_proxy: {_fmt(proxy_first.get('mean_participation_rate_fitted_proxy'))}",
        f"- rows in first pair proxy trade table: {len(proxy_trades)}",
        "",
        ])
    lines.extend([
        "## 8. PnL Formula and Timing Validation",
        validation.to_string(index=False) if len(validation) else "Validation checks were not available.",
        "",
    ])
    if len(wrong_model_rows):
        lines.extend([
        "## 9. Wrong Model Stress",
        wrong_model_rows.to_string(index=False),
        "",
        ])
    if len(signal_delay_rows):
        lines.extend([
        "## 10. Signal Delay Stress",
        signal_delay_rows.to_string(index=False),
        "",
        ])
    if len(forced_liq_rows):
        lines.extend([
        "## 11. Forced Liquidation Stress: Hard Block vs Capped Residual",
        "- Hard block liquidation forces Q -> 0 immediately at the liquidation timestamp and may violate participation caps by design.",
        "- Capped residual liquidation respects liquidation participation caps. If the position cannot be fully liquidated, residual inventory remains and liquidation trades have priority over new alpha trades until cleared.",
        "- Hard block answers the exact Section 2.7 single-block stress question; capped residual is the operationally realistic variant with inventory carry risk.",
        forced_liq_rows.to_string(index=False),
        "",
        ])
    if len(sensitivity):
        lines.extend([
        "## 12. Sizing Sensitivity",
        sensitivity.head(20).to_string(index=False),
        "",
        ])
    lines.extend([
        "## 13. Figures",
        "- Portfolio-level plots aggregate PnL and costs across all stocks at each timestamp before taking cumulative sums.",
        "- Sample path plots explicitly use one selected stock/day only.",
        "\n".join(f"- {p}" for p in figure_paths) if figure_paths else "No figures found.",
        "",
        "## 14. Caveats",
        "- orderFlow_scenario = orderFlow_market + q_strategy.",
        "- marginal impact bps = predicted_with_trade - predicted_market.",
        "- Positive gross PnL means the strategy captures alpha before costs. Negative fitted net PnL can occur if model-implied impact costs exceed this gross alpha capture; that is not automatically a bug.",
        "- Teammate coefficients are regression coefficients, not structural lambda/beta.",
        "- reduced_form is regression, not dynamic AFS.",
        "- x_flow is not silently treated as structural lambda.",
        "- Costs are model-implied diagnostics.",
        "- Reportable final results require full test months, not a max_rows debug sample.",
    ])
    text = "\n".join(lines) + "\n"
    (output_dir / "integrated_rolling_report.md").write_text(text, encoding="utf-8")
    (output_dir / "integrated_rolling_report.txt").write_text(text, encoding="utf-8")
