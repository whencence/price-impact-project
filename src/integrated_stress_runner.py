"""Rolling-pair aware stress tests using teammate fitted regression evaluators."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.integration_config import IntegratedRunConfig
from src.fitted_regression_proxy_strategy import (
    FittedRegressionProxyConfig,
    run_fitted_regression_proxy_strategy,
)
from src.integrated_ow_runner import prepare_integrated_ow_strategy_input, run_my_ow_strategy_on_pair
from src.reduced_form_regression_evaluator import evaluate_marginal_impact_from_strategy_trades
from src.stress_config import StressTestConfig
from src.stress_tests import apply_clock_time_signal_delay, run_ow_strategy_with_forced_liquidation
from src.ow_strategy import run_ow_strategy
from src.strategy_metrics import compute_overall_strategy_summary


def _strategy_model_to_fitted_model(strategy_model: str) -> str:
    if strategy_model == "OW_transient_proxy":
        return "OW_transient"
    if strategy_model == "reduced_form_proxy":
        return "reduced_form"
    return "theoretical_OW_legacy"


def _scale_alpha_for_proxy(alpha_df: pd.DataFrame, run_config: IntegratedRunConfig) -> pd.DataFrame:
    out = alpha_df.copy()
    scale = float(run_config.alpha_scale) * float(run_config.target_impact_scale)
    if scale != 1.0 and "alpha_for_strategy" in out.columns:
        out["alpha_for_strategy"] = pd.to_numeric(out["alpha_for_strategy"], errors="coerce").fillna(0.0) * scale
    return out


def _run_proxy_strategy_for_stress(
    pair_row: pd.Series,
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    ow_params_df: pd.DataFrame,
    reduced_form_params_df: pd.DataFrame,
    run_config: IntegratedRunConfig,
) -> pd.DataFrame:
    """Run the selected fitted proxy strategy for stress scenarios."""

    model_name = _strategy_model_to_fitted_model(run_config.strategy_model)
    if model_name == "theoretical_OW_legacy":
        stocks = sorted(test_raw_df["stock"].astype(str).unique())
        return run_my_ow_strategy_on_pair(pair_row, train_raw_df, test_raw_df, stocks, alpha_df, ow_params_df, run_config)
    params = ow_params_df if model_name == "OW_transient" else reduced_form_params_df
    cfg = FittedRegressionProxyConfig(
        model_name=model_name,
        max_participation_rate_per_trade=run_config.max_participation_rate_per_trade,
        max_abs_trade_adv_fraction=run_config.max_abs_trade_adv_fraction,
        max_abs_position_adv_fraction=run_config.max_abs_position_adv_fraction,
    )
    trades = run_fitted_regression_proxy_strategy(
        train_raw_df,
        test_raw_df,
        _scale_alpha_for_proxy(alpha_df, run_config),
        params,
        int(pair_row["pair_id"]),
        cfg,
    )
    trades["pair_id"] = int(pair_row["pair_id"])
    trades["strategy_model"] = run_config.strategy_model
    trades["reportable_strategy"] = run_config.strategy_model != "theoretical_OW_legacy"
    return trades


def summarize_regression_evaluator(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Summarize fitted-regression evaluator output."""

    return {
        f"total_fitted_cost_{prefix}": float(df["fitted_impact_cost_signed"].sum()),
        f"total_abs_fitted_cost_{prefix}": float(df["fitted_impact_cost_abs"].sum()),
        f"net_pnl_under_{prefix}_eval": float(df["net_pnl_fitted_model"].sum()),
        f"mean_abs_marginal_impact_{prefix}_bps": float(df["marginal_impact_bps"].abs().mean()),
    }


def _portfolio_timeseries(df: pd.DataFrame, timestamp_col: str, cols: list[str]) -> pd.DataFrame:
    """Aggregate rows across stocks by timestamp before cumulative sums."""

    out = df[[timestamp_col, *cols]].copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    grouped = out.groupby(timestamp_col, as_index=False, sort=True)[cols].sum()
    for col in cols:
        grouped[f"cumulative_{col}"] = grouped[col].cumsum()
    return grouped


def _evaluator_cell_summary(
    pair_id: int,
    strategy_model: str,
    evaluator_model: str,
    trades: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict[str, float | str | int]:
    """Summarize one strategy/evaluator cell in the wrong-model matrix."""

    turnover = float(pd.to_numeric(trades.get("notional_turnover", trades["signed_volume"].abs() * trades["mid"]), errors="coerce").sum())
    cost = float(eval_df["fitted_impact_cost_signed"].sum())
    daily = eval_df.groupby("trading_date", sort=True)["net_pnl_fitted_model"].sum()
    sharpe = daily.mean() / daily.std(ddof=1) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    wealth = _portfolio_timeseries(eval_df, "datetime", ["net_pnl_fitted_model"])
    drawdown = wealth["cumulative_net_pnl_fitted_model"] - wealth["cumulative_net_pnl_fitted_model"].cummax() if len(wealth) else pd.Series(dtype=float)
    return {
        "pair_id": pair_id,
        "strategy_model": strategy_model,
        "assumed_model": strategy_model.replace("_proxy", ""),
        "evaluator_model": evaluator_model,
        "true_model": evaluator_model,
        "total_gross_pnl": float(eval_df["gross_pnl"].sum()),
        "total_fitted_cost": cost,
        "total_abs_fitted_cost": float(eval_df["fitted_impact_cost_abs"].sum()),
        "net_pnl": float(eval_df["net_pnl_fitted_model"].sum()),
        "total_turnover": float(trades["signed_volume"].abs().sum()),
        "total_notional_turnover": turnover,
        "cost_bps_of_turnover": 10000.0 * cost / turnover if turnover > 0 else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "daily_sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "mean_abs_marginal_impact_bps": float(eval_df["marginal_impact_bps"].abs().mean()),
        "max_participation_rate": float(pd.to_numeric(trades.get("participation_rate"), errors="coerce").max(skipna=True)),
        "mean_participation_rate": float(pd.to_numeric(trades.get("participation_rate"), errors="coerce").mean(skipna=True)),
    }


def _plot_wrong_model_outputs(output_dir: Path, matrix: pd.DataFrame, evals: dict[tuple[str, str], pd.DataFrame], sensitivity: pd.DataFrame) -> None:
    """Save wrong-model stress figures."""

    fig_dir = output_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if len(matrix):
        pivot = matrix.pivot(index="strategy_model", columns="evaluator_model", values="net_pnl")
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2e}", ha="center", va="center", fontsize=8)
        ax.set_title("Wrong-model net PnL matrix")
        fig.colorbar(im, ax=ax, label="net PnL")
        fig.tight_layout()
        fig.savefig(fig_dir / "wrong_model_net_pnl_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        cost = matrix.pivot(index="strategy_model", columns="evaluator_model", values="total_fitted_cost")
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cost.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_xticks(np.arange(len(cost.columns)))
        ax.set_xticklabels(cost.columns)
        ax.set_yticks(np.arange(len(cost.index)))
        ax.set_yticklabels(cost.index)
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                ax.text(j, i, f"{cost.iloc[i, j]:.2e}", ha="center", va="center", fontsize=8)
        ax.set_title("Wrong-model fitted cost matrix")
        fig.colorbar(im, ax=ax, label="fitted cost")
        fig.tight_layout()
        fig.savefig(fig_dir / "wrong_model_cost_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        turn = matrix.drop_duplicates("strategy_model").copy()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(turn["strategy_model"], turn["total_turnover"])
        ax.set_title("Wrong-model turnover by strategy")
        ax.set_ylabel("shares")
        fig.tight_layout()
        fig.savefig(fig_dir / "wrong_model_turnover_by_strategy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (strategy_model, evaluator_model), eval_df in evals.items():
        ts = _portfolio_timeseries(eval_df, "datetime", ["net_pnl_fitted_model"])
        if len(ts):
            ax.plot(ts["datetime"], ts["cumulative_net_pnl_fitted_model"], label=f"{strategy_model} under {evaluator_model}")
    ax.set_title("Wrong-model cumulative wealth matrix")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "wrong_model_cumulative_wealth_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if len(sensitivity):
        fig, ax = plt.subplots(figsize=(10, 5))
        for col, label in [
            ("net_pnl_under_ow_regression_eval", "OW_transient evaluator"),
            ("net_pnl_under_reduced_form_eval", "reduced_form evaluator"),
        ]:
            if col in sensitivity.columns:
                ax.bar(label, float(sensitivity[col].iloc[0]))
        ax.set_title("Impact evaluator sensitivity: same reportable trade path")
        ax.set_ylabel("net PnL")
        fig.tight_layout()
        fig.savefig(fig_dir / "impact_evaluator_sensitivity_same_trade_path.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_wrong_model_stress_pair(
    pair_row: pd.Series,
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    ow_trades_df: pd.DataFrame,
    ow_params_df: pd.DataFrame,
    reduced_form_params_df: pd.DataFrame,
    run_config: IntegratedRunConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run evaluator sensitivity and fitted-proxy wrong-model stress matrix."""

    pair_id = int(pair_row["pair_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = output_dir.parent / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: same reportable trade path under both fitted evaluators. This is
    # useful evaluator sensitivity, not full wrong-model strategy
    # misspecification.
    ow_eval = evaluate_marginal_impact_from_strategy_trades(
        train_raw_df, test_raw_df, ow_trades_df, ow_params_df, pair_id, model_name="OW_transient"
    )
    rf_eval = evaluate_marginal_impact_from_strategy_trades(
        train_raw_df, test_raw_df, ow_trades_df, reduced_form_params_df, pair_id, model_name="reduced_form"
    )
    keys = ["stock", "trading_date", "datetime"]
    merged = ow_eval.merge(
        rf_eval[keys + ["marginal_impact_bps", "fitted_impact_cost_signed", "net_pnl_fitted_model", "cumulative_wealth_fitted_model"]],
        on=keys,
        how="left",
        suffixes=("_ow", "_rf"),
    )
    merged["marginal_impact_diff_bps"] = merged["marginal_impact_bps_rf"] - merged["marginal_impact_bps_ow"]
    merged["fitted_cost_diff"] = merged["fitted_impact_cost_signed_rf"] - merged["fitted_impact_cost_signed_ow"]
    merged.to_csv(output_dir / "wrong_model_regression_evaluator.csv", index=False)
    merged.to_csv(output_dir / "impact_evaluator_sensitivity.csv", index=False)
    sensitivity_summary = {
        "pair_id": pair_id,
        "scenario_name": f"impact_evaluator_sensitivity_same_{run_config.strategy_model}_trade_path",
        "scenario_type": "impact_evaluator_sensitivity",
        "fixed_path_strategy_model": run_config.strategy_model,
        **summarize_regression_evaluator(ow_eval, "ow_regression"),
        **summarize_regression_evaluator(rf_eval, "reduced_form"),
        "total_cost_difference_rf_minus_ow": float(merged["fitted_cost_diff"].sum()),
        "net_pnl_difference_rf_minus_ow": float(rf_eval["net_pnl_fitted_model"].sum() - ow_eval["net_pnl_fitted_model"].sum()),
        "degradation_rf_vs_ow": float(rf_eval["net_pnl_fitted_model"].sum() - ow_eval["net_pnl_fitted_model"].sum()),
        "corr_between_ow_and_rf_marginal_impact": float(merged[["marginal_impact_bps_ow", "marginal_impact_bps_rf"]].corr().iloc[0, 1]),
    }
    pd.DataFrame([sensitivity_summary]).to_csv(output_dir / "impact_evaluator_sensitivity_summary.csv", index=False)

    mode = getattr(run_config, "wrong_model_mode", "both")
    if mode == "evaluator_sensitivity":
        _plot_wrong_model_outputs(output_dir, pd.DataFrame(), {}, pd.DataFrame([sensitivity_summary]))
        return merged, sensitivity_summary

    proxy_cfg_common = {
        "alpha_col": "alpha" if "alpha" in ow_trades_df.columns else "alpha_for_strategy",
        "max_participation_rate_per_trade": run_config.max_participation_rate_per_trade,
        "max_abs_trade_adv_fraction": run_config.max_abs_trade_adv_fraction,
        "max_abs_position_adv_fraction": run_config.max_abs_position_adv_fraction,
    }
    ow_proxy = run_fitted_regression_proxy_strategy(
        train_raw_df,
        test_raw_df,
        ow_trades_df,
        ow_params_df,
        pair_id,
        FittedRegressionProxyConfig(model_name="OW_transient", **proxy_cfg_common),
    )
    rf_proxy = run_fitted_regression_proxy_strategy(
        train_raw_df,
        test_raw_df,
        ow_trades_df,
        reduced_form_params_df,
        pair_id,
        FittedRegressionProxyConfig(model_name="reduced_form", **proxy_cfg_common),
    )
    ow_proxy.to_csv(trades_dir / "fitted_proxy_strategy_OW_transient.csv", index=False)
    rf_proxy.to_csv(trades_dir / "fitted_proxy_strategy_reduced_form.csv", index=False)
    ow_proxy.to_csv(output_dir / "fitted_proxy_strategy_OW_transient.csv", index=False)
    rf_proxy.to_csv(output_dir / "fitted_proxy_strategy_reduced_form.csv", index=False)

    evals: dict[tuple[str, str], pd.DataFrame] = {}
    matrix_rows: list[dict] = []
    for strategy_model, trades_df in [
        ("OW_transient_proxy", ow_proxy),
        ("reduced_form_proxy", rf_proxy),
    ]:
        for evaluator_model, params_df in [
            ("OW_transient", ow_params_df),
            ("reduced_form", reduced_form_params_df),
        ]:
            eval_df = evaluate_marginal_impact_from_strategy_trades(
                train_raw_df,
                test_raw_df,
                trades_df,
                params_df,
                pair_id,
                model_name=evaluator_model,
            )
            eval_df["strategy_model"] = strategy_model
            eval_df["assumed_model"] = strategy_model.replace("_proxy", "")
            eval_df["evaluator_model"] = evaluator_model
            eval_df["true_model"] = evaluator_model
            evals[(strategy_model, evaluator_model)] = eval_df
            eval_df.to_csv(output_dir / f"wrong_model_eval_{strategy_model}_under_{evaluator_model}.csv", index=False)
            matrix_rows.append(_evaluator_cell_summary(pair_id, strategy_model, evaluator_model, trades_df, eval_df))

    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(output_dir / "wrong_model_matrix.csv", index=False)

    def _cell(strategy: str, evaluator: str) -> float:
        rows = matrix.loc[matrix["strategy_model"].eq(strategy) & matrix["evaluator_model"].eq(evaluator), "net_pnl"]
        return float(rows.iloc[0]) if len(rows) else np.nan

    rf_correct = _cell("reduced_form_proxy", "reduced_form")
    ow_assumed_rf_true = _cell("OW_transient_proxy", "reduced_form")
    ow_correct = _cell("OW_transient_proxy", "OW_transient")
    rf_assumed_ow_true = _cell("reduced_form_proxy", "OW_transient")
    losses = pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "true_model": "reduced_form",
                "correct_strategy_model": "reduced_form_proxy",
                "wrong_strategy_model": "OW_transient_proxy",
                "correct_model_pnl": rf_correct,
                "wrong_model_pnl": ow_assumed_rf_true,
                "wrong_model_loss": ow_assumed_rf_true - rf_correct,
            },
            {
                "pair_id": pair_id,
                "true_model": "OW_transient",
                "correct_strategy_model": "OW_transient_proxy",
                "wrong_strategy_model": "reduced_form_proxy",
                "correct_model_pnl": ow_correct,
                "wrong_model_pnl": rf_assumed_ow_true,
                "wrong_model_loss": rf_assumed_ow_true - ow_correct,
            },
        ]
    )
    losses.to_csv(output_dir / "wrong_model_losses.csv", index=False)

    proxy_summary_rows = []
    for model_name, trades_df in [("OW_transient", ow_proxy), ("reduced_form", rf_proxy)]:
        proxy_summary_rows.append(
            {
                "pair_id": pair_id,
                "strategy_model": f"{model_name}_proxy",
                "assumed_model": model_name,
                "total_turnover": float(trades_df["signed_volume"].abs().sum()),
                "total_notional_turnover": float(trades_df["notional_turnover"].sum()),
                "max_participation_rate": float(trades_df["participation_rate"].max(skipna=True)),
                "mean_participation_rate": float(trades_df["participation_rate"].mean(skipna=True)),
                "total_internal_proxy_cost": float(trades_df["fitted_impact_cost"].sum()),
                "total_internal_proxy_net_pnl": float(trades_df["net_pnl"].sum()),
            }
        )
    proxy_summary = pd.DataFrame(proxy_summary_rows)
    proxy_summary.to_csv(output_dir / "wrong_model_proxy_strategy_summary.csv", index=False)

    checks = []
    same_trade = np.allclose(
        pd.to_numeric(ow_eval["signed_volume"], errors="coerce").fillna(0.0),
        pd.to_numeric(rf_eval["signed_volume"], errors="coerce").fillna(0.0),
    )
    checks.append({"check": "same_trade_evaluator_sensitivity", "status": "PASS" if same_trade else "FAIL", "message": f"same {run_config.strategy_model} trades under both evaluators"})
    checks.append({"check": "evaluator_sensitivity_fixed_path", "status": "PASS" if run_config.strategy_model == "OW_transient_proxy" or run_config.include_legacy_theoretical_ow else "WARN", "message": f"fixed_path_strategy_model={run_config.strategy_model}"})
    distinct = not np.allclose(
        pd.to_numeric(ow_proxy["signed_volume"], errors="coerce").fillna(0.0),
        pd.to_numeric(rf_proxy["signed_volume"], errors="coerce").fillna(0.0),
    )
    checks.append({"check": "proxy_strategy_model_distinct", "status": "PASS" if distinct else "WARN", "message": "OW_transient_proxy and reduced_form_proxy trade paths compared"})
    checks.append({"check": "wrong_model_matrix_complete", "status": "PASS" if len(matrix) == 4 else "FAIL", "message": f"cells={len(matrix)}"})
    loss_rf_ok = np.isclose(losses.loc[losses["true_model"].eq("reduced_form"), "wrong_model_loss"].iloc[0], ow_assumed_rf_true - rf_correct)
    loss_ow_ok = np.isclose(losses.loc[losses["true_model"].eq("OW_transient"), "wrong_model_loss"].iloc[0], rf_assumed_ow_true - ow_correct)
    checks.append({"check": "wrong_model_loss_formula", "status": "PASS" if loss_rf_ok and loss_ow_ok else "FAIL", "message": "loss = wrong_strategy_pnl - correct_strategy_pnl"})
    for strategy_model, trades_df, evaluator_model in [
        ("OW_transient_proxy", ow_proxy, "OW_transient"),
        ("reduced_form_proxy", rf_proxy, "reduced_form"),
    ]:
        internal_cost = float(trades_df["fitted_impact_cost"].sum())
        full_cost = float(evals[(strategy_model, evaluator_model)]["fitted_impact_cost_signed"].sum())
        ratio = full_cost / internal_cost if abs(internal_cost) > 1e-12 else np.nan
        status = "PASS" if np.isfinite(ratio) and 0.2 <= ratio <= 5.0 else "WARN"
        checks.append(
            {
                "check": f"proxy_strategy_cost_consistency_{strategy_model}",
                "status": status,
                "message": f"full_evaluator/internal_proxy_cost_ratio={ratio:.6g}",
            }
        )
    for key, eval_df in evals.items():
        flow_ok = np.allclose(eval_df["orderFlow_scenario"], eval_df["orderFlow_market"] + eval_df["strategy_signed_volume"])
        checks.append({"check": f"orderFlow_scenario_convention_{key[0]}_{key[1]}", "status": "PASS" if flow_ok else "FAIL", "message": "orderFlow_scenario = orderFlow_market + q_strategy"})
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(output_dir / "wrong_model_validation_checks.csv", index=False)

    _plot_wrong_model_outputs(output_dir, matrix, evals, pd.DataFrame([sensitivity_summary]))

    summary = {
        "pair_id": pair_id,
        "scenario_name": "wrong_model_strategy_misspecification",
        "scenario_type": "wrong_model_strategy_misspecification",
        "rf_true_correct_model_pnl": rf_correct,
        "rf_true_wrong_model_pnl": ow_assumed_rf_true,
        "wrong_model_loss_rf_true": ow_assumed_rf_true - rf_correct,
        "ow_true_correct_model_pnl": ow_correct,
        "ow_true_wrong_model_pnl": rf_assumed_ow_true,
        "wrong_model_loss_ow_true": rf_assumed_ow_true - ow_correct,
        "net_pnl_difference_rf_minus_ow": rf_correct - ow_correct,
        "impact_evaluator_sensitivity_net_pnl_difference_rf_minus_ow": sensitivity_summary["net_pnl_difference_rf_minus_ow"],
    }
    combined = pd.concat([pd.DataFrame([sensitivity_summary]), pd.DataFrame([summary])], ignore_index=True, sort=False)
    combined.to_csv(output_dir / "wrong_model_summary.csv", index=False)
    return matrix, summary


def _strategy_summary_row(pair_id: int, scenario: str, scenario_type: str, trades: pd.DataFrame) -> dict:
    """Summarize an integrated strategy stress scenario."""

    summary = compute_overall_strategy_summary(trades)
    return {"pair_id": pair_id, "scenario_name": scenario, "scenario_type": scenario_type, **summary}


def _stock_day_count(df: pd.DataFrame) -> int:
    """Count stock-days in a trade path."""

    if df.empty or "stock" not in df.columns:
        return 0
    date_col = _normal_date_col(df)
    return int(df.groupby(["stock", date_col], sort=False).ngroups)


def _liquidation_modes(run_config: IntegratedRunConfig) -> list[str]:
    if run_config.liquidation_mode == "both":
        return ["hard_block", "capped_with_residual"]
    return [run_config.liquidation_mode]


def _liquidation_trigger_modes(run_config: IntegratedRunConfig) -> list[str]:
    if run_config.liquidation_trigger_mode == "both":
        return ["deterministic_daily", "probabilistic_daily"]
    return [run_config.liquidation_trigger_mode]


def _probability_label(probability: float) -> str:
    return f"p{int(round(float(probability) * 100)):03d}"


def _forced_prefix(trigger_mode: str, probability: float, seed: int, mode: str) -> str:
    mode_label = "hard_block" if mode == "hard_block" else "capped_residual"
    if trigger_mode == "probabilistic_daily":
        return f"forced_liq_probabilistic_daily_{_probability_label(probability)}_seed{int(seed)}_{mode_label}"
    return f"forced_liq_deterministic_daily_{mode_label}"


def _forced_scenario_name(trigger_mode: str, mode: str, strategy_model: str, probability: float, seed: int) -> str:
    """Build an explicit scenario name for forced-liquidation summary rows."""

    return f"{_forced_prefix(trigger_mode, probability, seed, mode)}_{strategy_model}"


def _normal_date_col(df: pd.DataFrame) -> str:
    return "date" if "date" in df.columns else "trading_date"


def select_liquidation_events(
    trades_df: pd.DataFrame,
    trigger_mode: str,
    liquidation_probability: float = 0.10,
    random_seed: int = 42,
    liquidation_time: str = "12:00:00",
) -> pd.DataFrame:
    """Select stock-day forced-liquidation events reproducibly."""

    if trigger_mode not in {"deterministic_daily", "probabilistic_daily"}:
        raise ValueError("trigger_mode must be deterministic_daily or probabilistic_daily")
    if not 0 <= liquidation_probability <= 1:
        raise ValueError("liquidation_probability must be in [0, 1]")
    df = trades_df.copy()
    date_col = _normal_date_col(df)
    ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    threshold = pd.to_datetime(liquidation_time).time()
    rows = []
    for (date, stock), group in df.sort_values([date_col, "stock", ts_col]).groupby([date_col, "stock"], sort=True):
        eligible = group.loc[group[ts_col].dt.time >= threshold]
        if eligible.empty:
            continue
        event = eligible.iloc[0]
        rows.append(
            {
                "pair_id": event.get("pair_id", np.nan),
                "stock": str(stock),
                "date": str(date),
                "trading_date": str(date),
                "liquidation_time": liquidation_time,
                "trigger_mode": trigger_mode,
                "liquidation_probability": liquidation_probability,
                "random_seed": random_seed,
                "trigger_selected": True,
                "event_timestamp": event[ts_col],
                "pre_liquidation_position": float(event.get("position_before", event.get("position_after", 0.0))),
                "abs_pre_liquidation_position": abs(float(event.get("position_before", event.get("position_after", 0.0)))),
            }
        )
    events = pd.DataFrame(rows).sort_values(["date", "stock"]).reset_index(drop=True)
    if events.empty:
        return events
    if trigger_mode == "probabilistic_daily":
        rng = np.random.default_rng(int(random_seed))
        draws = rng.random(len(events))
        events["random_draw"] = draws
        events["trigger_selected"] = draws < float(liquidation_probability)
        events = events.loc[events["trigger_selected"]].copy().reset_index(drop=True)
    else:
        events["random_draw"] = np.nan
    return events


def _allowed_liquidation_trade(row: pd.Series, run_config: IntegratedRunConfig) -> float:
    adv = float(row.get("ADV", np.nan))
    if not np.isfinite(adv) or adv <= 0:
        return np.inf
    cap = run_config.max_liquidation_participation_rate
    if cap is None:
        cap = run_config.max_participation_rate_per_trade
    if cap is None:
        return np.inf
    return float(cap * adv)


def apply_forced_liquidation_stress_to_trades(
    baseline_trades: pd.DataFrame,
    liquidation_time: str,
    mode: str,
    run_config: IntegratedRunConfig,
    selected_events: pd.DataFrame | None = None,
    trigger_mode: str = "deterministic_daily",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply hard-block or cap-respecting forced liquidation to a trade path."""

    if mode not in {"hard_block", "capped_with_residual"}:
        raise ValueError("mode must be 'hard_block' or 'capped_with_residual'")
    threshold = pd.to_datetime(liquidation_time).time()
    df = baseline_trades.copy()
    date_col = _normal_date_col(df)
    ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(["stock", date_col, ts_col]).reset_index(drop=True)
    selected_lookup: dict[tuple[str, str], pd.Timestamp] = {}
    if selected_events is not None and len(selected_events):
        ev = selected_events.copy()
        ev["event_timestamp"] = pd.to_datetime(ev["event_timestamp"], errors="coerce")
        for _, row in ev.iterrows():
            selected_lookup[(str(row["stock"]), str(row.get(date_col, row.get("date", row.get("trading_date")))))] = row["event_timestamp"]
    records: list[dict] = []
    events: list[dict] = []
    carry_by_stock: dict[str, float] = {}
    for stock, stock_df in df.groupby("stock", sort=False):
        carried_position = 0.0
        for date, group in stock_df.groupby(date_col, sort=False):
            position_prev = carried_position if run_config.carry_residual_overnight else 0.0
            carried_overnight = abs(position_prev) > 1e-9
            liquidation_started = False
            liquidation_completed_ts = pd.NaT
            first_liq_residual = np.nan
            position_before_liq = np.nan
            total_liq_volume = 0.0
            n_liq_trades = 0
            max_liq_part = np.nan
            liq_parts: list[float] = []
            alpha_suppressed = 0
            liquidation_start_ts = pd.NaT
            event_key = (str(stock), str(date))
            event_ts_selected = selected_lookup.get(event_key)
            if selected_events is None:
                has_liq_timestamp = bool((pd.to_datetime(group[ts_col]).dt.time >= threshold).any())
            else:
                has_liq_timestamp = event_ts_selected is not None
            for _, row in group.iterrows():
                row_ts = pd.to_datetime(row[ts_col])
                baseline_trade = float(row.get("signed_volume", row.get("trade", 0.0)))
                delta_mid = float(row.get("delta_mid", 0.0))
                trade = baseline_trade
                is_start = False
                is_liq_trade = False
                residual_active_before = liquidation_started and abs(position_prev) > 1e-9
                desired_liq = 0.0
                actual_liq = 0.0
                unfilled_liq = 0.0
                suppressed = False

                should_start = row_ts.time() >= threshold if selected_events is None else row_ts == event_ts_selected
                if has_liq_timestamp and (not liquidation_started) and should_start:
                    liquidation_started = True
                    liquidation_start_ts = row_ts
                    position_before_liq = position_prev
                    is_start = True

                if liquidation_started:
                    if mode == "hard_block":
                        if is_start:
                            desired_liq = -position_prev
                            actual_liq = desired_liq
                            trade = actual_liq
                            is_liq_trade = abs(trade) > 1e-12
                        elif not run_config.resume_alpha_after_liquidation:
                            trade = 0.0
                            suppressed = abs(baseline_trade) > 1e-12
                        else:
                            trade = baseline_trade
                    else:
                        if abs(position_prev) > 1e-9:
                            desired_liq = -position_prev
                            max_allowed = _allowed_liquidation_trade(row, run_config)
                            actual_liq = float(np.clip(desired_liq, -max_allowed, max_allowed))
                            trade = actual_liq
                            unfilled_liq = desired_liq - actual_liq
                            is_liq_trade = abs(trade) > 1e-12
                            suppressed = abs(baseline_trade) > 1e-12
                        elif run_config.resume_alpha_after_liquidation:
                            trade = baseline_trade
                        else:
                            trade = 0.0
                            suppressed = abs(baseline_trade) > 1e-12

                position_before = position_prev
                position_after = position_before + trade
                if mode == "hard_block" and is_start:
                    position_after = 0.0
                    trade = -position_before
                    actual_liq = trade
                    is_liq_trade = abs(trade) > 1e-12

                adv = float(row.get("ADV", np.nan))
                part = abs(trade) / adv if np.isfinite(adv) and adv > 0 else np.nan
                cap = run_config.max_liquidation_participation_rate or run_config.max_participation_rate_per_trade
                violates = bool(is_liq_trade and cap is not None and np.isfinite(part) and part > cap + 1e-12)
                gross_pnl = position_before * delta_mid
                rec = row.to_dict()
                rec.update(
                    {
                        "signed_volume": trade,
                        "trade": trade,
                        "position_before": position_before,
                        "position_after": position_after,
                        "gross_pnl": gross_pnl,
                        "signed_impact_cost_normalized": 0.0,
                        "quadratic_impact_cost_normalized": 0.0,
                        "net_pnl": gross_pnl,
                        "liquidation_mode": mode,
                        "liquidation_trigger_mode": trigger_mode,
                        "liquidation_probability": run_config.liquidation_probability if trigger_mode == "probabilistic_daily" else 1.0,
                        "liquidation_random_seed": run_config.liquidation_random_seed,
                        "is_forced_liquidation_start": is_start,
                        "is_liquidation_trade": is_liq_trade,
                        "is_forced_liquidation": is_start,
                        "liquidation_trade": trade if is_liq_trade else 0.0,
                        "forced_liquidation_trade": trade if is_start else 0.0,
                        "desired_liquidation_trade": desired_liq,
                        "actual_liquidation_trade": actual_liq,
                        "unfilled_liquidation_trade": unfilled_liq,
                        "residual_inventory_active": bool(liquidation_started and abs(position_after) > 1e-9),
                        "residual_position": position_after if liquidation_started else 0.0,
                        "participation_rate": part,
                        "participation_rate_liquidation": part if is_liq_trade else np.nan,
                        "violates_participation_cap": violates,
                        "signed_volume_notional": trade * float(row.get("mid", np.nan)),
                        "abs_signed_volume": abs(trade),
                        "alpha_trade_suppressed_due_to_liquidation": suppressed,
                        "overnight_inventory_carried": carried_overnight,
                        "liquidation_fully_completed": bool(liquidation_started and abs(position_after) <= 1e-9),
                    }
                )
                records.append(rec)

                if is_liq_trade:
                    total_liq_volume += abs(trade)
                    n_liq_trades += 1
                    if np.isfinite(part):
                        liq_parts.append(part)
                        max_liq_part = max(liq_parts)
                if is_start:
                    first_liq_residual = position_after
                if liquidation_started and pd.isna(liquidation_completed_ts) and abs(position_after) <= 1e-9:
                    liquidation_completed_ts = row_ts
                if suppressed:
                    alpha_suppressed += 1
                position_prev = position_after

            final_residual = position_prev if has_liq_timestamp else 0.0
            if has_liq_timestamp:
                time_to_liq = (
                    (liquidation_completed_ts - liquidation_start_ts).total_seconds() / 60.0
                    if pd.notna(liquidation_completed_ts) and pd.notna(liquidation_start_ts)
                    else np.nan
                )
                events.append(
                    {
                        "stock": stock,
                        "trading_date": date,
                        "liquidation_mode": mode,
                        "liquidation_trigger_mode": trigger_mode,
                        "liquidation_probability": run_config.liquidation_probability if trigger_mode == "probabilistic_daily" else 1.0,
                        "liquidation_random_seed": run_config.liquidation_random_seed,
                        "liquidation_start_timestamp": liquidation_start_ts,
                        "liquidation_completion_timestamp": liquidation_completed_ts,
                        "position_before_liquidation": position_before_liq,
                        "total_liquidation_volume_executed": total_liq_volume,
                        "residual_position_after_first_liquidation": first_liq_residual,
                        "final_residual_position": final_residual,
                        "fully_liquidated_immediately": bool(abs(first_liq_residual) <= 1e-9),
                        "fully_liquidated_eventually": bool(abs(final_residual) <= 1e-9),
                        "carried_overnight": bool(abs(final_residual) > 1e-9 and run_config.carry_residual_overnight),
                        "number_of_liquidation_trades": n_liq_trades,
                        "max_liquidation_participation_rate": max_liq_part,
                        "mean_liquidation_participation_rate": float(np.nanmean(liq_parts)) if liq_parts else np.nan,
                        "time_to_liquidate_minutes": time_to_liq,
                        "alpha_trades_suppressed_due_to_liquidation": alpha_suppressed,
                    }
                )
            carried_position = final_residual if run_config.carry_residual_overnight else 0.0
        carry_by_stock[str(stock)] = carried_position
    out = pd.DataFrame(records).sort_values(["stock", date_col, ts_col]).reset_index(drop=True)
    out["cumulative_wealth"] = out["net_pnl"].cumsum()
    out["cumulative_gross_pnl"] = out["gross_pnl"].cumsum()
    out["terminal_inventory_value"] = out["position_after"] * out["mid"]
    return out, pd.DataFrame(events)


def _plot_forced_liquidation_outputs(
    output_dir: Path,
    summary_df: pd.DataFrame,
    baseline_rf_eval: pd.DataFrame,
    eval_paths: dict[tuple[str, str], Path],
    event_paths: dict[tuple[str, str], Path],
) -> None:
    """Save forced-liquidation comparison figures."""

    fig_dir = output_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return
    plot_df = summary_df.copy()
    plot_df["plot_label"] = plot_df["liquidation_trigger_mode"].astype(str) + "\n" + plot_df["liquidation_mode"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["plot_label"], pd.to_numeric(plot_df["net_pnl_under_reduced_form_eval"], errors="coerce"))
    if "baseline_net_pnl_rf_eval" in plot_df and plot_df["baseline_net_pnl_rf_eval"].notna().any():
        ax.axhline(float(plot_df["baseline_net_pnl_rf_eval"].dropna().iloc[0]), color="black", linestyle="--", linewidth=1, label="baseline RF eval")
        ax.legend()
    ax.set_title("Forced liquidation RF net PnL by trigger and mode")
    ax.set_ylabel("net PnL under reduced_form evaluator")
    fig.tight_layout()
    fig.savefig(fig_dir / "forced_liq_net_pnl_by_trigger_and_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["plot_label"], pd.to_numeric(plot_df.get("degradation_rf_eval"), errors="coerce"))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Forced liquidation degradation vs baseline")
    ax.set_ylabel("stress RF net PnL - baseline RF net PnL")
    fig.tight_layout()
    fig.savefig(fig_dir / "forced_liq_degradation_by_trigger_and_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    counts = plot_df.groupby("liquidation_trigger_mode", as_index=False)["number_of_liquidation_events"].max()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts["liquidation_trigger_mode"], counts["number_of_liquidation_events"])
    ax.set_title("Forced liquidation selected event counts")
    ax.set_ylabel("selected stock-days")
    fig.tight_layout()
    fig.savefig(fig_dir / "forced_liq_event_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    prob = plot_df.loc[plot_df["liquidation_trigger_mode"].astype(str).eq("probabilistic_daily")]
    if len(prob):
        rate = prob.groupby("liquidation_trigger_mode", as_index=False).agg(
            realized=("liquidation_event_rate_realized", "mean"),
            target=("liquidation_probability", "mean"),
        )
        x = np.arange(len(rate))
        ax.bar(x - 0.2, rate["realized"], width=0.4, label="realized")
        ax.bar(x + 0.2, rate["target"], width=0.4, label="target")
        ax.set_xticks(x)
        ax.set_xticklabels(rate["liquidation_trigger_mode"])
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No probabilistic liquidation rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_ylim(bottom=0)
    ax.set_title("Probabilistic liquidation realized event rate")
    ax.set_ylabel("event rate")
    fig.tight_layout()
    fig.savefig(fig_dir / "forced_liq_realized_event_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    baseline_port = _portfolio_timeseries(baseline_rf_eval, "datetime", ["net_pnl_fitted_model"]) if len(baseline_rf_eval) else pd.DataFrame()
    for trigger_mode in sorted(plot_df["liquidation_trigger_mode"].dropna().astype(str).unique()):
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(baseline_port):
            ax.plot(baseline_port["datetime"], baseline_port["cumulative_net_pnl_fitted_model"], label="baseline RF eval")
        for mode in ["hard_block", "capped_with_residual"]:
            path = eval_paths.get((trigger_mode, mode))
            if path is None or not path.exists():
                continue
            eval_df = pd.read_csv(path)
            net_col = "net_pnl_fitted_model_rf" if "net_pnl_fitted_model_rf" in eval_df.columns else "net_pnl_fitted_model"
            ts = _portfolio_timeseries(eval_df, "datetime", [net_col])
            if len(ts):
                label = "hard block" if mode == "hard_block" else "capped residual"
                ax.plot(ts["datetime"], ts[f"cumulative_{net_col}"], label=label)
        ax.set_title(f"Forced liquidation wealth: {trigger_mode}")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        suffix = trigger_mode
        if trigger_mode == "probabilistic_daily":
            prob_row = plot_df.loc[plot_df["liquidation_trigger_mode"].astype(str).eq(trigger_mode)].iloc[0]
            suffix = f"{trigger_mode}_{_probability_label(float(prob_row['liquidation_probability']))}_seed{int(prob_row['liquidation_random_seed'])}"
        fig.savefig(fig_dir / f"forced_liq_hard_vs_capped_wealth_{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    all_events = []
    for path in event_paths.values():
        if path.exists():
            all_events.append(pd.read_csv(path))
    events = pd.concat(all_events, ignore_index=True, sort=False) if all_events else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(events) and "time_to_liquidate_minutes" in events:
        for trigger_mode, group in events.groupby("liquidation_trigger_mode", sort=True):
            vals = pd.to_numeric(group["time_to_liquidate_minutes"], errors="coerce").dropna()
            if len(vals):
                ax.hist(vals, bins=30, alpha=0.5, label=str(trigger_mode))
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No time-to-liquidate values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Forced liquidation time to liquidate by trigger mode")
    ax.set_xlabel("minutes")
    fig.tight_layout()
    fig.savefig(fig_dir / "forced_liq_time_to_liquidate_by_trigger_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(events) and "final_residual_position" in events:
        residual = events.groupby("liquidation_trigger_mode", as_index=False)["final_residual_position"].apply(lambda s: s.abs().sum())
        ax.bar(residual["liquidation_trigger_mode"], residual["final_residual_position"])
    else:
        ax.text(0.5, 0.5, "No residual inventory events", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Forced liquidation residual inventory by trigger mode")
    ax.set_ylabel("total abs final residual position")
    fig.tight_layout()
    fig.savefig(fig_dir / "forced_liq_residual_inventory_by_trigger_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_signal_delay_stress_pair(
    pair_row: pd.Series,
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    stocks: list[str],
    alpha_df: pd.DataFrame,
    teammate_ow_params_df: pd.DataFrame,
    reduced_form_params_df: pd.DataFrame,
    run_config: IntegratedRunConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Run clock-time signal delay stress for an integrated pair."""

    pair_id = int(pair_row["pair_id"])
    delay = float(run_config.signal_delay_minutes)
    delayed_col = f"alpha_delayed_{delay:g}m"
    delayed = apply_clock_time_signal_delay(
        alpha_df,
        alpha_col="alpha_for_strategy",
        delay_minutes=delay,
        output_col=delayed_col,
    )
    delayed["alpha_for_strategy"] = delayed[delayed_col]
    trades = _run_proxy_strategy_for_stress(
        pair_row,
        train_raw_df,
        test_raw_df,
        delayed,
        teammate_ow_params_df,
        reduced_form_params_df,
        run_config,
    )
    trades["scenario_name"] = f"signal_delay_{delay:g}m_{run_config.strategy_model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / f"signal_delay_{run_config.strategy_model}_trades.csv", index=False)
    trades.to_csv(output_dir / "signal_delay_trades.csv", index=False)
    summary = _strategy_summary_row(pair_id, f"signal_delay_{delay:g}m_{run_config.strategy_model}", "signal_delay_stress", trades)
    summary["strategy_model"] = run_config.strategy_model
    pd.DataFrame([summary]).to_csv(output_dir / "signal_delay_summary.csv", index=False)
    return trades, summary


def run_forced_liquidation_stress_pair(
    pair_row: pd.Series,
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    teammate_ow_params_df: pd.DataFrame,
    reduced_form_params_df: pd.DataFrame,
    run_config: IntegratedRunConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Run selected forced-liquidation trigger/mode combinations."""

    pair_id = int(pair_row["pair_id"])
    baseline = _run_proxy_strategy_for_stress(
        pair_row,
        train_raw_df,
        test_raw_df,
        alpha_df,
        teammate_ow_params_df,
        reduced_form_params_df,
        run_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    validation_rows: list[dict[str, str | int]] = []
    first_trades = pd.DataFrame()
    baseline_ow_eval = evaluate_marginal_impact_from_strategy_trades(
        train_raw_df, test_raw_df, baseline, teammate_ow_params_df, pair_id, model_name="OW_transient"
    )
    baseline_rf_eval = evaluate_marginal_impact_from_strategy_trades(
        train_raw_df, test_raw_df, baseline, reduced_form_params_df, pair_id, model_name="reduced_form"
    )
    baseline_ow_net = float(baseline_ow_eval["net_pnl_fitted_model"].sum())
    baseline_rf_net = float(baseline_rf_eval["net_pnl_fitted_model"].sum())
    number_of_stock_days = _stock_day_count(baseline)
    eval_paths: dict[tuple[str, str], Path] = {}
    event_paths: dict[tuple[str, str], Path] = {}

    for trigger_mode in _liquidation_trigger_modes(run_config):
        selected_events = select_liquidation_events(
            baseline,
            trigger_mode,
            liquidation_probability=run_config.liquidation_probability,
            random_seed=run_config.liquidation_random_seed,
            liquidation_time=run_config.forced_liquidation_time,
        )
        if trigger_mode == "probabilistic_daily":
            selected_again = select_liquidation_events(
                baseline,
                trigger_mode,
                liquidation_probability=run_config.liquidation_probability,
                random_seed=run_config.liquidation_random_seed,
                liquidation_time=run_config.forced_liquidation_time,
            )
            keys = ["stock", "trading_date", "event_timestamp"]
            stable = selected_events[keys].astype(str).equals(selected_again[keys].astype(str)) if len(selected_events) == len(selected_again) else False
            validation_rows.append({"check": "probabilistic_reproducibility", "status": "PASS" if stable else "FAIL", "message": f"selected_events={len(selected_events)}", "pair_id": pair_id})
        for mode in _liquidation_modes(run_config):
            trades, events = apply_forced_liquidation_stress_to_trades(
                baseline,
                run_config.forced_liquidation_time,
                mode,
                run_config,
                selected_events=selected_events,
                trigger_mode=trigger_mode,
            )
            trades["pair_id"] = pair_id
            trades["train_month"] = str(pair_row["train_month"])
            trades["test_month"] = str(pair_row["test_month"])
            scenario = _forced_scenario_name(trigger_mode, mode, run_config.strategy_model, run_config.liquidation_probability, run_config.liquidation_random_seed)
            trades["scenario_name"] = scenario
            if first_trades.empty:
                first_trades = trades

            ow_eval = evaluate_marginal_impact_from_strategy_trades(
                train_raw_df, test_raw_df, trades, teammate_ow_params_df, pair_id, model_name="OW_transient"
            )
            rf_eval = evaluate_marginal_impact_from_strategy_trades(
                train_raw_df, test_raw_df, trades, reduced_form_params_df, pair_id, model_name="reduced_form"
            )
            eval_keys = ["stock", "trading_date", "datetime"]
            merged_eval = ow_eval.merge(
                rf_eval[
                    eval_keys
                    + [
                        "pred_ret_bps_market",
                        "pred_ret_bps_with_strategy",
                        "marginal_impact_bps",
                        "fitted_impact_cost_signed",
                        "fitted_impact_cost_abs",
                        "net_pnl_fitted_model",
                        "cumulative_wealth_fitted_model",
                    ]
                ],
                on=eval_keys,
                how="left",
                suffixes=("_ow", "_rf"),
            )

            prefix = _forced_prefix(trigger_mode, run_config.liquidation_probability, run_config.liquidation_random_seed, mode)
            trades_path = output_dir / f"{prefix}_trades.csv"
            events_path = output_dir / f"{prefix}_events.csv"
            eval_path = output_dir / f"{prefix}_fitted_evaluator.csv"
            trades.to_csv(trades_path, index=False)
            merged_eval.to_csv(eval_path, index=False)
            eval_paths[(trigger_mode, mode)] = eval_path

            events["pair_id"] = pair_id
            if len(events):
                liq_keys = ["stock", "trading_date"]
                ow_costs = (
                    ow_eval.loc[ow_eval.get("is_liquidation_trade", pd.Series(False, index=ow_eval.index)).fillna(False).astype(bool)]
                    .groupby(liq_keys, as_index=False)["fitted_impact_cost_signed"]
                    .sum()
                    .rename(columns={"fitted_impact_cost_signed": "fitted_liquidation_cost_ow"})
                )
                rf_costs = (
                    rf_eval.loc[rf_eval.get("is_liquidation_trade", pd.Series(False, index=rf_eval.index)).fillna(False).astype(bool)]
                    .groupby(liq_keys, as_index=False)["fitted_impact_cost_signed"]
                    .sum()
                    .rename(columns={"fitted_impact_cost_signed": "fitted_liquidation_cost_rf"})
                )
                events = events.merge(ow_costs, on=liq_keys, how="left")
                events = events.merge(rf_costs, on=liq_keys, how="left")
                events["fitted_liquidation_cost_ow"] = events["fitted_liquidation_cost_ow"].fillna(0.0)
                events["fitted_liquidation_cost_rf"] = events["fitted_liquidation_cost_rf"].fillna(0.0)
                events["wealth_drop_at_liquidation_ow"] = -events["fitted_liquidation_cost_ow"]
                events["wealth_drop_at_liquidation_rf"] = -events["fitted_liquidation_cost_rf"]
            events.to_csv(events_path, index=False)
            event_paths[(trigger_mode, mode)] = events_path
            validation_rows.append({"check": "event_file_always_saved", "status": "PASS" if events_path.exists() else "FAIL", "message": str(events_path), "pair_id": pair_id})

            # Backward-compatible aliases for existing notebooks/reports.
            if trigger_mode == "deterministic_daily":
                mode_label = "hard_block" if mode == "hard_block" else "capped_residual"
                trades.to_csv(output_dir / f"forced_liq_{mode_label}_{run_config.strategy_model}_trades.csv", index=False)
                trades.to_csv(output_dir / f"forced_liq_{mode_label}_trades.csv", index=False)
                events.to_csv(output_dir / f"forced_liq_{mode_label}_{run_config.strategy_model}_events.csv", index=False)
                events.to_csv(output_dir / f"forced_liq_{mode_label}_events.csv", index=False)
                merged_eval.to_csv(output_dir / f"forced_liq_{mode_label}_fitted_evaluator.csv", index=False)

            summary = _strategy_summary_row(pair_id, scenario, "forced_liquidation_stress", trades)
            summary["strategy_model"] = run_config.strategy_model
            summary.update(summarize_regression_evaluator(ow_eval, "ow_regression"))
            summary.update(summarize_regression_evaluator(rf_eval, "reduced_form"))
            event_count = int(len(events))
            rate = event_count / number_of_stock_days if number_of_stock_days else np.nan
            liq_rows = trades.loc[trades.get("is_liquidation_trade", pd.Series(False, index=trades.index)).fillna(False).astype(bool)]
            summary.update(
                {
                    "liquidation_trigger_mode": trigger_mode,
                    "liquidation_probability": run_config.liquidation_probability if trigger_mode == "probabilistic_daily" else 1.0,
                    "liquidation_random_seed": run_config.liquidation_random_seed,
                    "liquidation_mode": mode,
                    "number_of_stock_days": number_of_stock_days,
                    "number_of_liquidation_events": event_count,
                    "number_no_liquidation_events": max(number_of_stock_days - event_count, 0),
                    "liquidation_event_rate_realized": rate,
                    "expected_number_of_liquidation_events": number_of_stock_days * (run_config.liquidation_probability if trigger_mode == "probabilistic_daily" else 1.0),
                    "number_fully_liquidated_immediately": int(events["fully_liquidated_immediately"].sum()) if len(events) else 0,
                    "number_fully_liquidated_eventually": int(events["fully_liquidated_eventually"].sum()) if len(events) else 0,
                    "number_with_residual_after_first_liquidation": int((events["residual_position_after_first_liquidation"].abs() > 1e-9).sum()) if len(events) else 0,
                    "number_with_overnight_residual": int(events["carried_overnight"].sum()) if len(events) else 0,
                    "max_residual_position": float(events["final_residual_position"].abs().max()) if len(events) else 0.0,
                    "total_abs_residual_position": float(events["final_residual_position"].abs().sum()) if len(events) else 0.0,
                    "mean_time_to_liquidate_minutes": float(events["time_to_liquidate_minutes"].mean(skipna=True)) if len(events) else np.nan,
                    "max_time_to_liquidate_minutes": float(events["time_to_liquidate_minutes"].max(skipna=True)) if len(events) else np.nan,
                    "total_alpha_trades_suppressed_due_to_liquidation": int(events["alpha_trades_suppressed_due_to_liquidation"].sum()) if len(events) else 0,
                    "hard_block_cap_violation_rate": float(liq_rows["violates_participation_cap"].mean()) if mode == "hard_block" and len(liq_rows) else 0.0,
                    "total_turnover": float(trades["signed_volume"].abs().sum()),
                    "total_notional_turnover": float((trades["signed_volume"].abs() * trades["mid"]).sum()) if "mid" in trades else np.nan,
                    "baseline_net_pnl_ow_eval": baseline_ow_net,
                    "stress_net_pnl_ow_eval": float(ow_eval["net_pnl_fitted_model"].sum()),
                    "degradation_ow_eval": float(ow_eval["net_pnl_fitted_model"].sum() - baseline_ow_net),
                    "baseline_net_pnl_rf_eval": baseline_rf_net,
                    "stress_net_pnl_rf_eval": float(rf_eval["net_pnl_fitted_model"].sum()),
                    "degradation_rf_eval": float(rf_eval["net_pnl_fitted_model"].sum() - baseline_rf_net),
                }
            )
            summaries.append(summary)

            if trigger_mode == "deterministic_daily":
                status = "PASS" if event_count == number_of_stock_days else "WARN"
                validation_rows.append({"check": "deterministic_event_count", "status": status, "message": f"events={event_count}; stock_days={number_of_stock_days}", "pair_id": pair_id})
            else:
                p = float(run_config.liquidation_probability)
                se = np.sqrt(p * (1 - p) / number_of_stock_days) if number_of_stock_days else np.nan
                status = "PASS" if np.isfinite(rate) and 0 <= rate <= 1 and (not np.isfinite(se) or abs(rate - p) <= 3 * se) else "WARN"
                validation_rows.append({"check": "probabilistic_event_count_reasonable", "status": status, "message": f"realized={rate:.4g}; target={p:.4g}; events={event_count}; stock_days={number_of_stock_days}", "pair_id": pair_id})
                date_col = _normal_date_col(baseline)
                selected_keys = set(zip(selected_events.get("stock", pd.Series(dtype=str)).astype(str), selected_events.get("trading_date", pd.Series(dtype=str)).astype(str)))
                nonselected = set(zip(baseline["stock"].astype(str), baseline[date_col].astype(str))).difference(selected_keys)
                if nonselected:
                    base_key = baseline["stock"].astype(str) + "|" + baseline[date_col].astype(str)
                    trade_key = trades["stock"].astype(str) + "|" + trades[date_col].astype(str)
                    keys_str = {f"{s}|{d}" for s, d in nonselected}
                    unchanged = np.allclose(
                        baseline.loc[base_key.isin(keys_str), "signed_volume"].to_numpy(),
                        trades.loc[trade_key.isin(keys_str), "signed_volume"].to_numpy(),
                    )
                else:
                    unchanged = True
                validation_rows.append({"check": "non_selected_days_unchanged", "status": "PASS" if unchanged else "FAIL", "message": f"nonselected_stock_days={len(nonselected)}", "pair_id": pair_id})
                suppress = trades.loc[~(trades["stock"].astype(str) + "|" + trades[date_col].astype(str)).isin({f"{s}|{d}" for s, d in selected_keys}), "alpha_trade_suppressed_due_to_liquidation"].fillna(False).astype(bool).sum()
                validation_rows.append({"check": "no_alpha_suppression_on_nonselected_days", "status": "PASS" if int(suppress) == 0 else "FAIL", "message": f"nonselected_suppressed_rows={int(suppress)}", "pair_id": pair_id})

            if mode == "hard_block":
                starts = trades.loc[trades.get("is_forced_liquidation_start", pd.Series(False, index=trades.index)).fillna(False).astype(bool)]
                max_pos = float(starts["position_after"].abs().max()) if len(starts) else 0.0
                validation_rows.append({"check": "hard_block_selected_flat", "status": "PASS" if max_pos < 1e-8 else "FAIL", "message": f"max_position_after_start={max_pos:.3g}", "pair_id": pair_id})
            else:
                cap = run_config.max_liquidation_participation_rate or run_config.max_participation_rate_per_trade
                max_part = float(liq_rows["participation_rate_liquidation"].max(skipna=True)) if len(liq_rows) else 0.0
                validation_rows.append({"check": "capped_residual_cap_respected", "status": "PASS" if cap is None or max_part <= cap + 1e-9 else "FAIL", "message": f"max_liq_participation={max_part:.6g}; cap={cap}", "pair_id": pair_id})

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "forced_liquidation_summary.csv", index=False)
    summary_df.to_csv(output_dir / f"forced_liq_{run_config.strategy_model}_summary.csv", index=False)
    validation_rows.append({"check": "summary_has_trigger_mode", "status": "PASS" if {"liquidation_trigger_mode", "liquidation_probability"}.issubset(summary_df.columns) else "FAIL", "message": "forced_liquidation_summary.csv", "pair_id": pair_id})
    pd.DataFrame(validation_rows).to_csv(output_dir / "forced_liq_validation_checks.csv", index=False)
    _plot_forced_liquidation_outputs(output_dir, summary_df, baseline_rf_eval, eval_paths, event_paths)
    return first_trades, summaries[0] if summaries else {}
