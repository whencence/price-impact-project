"""Rolling-pair aware stress tests using teammate fitted regression evaluators."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.integration_config import IntegratedRunConfig
from src.integrated_ow_runner import prepare_integrated_ow_strategy_input, run_my_ow_strategy_on_pair
from src.reduced_form_regression_evaluator import evaluate_marginal_impact_from_strategy_trades
from src.stress_config import StressTestConfig
from src.stress_tests import apply_clock_time_signal_delay, run_ow_strategy_with_forced_liquidation
from src.ow_strategy import run_ow_strategy
from src.strategy_metrics import compute_overall_strategy_summary


def summarize_regression_evaluator(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Summarize fitted-regression evaluator output."""

    return {
        f"total_fitted_cost_{prefix}": float(df["fitted_impact_cost_signed"].sum()),
        f"total_abs_fitted_cost_{prefix}": float(df["fitted_impact_cost_abs"].sum()),
        f"net_pnl_under_{prefix}_eval": float(df["net_pnl_fitted_model"].sum()),
        f"mean_abs_marginal_impact_{prefix}_bps": float(df["marginal_impact_bps"].abs().mean()),
    }


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
    """Evaluate fixed OW trades under teammate OW and reduced-form regressions."""

    pair_id = int(pair_row["pair_id"])
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
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "wrong_model_regression_evaluator.csv", index=False)
    summary = {
        "pair_id": pair_id,
        **summarize_regression_evaluator(ow_eval, "ow_regression"),
        **summarize_regression_evaluator(rf_eval, "reduced_form"),
        "total_cost_difference_rf_minus_ow": float(merged["fitted_cost_diff"].sum()),
        "degradation_rf_vs_ow": float(rf_eval["net_pnl_fitted_model"].sum() - ow_eval["net_pnl_fitted_model"].sum()),
        "corr_between_ow_and_rf_marginal_impact": float(merged[["marginal_impact_bps_ow", "marginal_impact_bps_rf"]].corr().iloc[0, 1]),
    }
    return merged, summary


def _strategy_summary_row(pair_id: int, scenario: str, scenario_type: str, trades: pd.DataFrame) -> dict:
    """Summarize an integrated strategy stress scenario."""

    summary = compute_overall_strategy_summary(trades)
    return {"pair_id": pair_id, "scenario_name": scenario, "scenario_type": scenario_type, **summary}


def _liquidation_modes(run_config: IntegratedRunConfig) -> list[str]:
    if run_config.liquidation_mode == "both":
        return ["hard_block", "capped_with_residual"]
    return [run_config.liquidation_mode]


def _normal_date_col(df: pd.DataFrame) -> str:
    return "date" if "date" in df.columns else "trading_date"


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
            has_liq_timestamp = bool((pd.to_datetime(group[ts_col]).dt.time >= threshold).any())
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

                if has_liq_timestamp and (not liquidation_started) and row_ts.time() >= threshold:
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


def run_signal_delay_stress_pair(
    pair_row: pd.Series,
    train_raw_df: pd.DataFrame,
    test_raw_df: pd.DataFrame,
    stocks: list[str],
    alpha_df: pd.DataFrame,
    teammate_ow_params_df: pd.DataFrame,
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
    trades = run_my_ow_strategy_on_pair(
        pair_row,
        train_raw_df,
        test_raw_df,
        stocks,
        delayed,
        teammate_ow_params_df,
        run_config,
    )
    trades["scenario_name"] = f"signal_delay_{delay:g}m"
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / "signal_delay_trades.csv", index=False)
    summary = _strategy_summary_row(pair_id, f"signal_delay_{delay:g}m_OW", "signal_delay_stress", trades)
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
    """Run hard-block and/or capped-residual forced liquidation stress."""

    pair_id = int(pair_row["pair_id"])
    strategy_input, ow_config = prepare_integrated_ow_strategy_input(
        pair_row,
        train_raw_df,
        test_raw_df,
        alpha_df,
        teammate_ow_params_df,
        run_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_ow_strategy(strategy_input, ow_config)
    summaries: list[dict] = []
    first_trades = pd.DataFrame()
    for mode in _liquidation_modes(run_config):
        trades, events = apply_forced_liquidation_stress_to_trades(
            baseline,
            run_config.forced_liquidation_time,
            mode,
            run_config,
        )
        trades["pair_id"] = pair_id
        trades["train_month"] = str(pair_row["train_month"])
        trades["test_month"] = str(pair_row["test_month"])
        scenario = (
            f"forced_liq_hard_block_stop_OW"
            if mode == "hard_block"
            else "forced_liq_capped_residual_stop_OW"
        )
        trades["scenario_name"] = scenario
        if first_trades.empty:
            first_trades = trades

        ow_eval = evaluate_marginal_impact_from_strategy_trades(
            train_raw_df, test_raw_df, trades, teammate_ow_params_df, pair_id, model_name="OW_transient"
        )
        rf_eval = evaluate_marginal_impact_from_strategy_trades(
            train_raw_df, test_raw_df, trades, reduced_form_params_df, pair_id, model_name="reduced_form"
        )
        ow_eval.to_csv(output_dir / f"forced_liq_{'hard_block' if mode == 'hard_block' else 'capped_residual'}_ow_regression_evaluator.csv", index=False)
        rf_eval.to_csv(output_dir / f"forced_liq_{'hard_block' if mode == 'hard_block' else 'capped_residual'}_fitted_evaluator.csv", index=False)
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
        merged_eval.to_csv(
            output_dir / f"forced_liq_{'hard_block' if mode == 'hard_block' else 'capped_residual'}_fitted_evaluator.csv",
            index=False,
        )

        trades.to_csv(output_dir / f"forced_liq_{'hard_block' if mode == 'hard_block' else 'capped_residual'}_trades.csv", index=False)
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
        events.to_csv(output_dir / f"forced_liq_{'hard_block' if mode == 'hard_block' else 'capped_residual'}_events.csv", index=False)

        summary = _strategy_summary_row(pair_id, scenario, "forced_liquidation_stress", trades)
        summary.update(summarize_regression_evaluator(ow_eval, "ow_regression"))
        summary.update(summarize_regression_evaluator(rf_eval, "reduced_form"))
        if len(events):
            summary.update(
                {
                    "liquidation_mode": mode,
                    "number_of_liquidation_events": int(len(events)),
                    "number_fully_liquidated_immediately": int(events["fully_liquidated_immediately"].sum()),
                    "number_fully_liquidated_eventually": int(events["fully_liquidated_eventually"].sum()),
                    "number_with_residual_after_first_liquidation": int((events["residual_position_after_first_liquidation"].abs() > 1e-9).sum()),
                    "number_with_overnight_residual": int(events["carried_overnight"].sum()),
                    "max_residual_position": float(events["final_residual_position"].abs().max()),
                    "total_abs_residual_position": float(events["final_residual_position"].abs().sum()),
                    "mean_time_to_liquidate_minutes": float(events["time_to_liquidate_minutes"].mean(skipna=True)),
                    "max_time_to_liquidate_minutes": float(events["time_to_liquidate_minutes"].max(skipna=True)),
                    "total_alpha_trades_suppressed_due_to_liquidation": int(events["alpha_trades_suppressed_due_to_liquidation"].sum()),
                    "hard_block_cap_violation_rate": float(trades.loc[trades["is_liquidation_trade"].astype(bool), "violates_participation_cap"].mean()) if mode == "hard_block" and trades["is_liquidation_trade"].any() else 0.0,
                }
            )
        else:
            summary.update(
                {
                    "liquidation_mode": mode,
                    "number_of_liquidation_events": 0,
                    "number_fully_liquidated_immediately": 0,
                    "number_fully_liquidated_eventually": 0,
                    "number_with_residual_after_first_liquidation": 0,
                    "number_with_overnight_residual": 0,
                    "max_residual_position": 0.0,
                    "total_abs_residual_position": 0.0,
                    "mean_time_to_liquidate_minutes": np.nan,
                    "max_time_to_liquidate_minutes": np.nan,
                    "total_alpha_trades_suppressed_due_to_liquidation": 0,
                    "hard_block_cap_violation_rate": 0.0,
                }
            )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "forced_liquidation_summary.csv", index=False)
    return first_trades, summaries[0] if summaries else {}
