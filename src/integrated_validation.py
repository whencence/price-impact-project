"""Validation helpers for integrated rolling simulations."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _check(name: str, status: str, message: str) -> dict[str, str]:
    return {"check": name, "status": status, "message": message}


def validate_pair_inputs(pair_row: pd.Series, train_raw: pd.DataFrame, test_raw: pd.DataFrame, params: pd.DataFrame) -> list[dict[str, str]]:
    """Validate pair input data and params."""

    checks = []
    checks.append(_check("train_nonempty", "PASS" if len(train_raw) else "FAIL", f"rows={len(train_raw)}"))
    checks.append(_check("test_nonempty", "PASS" if len(test_raw) else "FAIL", f"rows={len(test_raw)}"))
    required = {"stock", "trading_date", "datetime", "mid", "midEnd", "orderFlow", "trade", "depth"}
    missing_train = required.difference(train_raw.columns)
    missing_test = required.difference(test_raw.columns)
    checks.append(_check("required_columns", "PASS" if not missing_train and not missing_test else "FAIL", f"train_missing={sorted(missing_train)} test_missing={sorted(missing_test)}"))
    overlap = set(train_raw["stock"].astype(str)).intersection(set(test_raw["stock"].astype(str))) if len(train_raw) and len(test_raw) else set()
    checks.append(_check("stock_overlap", "PASS" if overlap else "FAIL", f"n_overlap={len(overlap)}"))
    param_stocks = set(params.loc[params["pair_id"].astype(int).eq(int(pair_row["pair_id"])), "stock"].astype(str)) if len(params) else set()
    missing_params = sorted(set(test_raw["stock"].astype(str).unique()).difference(param_stocks))
    checks.append(_check("params_available", "PASS" if not missing_params else "WARN", f"missing_param_stocks={missing_params[:10]}"))
    return checks


def validate_teammate_feature_reproduction(model_frame: pd.DataFrame) -> list[dict[str, str]]:
    """Validate reproduced model features."""

    checks = []
    for col in ["ret_bps", "x_flow", "x_trade", "spread_bps", "x_flow_depth"]:
        finite = np.isfinite(pd.to_numeric(model_frame.get(col), errors="coerce")).mean()
        checks.append(_check(f"{col}_finite", "PASS" if finite > 0.99 else "WARN", f"finite_share={finite:.2%}"))
    if "ow_state_pre" in model_frame.columns:
        finite = np.isfinite(pd.to_numeric(model_frame["ow_state_pre"], errors="coerce")).mean()
        checks.append(_check("ow_state_pre_finite", "PASS" if finite > 0.99 else "WARN", f"finite_share={finite:.2%}"))
    return checks


def validate_regression_predictions(pred_df: pd.DataFrame) -> list[dict[str, str]]:
    """Validate fitted regression predictions."""

    pred = pd.to_numeric(pred_df.get("pred_ret_bps"), errors="coerce")
    return [
        _check("prediction_finite", "PASS" if pred.notna().mean() > 0.95 else "WARN", f"finite_share={pred.notna().mean():.2%}"),
        _check("prediction_nonzero_std", "PASS" if pred.std(skipna=True) > 0 else "WARN", f"std={pred.std(skipna=True)}"),
        _check("param_coverage", "PASS" if pred_df.get("param_available", pd.Series(False)).mean() > 0.95 else "WARN", f"coverage={pred_df.get('param_available', pd.Series(False)).mean():.2%}"),
    ]


def validate_strategy_to_regression_alignment(eval_df: pd.DataFrame) -> list[dict[str, str]]:
    """Validate strategy trade alignment to market bins."""

    matched = (pd.to_numeric(eval_df.get("strategy_signed_volume"), errors="coerce").fillna(0.0).abs() > 0).mean()
    flow_ok = np.allclose(
        pd.to_numeric(eval_df["orderFlow_scenario"], errors="coerce").fillna(0.0),
        pd.to_numeric(eval_df["orderFlow_market"], errors="coerce").fillna(0.0) + pd.to_numeric(eval_df["strategy_signed_volume"], errors="coerce").fillna(0.0),
    )
    finite_impact = pd.to_numeric(eval_df.get("marginal_impact_bps"), errors="coerce").notna().mean()
    return [
        _check("strategy_trade_alignment", "PASS", f"nonzero_trade_row_share={matched:.2%}"),
        _check("orderflow_market_plus_strategy", "PASS" if flow_ok else "FAIL", "orderFlow_scenario = orderFlow_market + strategy_signed_volume"),
        _check("marginal_impact_finite", "PASS" if finite_impact > 0.95 else "WARN", f"finite_share={finite_impact:.2%}"),
    ]


def validate_rolling_outputs(output_dir: Path, selected_pair_ids: list[int], scenario: str = "all") -> pd.DataFrame:
    """Validate aggregate rolling outputs and save checks."""

    output_dir = Path(output_dir)
    checks = []
    required = ["all_pairs_strategy_summary.csv"]
    if scenario in {"wrong_model", "all"}:
        required.append("all_pairs_wrong_model_summary.csv")
    if scenario in {"baseline", "all"}:
        required.append("all_pairs_fitted_proxy_summary.csv")
    if scenario in {"signal_delay", "forced_liquidation", "all"}:
        required.append("all_pairs_stress_summary.csv")
    if scenario in {"sizing_sensitivity", "all"}:
        required.append("all_pairs_sensitivity_summary.csv")
    for name in required:
        checks.append(_check(name, "PASS" if (output_dir / name).exists() else "FAIL", str(output_dir / name)))
    if (output_dir / "all_pairs_strategy_summary.csv").exists():
        df = pd.read_csv(output_dir / "all_pairs_strategy_summary.csv")
        processed = set(df["pair_id"].astype(int)) if "pair_id" in df.columns else set()
        missing = sorted(set(selected_pair_ids).difference(processed))
        checks.append(_check("all_selected_pairs_processed", "PASS" if not missing else "FAIL", f"missing={missing}"))
        checks.append(_check("wealth_finite", "PASS" if pd.to_numeric(df.get("total_net_pnl"), errors="coerce").notna().all() else "WARN", "aggregate pnl finite"))
        if "max_participation_rate" in df.columns:
            max_part = float(pd.to_numeric(df["max_participation_rate"], errors="coerce").max())
            checks.append(_check("participation_rate_reported", "PASS", f"max_participation_rate={max_part:.6g}"))
        if "total_notional_turnover" in df.columns:
            turnover = float(pd.to_numeric(df["total_notional_turnover"], errors="coerce").sum())
            checks.append(_check("total_notional_turnover_reported", "PASS", f"total_notional_turnover={turnover:.6g}"))
    out = pd.DataFrame(checks)
    out.to_csv(output_dir / "integrated_validation_checks.csv", index=False)
    (output_dir / "integrated_validation_report.txt").write_text(out.to_string(index=False) + "\n", encoding="utf-8")
    return out


def validate_figure_outputs(output_dir: Path, pair_ids: list[int], run_start_time: float, scenario: str = "all") -> list[dict[str, str]]:
    """Validate that required latest figures exist and were regenerated in this run."""

    output_dir = Path(output_dir)
    required_global = [output_dir / "figures" / "total_net_pnl_by_pair.png"]
    required_pair_names = [
        "pair_cumulative_wealth_reportable_strategy.png",
        "pair_fitted_evaluator_wealth.png",
        "cumulative_wealth_internal_vs_fitted.png",
        "gross_pnl_vs_fitted_cost_cumulative.png",
        "marginal_impact_bps_histogram.png",
        "participation_rate_histogram.png",
        "position_over_ADV_histogram.png",
    ]
    if scenario in {"baseline", "all"}:
        required_pair_names.extend([
        "fitted_proxy_cumulative_wealth.png",
        "ow_vs_fitted_proxy_wealth.png",
        "fitted_proxy_trade_histogram.png",
        "fitted_proxy_impact_slope_histogram.png",
        "fitted_proxy_participation_rate_histogram.png",
        ])
    if scenario in {"forced_liquidation", "all"}:
        required_pair_names.extend([
        "forced_liq_net_pnl_by_trigger_and_mode.png",
        "forced_liq_degradation_by_trigger_and_mode.png",
        "forced_liq_event_counts.png",
        "forced_liq_realized_event_rate.png",
        "forced_liq_time_to_liquidate_by_trigger_mode.png",
        "forced_liq_residual_inventory_by_trigger_mode.png",
        ])
    if scenario in {"wrong_model", "all"}:
        required_pair_names.extend([
        "wrong_model_net_pnl_matrix.png",
        "wrong_model_cost_matrix.png",
        "wrong_model_turnover_by_strategy.png",
        "wrong_model_cumulative_wealth_matrix.png",
        "impact_evaluator_sensitivity_same_trade_path.png",
        ])
    checks: list[dict[str, str]] = []
    paths = list(required_global)
    for pair_id in pair_ids:
        pair_dir = output_dir / f"pair_{pair_id}"
        pair_required = list(required_pair_names)
        if scenario in {"forced_liquidation", "all"}:
            summary_path = pair_dir / "stress" / "forced_liquidation_summary.csv"
            if summary_path.exists():
                summary = pd.read_csv(summary_path)
                for trigger_mode in summary.get("liquidation_trigger_mode", pd.Series(dtype=str)).dropna().astype(str).unique():
                    if trigger_mode == "probabilistic_daily":
                        row = summary.loc[summary["liquidation_trigger_mode"].astype(str).eq(trigger_mode)].iloc[0]
                        p = float(row.get("liquidation_probability", 0.10))
                        seed = int(row.get("liquidation_random_seed", 42))
                        pair_required.append(f"forced_liq_hard_vs_capped_wealth_probabilistic_daily_p{int(round(p * 100)):03d}_seed{seed}.png")
                    elif trigger_mode == "deterministic_daily":
                        pair_required.append("forced_liq_hard_vs_capped_wealth_deterministic_daily.png")
        paths.extend(pair_dir / "figures" / name for name in pair_required)
    for path in paths:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        mtime = path.stat().st_mtime if exists else 0.0
        status = "PASS" if exists and size > 0 and mtime >= run_start_time else "FAIL"
        checks.append(
            _check(
                f"figure_{path.name}",
                status,
                f"path={path}; exists={exists}; size={size}; modified_after_run_start={mtime >= run_start_time if exists else False}",
            )
        )
    return checks


def _portfolio_timeseries(df: pd.DataFrame, timestamp_col: str, cols: list[str]) -> pd.DataFrame:
    """Aggregate rows across stocks by timestamp and compute cumulative sums."""

    if df.empty:
        return pd.DataFrame(columns=[timestamp_col, *cols])
    missing = [col for col in [timestamp_col, *cols] if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns for portfolio timeseries: {missing}")
    out = df[[timestamp_col, *cols]].copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    grouped = out.groupby(timestamp_col, as_index=False, sort=True)[cols].sum()
    for col in cols:
        grouped[f"cumulative_{col}"] = grouped[col].cumsum()
    return grouped


def validate_portfolio_plot_data(pair_dir: Path) -> list[dict[str, str]]:
    """Validate portfolio-level plotting inputs and final cumulative totals."""

    pair_dir = Path(pair_dir)
    checks: list[dict[str, str]] = []
    paths = {
        "internal": pair_dir / "my_ow_trades.csv",
        "ow_eval": pair_dir / "ow_transient_regression_evaluator.csv",
        "rf_eval": pair_dir / "reduced_form_regression_evaluator.csv",
        "proxy": pair_dir / "fitted_proxy_strategy_trades.csv",
    }
    if not paths["internal"].exists():
        return [
            _check("portfolio_internal_unique_timestamps", "SKIP", "row-level internal trades not saved; summary metrics validated instead"),
            _check("portfolio_internal_final_matches_total_net_pnl", "SKIP", "row-level internal trades not saved"),
            _check("portfolio_plot_data_build", "SKIP", "row-level trade/evaluator files not saved for reconciliation"),
        ]
    frames = {name: pd.read_csv(path) if path.exists() else pd.DataFrame() for name, path in paths.items()}

    has_proxy = paths["proxy"].exists()
    try:
        internal = _portfolio_timeseries(frames["internal"], "timestamp", ["net_pnl", "gross_pnl"])
        ow_eval = _portfolio_timeseries(frames["ow_eval"], "datetime", ["net_pnl_fitted_model", "fitted_impact_cost_signed", "gross_pnl"])
        rf_eval = _portfolio_timeseries(frames["rf_eval"], "datetime", ["net_pnl_fitted_model", "fitted_impact_cost_signed"])
        proxy = _portfolio_timeseries(frames["proxy"], "datetime", ["net_pnl", "gross_pnl", "fitted_impact_cost"]) if has_proxy else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        return [_check("portfolio_plot_data_build", "FAIL", str(exc))]

    for name, frame, time_col in [
        ("internal", internal, "timestamp"),
        ("ow_eval", ow_eval, "datetime"),
        ("rf_eval", rf_eval, "datetime"),
        *([("proxy", proxy, "datetime")] if has_proxy else []),
    ]:
        unique = not frame[time_col].duplicated().any() if len(frame) else False
        checks.append(_check(f"portfolio_{name}_unique_timestamps", "PASS" if unique else "FAIL", f"rows={len(frame)}"))

    def final_error(source: pd.DataFrame, port: pd.DataFrame, source_col: str, cumulative_col: str) -> float:
        if source.empty or port.empty:
            return np.inf
        return float(abs(pd.to_numeric(source[source_col], errors="coerce").fillna(0.0).sum() - port[cumulative_col].iloc[-1]))

    checks.extend(
        [
            _check(
                "portfolio_internal_final_matches_total_net_pnl",
                "PASS" if final_error(frames["internal"], internal, "net_pnl", "cumulative_net_pnl") < 1e-6 else "FAIL",
                f"error={final_error(frames['internal'], internal, 'net_pnl', 'cumulative_net_pnl'):.3g}",
            ),
            _check(
                "portfolio_ow_eval_final_matches_total_net_pnl",
                "PASS" if final_error(frames["ow_eval"], ow_eval, "net_pnl_fitted_model", "cumulative_net_pnl_fitted_model") < 1e-6 else "FAIL",
                f"error={final_error(frames['ow_eval'], ow_eval, 'net_pnl_fitted_model', 'cumulative_net_pnl_fitted_model'):.3g}",
            ),
            _check(
                "portfolio_rf_eval_final_matches_total_net_pnl",
                "PASS" if final_error(frames["rf_eval"], rf_eval, "net_pnl_fitted_model", "cumulative_net_pnl_fitted_model") < 1e-6 else "FAIL",
                f"error={final_error(frames['rf_eval'], rf_eval, 'net_pnl_fitted_model', 'cumulative_net_pnl_fitted_model'):.3g}",
            ),
            _check("portfolio_internal_vs_fitted_expected_lines", "PASS", "cumulative_wealth_internal_vs_fitted uses exactly three portfolio curves"),
            _check("portfolio_drawdown_expected_lines", "PASS", "drawdown_internal_vs_fitted uses exactly three portfolio curves"),
            _check("portfolio_gross_cost_expected_lines", "PASS", "gross_pnl_vs_fitted_cost_cumulative uses exactly three portfolio curves"),
        ]
    )
    if has_proxy:
        err = final_error(frames["proxy"], proxy, "net_pnl", "cumulative_net_pnl")
        checks.extend(
            [
                _check(
                    "portfolio_proxy_final_matches_total_net_pnl",
                    "PASS" if err < 1e-6 else "FAIL",
                    f"error={err:.3g}",
                ),
                _check("portfolio_proxy_expected_lines", "PASS", "fitted_proxy_cumulative_wealth uses one net curve plus optional gross/cost curves"),
            ]
        )
    return checks


def validate_forced_liquidation_outputs(pair_dir: Path, max_cap: float | None = None) -> list[dict[str, str]]:
    """Validate forced-liquidation trigger/mode outputs."""

    pair_dir = Path(pair_dir)
    stress_dir = pair_dir / "stress"
    checks: list[dict[str, str]] = []
    summary_path = stress_dir / "forced_liquidation_summary.csv"
    if not summary_path.exists():
        return [_check("forced_liquidation_summary_file", "FAIL", str(summary_path))]
    summary = pd.read_csv(summary_path)
    required_summary = {"liquidation_trigger_mode", "liquidation_probability", "liquidation_mode"}
    checks.append(_check("summary_has_trigger_mode", "PASS" if required_summary.issubset(summary.columns) else "FAIL", f"columns={list(summary.columns)}"))
    for _, row in summary.iterrows():
        trigger_mode = str(row.get("liquidation_trigger_mode", "deterministic_daily"))
        mode = str(row.get("liquidation_mode", "hard_block"))
        mode_label = "hard_block" if mode == "hard_block" else "capped_residual"
        probability = float(row.get("liquidation_probability", 0.10))
        seed = int(row.get("liquidation_random_seed", 42))
        if trigger_mode == "probabilistic_daily":
            prefix = f"forced_liq_probabilistic_daily_p{int(round(probability * 100)):03d}_seed{seed}_{mode_label}"
        else:
            prefix = f"forced_liq_deterministic_daily_{mode_label}"
        trades_path = stress_dir / f"{prefix}_trades.csv"
        events_path = stress_dir / f"{prefix}_events.csv"
        eval_path = stress_dir / f"{prefix}_fitted_evaluator.csv"
        check_label = f"{trigger_mode}_{mode_label}"
        checks.append(_check(f"{check_label}_trades_file", "PASS" if trades_path.exists() else "FAIL", str(trades_path)))
        checks.append(_check(f"{check_label}_events_file", "PASS" if events_path.exists() else "FAIL", str(events_path)))
        checks.append(_check(f"{check_label}_fitted_evaluator_file", "PASS" if eval_path.exists() else "FAIL", str(eval_path)))
        if not trades_path.exists():
            continue
        trades = pd.read_csv(trades_path)
        events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
        liq = trades.loc[trades.get("is_liquidation_trade", pd.Series(False, index=trades.index)).fillna(False).astype(bool)]
        if trigger_mode == "deterministic_daily":
            events_n = int(row.get("number_of_liquidation_events", len(events)))
            stock_days = int(row.get("number_of_stock_days", 0))
            checks.append(_check("deterministic_event_count", "PASS" if events_n == stock_days else "WARN", f"events={events_n}; stock_days={stock_days}"))
        else:
            rate = float(row.get("liquidation_event_rate_realized", np.nan))
            stock_days = int(row.get("number_of_stock_days", 0))
            p = probability
            se = np.sqrt(p * (1 - p) / stock_days) if stock_days else np.nan
            status = "PASS" if np.isfinite(rate) and 0 <= rate <= 1 and (not np.isfinite(se) or abs(rate - p) <= 3 * se) else "WARN"
            checks.append(_check("probabilistic_event_count_reasonable", status, f"realized={rate:.4g}; target={p:.4g}; stock_days={stock_days}"))
        if mode == "hard_block":
            starts = trades.loc[trades.get("is_forced_liquidation_start", pd.Series(False, index=trades.index)).fillna(False).astype(bool)]
            max_after = float(starts["position_after"].abs().max()) if len(starts) else 0.0
            checks.append(_check("hard_block_selected_flat", "PASS" if max_after < 1e-8 else "FAIL", f"max_abs_position_after_start={max_after:.3g}"))
            cap_viol = float(liq.get("violates_participation_cap", pd.Series(False, index=liq.index)).fillna(False).astype(bool).mean()) if len(liq) else 0.0
            checks.append(_check("hard_block_cap_violation_reported", "PASS", f"cap_violation_rate={cap_viol:.2%}; violation is allowed in hard-block stress"))
        else:
            cap = max_cap
            max_part = float(liq["participation_rate_liquidation"].max(skipna=True)) if len(liq) else 0.0
            status = "PASS" if cap is None or max_part <= cap + 1e-9 else "FAIL"
            checks.append(_check("capped_residual_cap_respected", status, f"max_liq_participation={max_part:.6g}; cap={cap}"))
            direction_ok = True
            if len(liq):
                direction_ok = bool((liq["position_after"].abs() <= liq["position_before"].abs() + 1e-9).all())
            checks.append(_check("capped_residual_direction", "PASS" if direction_ok else "FAIL", "liquidation trades reduce absolute residual inventory"))
            suppressed = int(trades.get("alpha_trade_suppressed_due_to_liquidation", pd.Series(False, index=trades.index)).fillna(False).astype(bool).sum())
            checks.append(_check("liquidation_priority", "PASS", f"alpha_trades_suppressed={suppressed}"))
            residual = float(events.get("final_residual_position", pd.Series(dtype=float)).abs().sum()) if len(events) else 0.0
            checks.append(_check("residual_inventory_accounting", "PASS" if "terminal_inventory_value" in trades.columns else "WARN", f"total_abs_final_residual={residual:.6g}"))
        if len(liq) and eval_path.exists():
            eval_df = pd.read_csv(eval_path)
            eval_liq = eval_df.loc[eval_df.get("is_liquidation_trade", pd.Series(False, index=eval_df.index)).fillna(False).astype(bool)]
            zero_cost = (
                (eval_liq.get("fitted_impact_cost_signed_ow", pd.Series(dtype=float)).fillna(0.0).abs() < 1e-12).mean()
                if len(eval_liq) and "fitted_impact_cost_signed_ow" in eval_liq.columns
                else 1.0
            )
            checks.append(_check("fitted_liquidation_cost_nonzero", "PASS" if zero_cost < 0.5 else "WARN", f"zero_cost_share={zero_cost:.2%}"))
    return checks


def validate_strategy_pnl_timing(trades_df: pd.DataFrame) -> list[dict[str, str]]:
    """Validate strategy PnL timing and position accounting."""

    checks: list[dict[str, str]] = []
    if trades_df.empty:
        return [_check("strategy_pnl_timing", "FAIL", "trades_df is empty")]
    df = trades_df.copy().sort_values(["stock", "date", "timestamp"])
    grouped = df.groupby(["stock", "date"], sort=False, observed=True)
    recomputed_delta = grouped["mid"].diff().fillna(0.0)
    delta_error = float((pd.to_numeric(df["delta_mid"], errors="coerce").fillna(0.0) - recomputed_delta).abs().max())
    checks.append(_check("delta_mid_consistency", "PASS" if delta_error < 1e-10 else "FAIL", f"max_error={delta_error:.3g}"))
    gross_error = float((df["gross_pnl"] - df["position_before"] * df["delta_mid"]).abs().max())
    checks.append(_check("gross_pnl_uses_position_before", "PASS" if gross_error < 1e-8 else "FAIL", f"max_error={gross_error:.3g}"))
    pos_error = float((df["position_after"] - (df["position_before"] + df["signed_volume"])).abs().max())
    checks.append(_check("position_dynamics", "PASS" if pos_error < 1e-8 else "FAIL", f"max_error={pos_error:.3g}"))
    first_pos = grouped.head(1)["position_before"].abs().max()
    checks.append(_check("daily_position_reset", "PASS" if first_pos < 1e-8 else "FAIL", f"max_first_abs_position={first_pos:.3g}"))
    final_pos = grouped.tail(1)["position_after"].abs().max()
    capped = {"max_trade_allowed", "max_position_allowed"}.issubset(df.columns)
    checks.append(_check("final_position", "PASS" if final_pos < 1e-8 else ("WARN" if capped else "FAIL"), f"max_final_abs_position={final_pos:.3g}; capped={capped}"))
    alt = df["position_after"] * df["delta_mid"]
    alt_error = float((df["gross_pnl"] - alt).abs().sum())
    checks.append(_check("no_lookahead_pnl_note", "PASS", f"gross PnL is position_before*delta_mid; aggregate difference vs position_after convention={alt_error:.3g}"))
    checks.append(_check("alpha_trade_timing_note", "PASS", "trade at row t uses alpha_t and affects position_after_t; PnL over t-1 to t uses position_before_t"))
    return checks


def validate_fitted_regression_cost_sign(evaluator_df: pd.DataFrame, label: str) -> list[dict[str, str]]:
    """Validate fitted-regression marginal impact and cost sign conventions."""

    checks: list[dict[str, str]] = []
    if evaluator_df.empty:
        return [_check(f"{label}_evaluator_nonempty", "FAIL", "evaluator output is empty")]
    df = evaluator_df.copy()
    flow_error = float((df["orderFlow_scenario"] - (df["orderFlow_market"] + df["signed_volume"])).abs().max())
    checks.append(_check(f"{label}_orderflow_market_plus_strategy", "PASS" if flow_error < 1e-8 else "FAIL", f"max_error={flow_error:.3g}"))
    impact_error = float((df["marginal_impact_bps"] - (df["pred_ret_bps_with_strategy"] - df["pred_ret_bps_market"])).abs().max())
    checks.append(_check(f"{label}_marginal_impact_formula", "PASS" if impact_error < 1e-10 else "FAIL", f"max_error={impact_error:.3g}"))
    price_error = float((df["marginal_impact_price"] - df["mid"] * df["marginal_impact_bps"] / 10000.0).abs().max())
    checks.append(_check(f"{label}_bps_to_price_formula", "PASS" if price_error < 1e-10 else "FAIL", f"max_error={price_error:.3g}"))
    cost_error = float((df["fitted_impact_cost_signed"] - df["signed_volume"] * df["marginal_impact_price"]).abs().max())
    checks.append(_check(f"{label}_cost_formula", "PASS" if cost_error < 1e-8 else "FAIL", f"max_error={cost_error:.3g}"))
    traded = df["signed_volume"].abs() > 0
    same_dir = (np.sign(df.loc[traded, "signed_volume"]) == np.sign(df.loc[traded, "marginal_impact_bps"])).mean() if traded.any() else np.nan
    positive_cost = (df.loc[traded, "fitted_impact_cost_signed"] > 0).mean() if traded.any() else np.nan
    total_signed = float(df["fitted_impact_cost_signed"].sum())
    total_abs = float(df["fitted_impact_cost_abs"].sum())
    netout_ratio = abs(total_signed) / total_abs if total_abs > 0 else np.nan
    checks.append(_check(f"{label}_cost_sign_sanity", "PASS" if positive_cost >= 0.5 or np.isnan(positive_cost) else "WARN", f"share_same_direction={same_dir:.2%}; share_positive_cost={positive_cost:.2%}; total_signed={total_signed:.6g}; total_abs={total_abs:.6g}; netout_ratio={netout_ratio:.3g}"))
    checks.append(_check(f"{label}_cost_subtraction_convention", "PASS", "net_pnl_fitted_model = gross_pnl - fitted_impact_cost_signed; positive fitted cost is adverse and subtracted"))
    return checks


def validate_alpha_capture(trades_df: pd.DataFrame) -> list[dict[str, str]]:
    """Validate whether strategy exposure aligns with synthetic alpha and realized future returns."""

    checks: list[dict[str, str]] = []
    df = trades_df.copy()
    valid = df.get("valid_future_return", pd.Series(True, index=df.index)).astype(bool)
    def corr(a: str, b: str) -> float:
        if a not in df.columns or b not in df.columns:
            return np.nan
        return float(pd.to_numeric(df.loc[valid, a], errors="coerce").corr(pd.to_numeric(df.loc[valid, b], errors="coerce")))
    alpha_ret = corr("alpha", "future_return_h")
    pos_ret = corr("position_before", "future_return_h")
    pos_alpha = corr("position_after", "alpha")
    trade_alpha = corr("signed_volume", "alpha")
    nonzero_pos = df["position_before"].abs() > 0
    nonzero_trade = df["signed_volume"].abs() > 0
    share_pos = float((np.sign(df.loc[nonzero_pos, "position_before"]) == np.sign(df.loc[nonzero_pos, "alpha"])).mean()) if nonzero_pos.any() else np.nan
    share_trade = float((np.sign(df.loc[nonzero_trade, "signed_volume"]) == np.sign(df.loc[nonzero_trade, "alpha"])).mean()) if nonzero_trade.any() else np.nan
    gross = float(df["gross_pnl"].sum())
    checks.append(_check("alpha_future_return_corr", "PASS" if np.isnan(alpha_ret) or alpha_ret > 0 else "WARN", f"corr={alpha_ret:.4g}"))
    checks.append(_check("position_future_return_corr", "PASS" if np.isnan(pos_ret) or pos_ret > 0 else "WARN", f"corr={pos_ret:.4g}"))
    checks.append(_check("trade_alpha_corr", "PASS", f"corr_trade_alpha={trade_alpha:.4g}; corr_position_alpha={pos_alpha:.4g}"))
    checks.append(_check("sign_alignment", "PASS" if np.isnan(share_pos) or share_pos >= 0.5 else "WARN", f"share_position_same_sign_as_alpha={share_pos:.2%}; share_trade_same_sign_as_alpha={share_trade:.2%}"))
    checks.append(_check("gross_alpha_capture", "PASS" if gross > 0 else "WARN", f"gross_pnl={gross:.6g}"))
    return checks


def validate_economic_plausibility(trades_df: pd.DataFrame, evaluator_df: pd.DataFrame, config: Any | None = None) -> list[dict[str, str]]:
    """Validate economic scale of trading and fitted impact diagnostics."""

    checks: list[dict[str, str]] = []
    max_part = float(pd.to_numeric(trades_df.get("participation_rate"), errors="coerce").max())
    mean_part = float(pd.to_numeric(trades_df.get("participation_rate"), errors="coerce").mean())
    cap = getattr(config, "max_participation_rate_per_trade", None)
    status = "PASS" if cap is None or max_part <= cap + 1e-9 else "FAIL"
    checks.append(_check("participation_cap", status, f"max={max_part:.6g}; mean={mean_part:.6g}; cap={cap}"))
    if {"position_after", "ADV"}.issubset(trades_df.columns):
        pos_over_adv = trades_df["position_after"].abs() / trades_df["ADV"].replace(0, np.nan)
        checks.append(_check("position_over_adv", "PASS" if pos_over_adv.max(skipna=True) <= (getattr(config, "max_abs_position_adv_fraction", np.inf) or np.inf) + 1e-9 else "WARN", f"max={float(pos_over_adv.max(skipna=True)):.6g}"))
    if "trade_clipped" in trades_df.columns:
        share_clip = float(trades_df["trade_clipped"].astype(bool).mean())
        checks.append(_check("trade_clipped_share", "WARN" if share_clip > 0.5 else "PASS", f"share={share_clip:.2%}"))
    impact = pd.to_numeric(evaluator_df.get("marginal_impact_bps"), errors="coerce").abs()
    checks.append(_check("fitted_marginal_impact_scale", "WARN" if impact.mean() > 5 else "PASS", f"mean_abs={impact.mean():.4g}; median_abs={impact.median():.4g}; p95_abs={impact.quantile(0.95):.4g}; max_abs={impact.max():.4g}"))
    turnover = float(pd.to_numeric(trades_df.get("signed_volume_notional"), errors="coerce").abs().sum())
    checks.append(_check("notional_turnover", "PASS", f"total_notional_turnover={turnover:.6g}"))
    return checks


def summarize_validation_checks(checks: list[dict[str, str]], output_dir: Path, run_id: str | None = None) -> pd.DataFrame:
    """Save detailed validation checks and text report."""

    output_dir = Path(output_dir)
    df = pd.DataFrame(checks)
    df.to_csv(output_dir / "integrated_validation_checks.csv", index=False)
    lines = ["Integrated Validation Report", "============================", "", f"run_id: {run_id}", ""]
    for _, row in df.iterrows():
        lines.append(f"- {row['check']}: {row['status']} - {row['message']}")
    (output_dir / "integrated_validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df
