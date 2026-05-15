"""Run integrated rolling simulations using teammate processed data and regressions."""

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime

import pandas as pd

from src.fitted_regression_params import load_ow_transient_params, load_reduced_form_params
from src.fitted_regression_proxy_strategy import (
    FittedRegressionProxyConfig,
    compute_fitted_proxy_summary,
    run_fitted_regression_proxy_strategy,
    save_fitted_proxy_plots,
    validate_fitted_proxy_strategy,
)
from src.integrated_alpha_runner import build_alpha_input_for_pair
from src.integrated_ow_runner import run_my_ow_strategy_on_pair
from src.integrated_reporting import (
    archive_figure_dir,
    clean_figure_dir,
    save_debug_strategy_path_sample,
    save_global_plots,
    save_pair_plots,
    write_global_report,
    write_pair_report,
)
from src.integrated_stress_runner import (
    run_forced_liquidation_stress_pair,
    run_signal_delay_stress_pair,
    summarize_regression_evaluator,
    run_wrong_model_stress_pair,
)
from src.integration_config import IntegratedRunConfig, TeammateDataConfig
from src.reduced_form_regression_evaluator import evaluate_marginal_impact_from_strategy_trades
from src.strategy_metrics import compute_overall_strategy_summary
from src.teammate_integration import (
    get_pair_records,
    load_baseline_20stocks,
    load_monthly_bin_data_for_pair,
    parse_pair_universe,
    resolve_project_root,
)
from src.integrated_validation import (
    summarize_validation_checks,
    validate_alpha_capture,
    validate_economic_plausibility,
    validate_fitted_regression_cost_sign,
    validate_figure_outputs,
    validate_forced_liquidation_outputs,
    validate_pair_inputs,
    validate_portfolio_plot_data,
    validate_rolling_outputs,
    validate_strategy_pnl_timing,
)


def parse_args() -> ArgumentParser:
    """Create CLI parser."""

    parser = ArgumentParser(description="Run integrated rolling simulations")
    parser.add_argument("--mode", choices=["single_pair", "all_pairs"], default="single_pair")
    parser.add_argument("--pair-id", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--save-trades", action="store_true")
    parser.add_argument("--use-baseline-20stocks", action="store_true")
    parser.add_argument("--use-x-flow-lambda-proxy", action="store_true")
    parser.add_argument("--strategy-model", choices=["OW_transient_proxy", "reduced_form_proxy", "theoretical_OW_legacy"], default="OW_transient_proxy")
    parser.add_argument("--include-legacy-theoretical-ow", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--scenario", choices=["baseline", "signal_delay", "wrong_model", "forced_liquidation", "sizing_sensitivity", "all"], default="baseline")
    parser.add_argument("--stress-scenario", choices=["baseline", "signal_delay", "wrong_model", "forced_liquidation", "sizing_sensitivity", "all"], default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--experiment-root", default="outputs/full_runs")
    parser.add_argument("--clean-experiment-folder", action=BooleanOptionalAction, default=True)
    parser.add_argument("--max-rows-per-pair", type=int, default=None)
    parser.add_argument("--max-participation-rate", type=float, default=0.01)
    parser.add_argument("--max-abs-trade-adv-fraction", type=float, default=0.01)
    parser.add_argument("--max-abs-position-adv-fraction", type=float, default=0.05)
    parser.add_argument("--target-impact-scale", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--clean-output-figures", action=BooleanOptionalAction, default=True)
    parser.add_argument("--run-strategy-audit", action="store_true")
    parser.add_argument("--liquidation-mode", choices=["hard_block", "capped_with_residual", "both"], default="both")
    parser.add_argument("--liquidation-trigger-mode", choices=["deterministic_daily", "probabilistic_daily", "both"], default="deterministic_daily")
    parser.add_argument("--liquidation-probability", type=float, default=0.10)
    parser.add_argument("--liquidation-random-seed", type=int, default=42)
    parser.add_argument("--wrong-model-mode", choices=["evaluator_sensitivity", "strategy_misspecification", "both"], default="both")
    parser.add_argument("--max-liquidation-participation-rate", type=float, default=None)
    parser.add_argument("--carry-residual-overnight", action=BooleanOptionalAction, default=True)
    parser.add_argument("--liquidation-priority-over-alpha", action=BooleanOptionalAction, default=True)
    parser.add_argument("--resume-alpha-after-liquidation", action=BooleanOptionalAction, default=False)
    return parser


def _auto_experiment_name(args, run_id: str) -> str:
    """Build deterministic-readable experiment name when not provided."""

    scenario = args.stress_scenario or args.scenario
    pair = f"pair{args.pair_id}" if args.mode == "single_pair" else "allpairs"
    sample = "full" if args.max_rows_per_pair is None else f"n{args.max_rows_per_pair}"
    return f"{args.mode}_{pair}_{scenario}_{sample}_{run_id}"


def _copy_if_exists(src: Path, dst: Path) -> None:
    """Copy a file if it exists."""

    if Path(src).exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_simple_markdown_report(path: Path, title: str, lines: list[str]) -> None:
    """Write a compact scenario-specific markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()


def _pair_summary(pair_id: int, trades: pd.DataFrame) -> dict:
    summary = compute_overall_strategy_summary(trades)
    unique_dates = int(trades["date"].nunique()) if "date" in trades.columns else None
    return {
        "pair_id": pair_id,
        "strategy_model": trades["strategy_model"].iloc[0] if "strategy_model" in trades.columns and len(trades) else "unknown",
        "n_rows": len(trades),
        "n_stocks": trades["stock"].nunique(),
        "number_of_unique_dates": unique_dates,
        **summary,
    }


def _proxy_model_from_strategy_name(strategy_model: str) -> str:
    if strategy_model == "OW_transient_proxy":
        return "OW_transient"
    if strategy_model == "reduced_form_proxy":
        return "reduced_form"
    raise ValueError(f"{strategy_model} is not a fitted proxy strategy")


def _scaled_alpha(alpha: pd.DataFrame, run_config: IntegratedRunConfig) -> pd.DataFrame:
    out = alpha.copy()
    scale = float(run_config.alpha_scale) * float(run_config.target_impact_scale)
    if scale != 1.0 and "alpha_for_strategy" in out.columns:
        out["alpha_for_strategy"] = pd.to_numeric(out["alpha_for_strategy"], errors="coerce").fillna(0.0) * scale
    return out


def run_reportable_strategy_on_pair(
    pair_row: pd.Series,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    alpha: pd.DataFrame,
    ow_params: pd.DataFrame,
    rf_params: pd.DataFrame,
    run_config: IntegratedRunConfig,
) -> pd.DataFrame:
    """Run the selected reportable strategy for an integrated pair."""

    if run_config.strategy_model == "theoretical_OW_legacy":
        stocks = sorted(test_raw["stock"].astype(str).unique())
        return run_my_ow_strategy_on_pair(pair_row, train_raw, test_raw, stocks, alpha, ow_params, run_config)
    model_name = _proxy_model_from_strategy_name(run_config.strategy_model)
    params = ow_params if model_name == "OW_transient" else rf_params
    cfg = FittedRegressionProxyConfig(
        model_name=model_name,
        max_participation_rate_per_trade=run_config.max_participation_rate_per_trade,
        max_abs_trade_adv_fraction=run_config.max_abs_trade_adv_fraction,
        max_abs_position_adv_fraction=run_config.max_abs_position_adv_fraction,
    )
    trades = run_fitted_regression_proxy_strategy(train_raw, test_raw, _scaled_alpha(alpha, run_config), params, int(pair_row["pair_id"]), cfg)
    trades["pair_id"] = int(pair_row["pair_id"])
    trades["train_month"] = str(pair_row["train_month"])
    trades["test_month"] = str(pair_row["test_month"])
    trades["strategy_model"] = run_config.strategy_model
    trades["reportable_strategy"] = True
    return trades


def _run_sizing_sensitivity_for_pair(
    pair_row: pd.Series,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    stocks: list[str],
    alpha: pd.DataFrame,
    ow_params: pd.DataFrame,
    rf_params: pd.DataFrame,
    run_config: IntegratedRunConfig,
    pair_dir: Path,
) -> pd.DataFrame:
    """Run a compact sizing sensitivity grid for one integrated pair."""

    rows = []
    for target_scale in [0.1, 0.25, 0.5, 1.0]:
        for participation_cap in [0.005, 0.01, 0.02]:
            cfg = replace(
                run_config,
                target_impact_scale=target_scale,
                max_participation_rate_per_trade=participation_cap,
                max_abs_trade_adv_fraction=min(
                    participation_cap,
                    run_config.max_abs_trade_adv_fraction or participation_cap,
                ),
            )
            trades = run_reportable_strategy_on_pair(pair_row, train_raw, test_raw, alpha, ow_params, rf_params, cfg)
            summary = _pair_summary(int(pair_row["pair_id"]), trades)
            ow_eval = evaluate_marginal_impact_from_strategy_trades(
                train_raw,
                test_raw,
                trades,
                ow_params,
                int(pair_row["pair_id"]),
                model_name="OW_transient",
            )
            rf_eval = evaluate_marginal_impact_from_strategy_trades(
                train_raw,
                test_raw,
                trades,
                rf_params,
                int(pair_row["pair_id"]),
                model_name="reduced_form",
            )
            rows.append(
                {
                    "pair_id": int(pair_row["pair_id"]),
                    "strategy_model": cfg.strategy_model,
                    "target_impact_scale": target_scale,
                    "proxy_trade_scale": target_scale,
                    "max_participation_rate": participation_cap,
                    "total_net_pnl": summary.get("total_net_pnl"),
                    "total_turnover": summary.get("total_signed_volume_turnover"),
                    "max_participation_rate_realized": summary.get("max_participation_rate"),
                    "mean_participation_rate": summary.get("mean_participation_rate"),
                    "max_abs_position": summary.get("max_abs_position"),
                    "share_trade_clipped": float(trades.get("trade_clipped", pd.Series(False)).mean()),
                    "share_position_clipped": float(trades.get("position_clipped", pd.Series(False)).mean()),
                    "total_fitted_cost_ow_regression": float(ow_eval["fitted_impact_cost_signed"].sum()),
                    "net_pnl_under_ow_regression_eval": float(ow_eval["net_pnl_fitted_model"].sum()),
                    "total_fitted_cost_reduced_form": float(rf_eval["fitted_impact_cost_signed"].sum()),
                    "net_pnl_under_reduced_form_eval": float(rf_eval["net_pnl_fitted_model"].sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(pair_dir / "sizing_sensitivity_summary.csv", index=False)
    out.to_csv(pair_dir / f"sizing_sensitivity_{run_config.strategy_model}_summary.csv", index=False)
    return out


def main() -> None:
    """Run the integrated rolling pipeline."""

    run_start_time = time.time()
    run_timestamp = datetime.now().astimezone()
    run_id = run_timestamp.strftime("%Y%m%d_%H%M%S")
    warnings.filterwarnings(
        "ignore",
        message="The default of observed=False is deprecated",
        category=FutureWarning,
    )
    args = parse_args().parse_args()
    root = resolve_project_root()
    scenario = args.stress_scenario or args.scenario
    experiment_name = args.experiment_name or _auto_experiment_name(args, run_id)
    experiment_root = root / args.experiment_root
    out_dir = experiment_root / experiment_name
    if args.clean_experiment_folder and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["metadata", "reports", "tables"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
    latest_pointer_dir = root / "outputs" / "rolling_runs"
    latest_pointer_dir.mkdir(parents=True, exist_ok=True)
    (latest_pointer_dir / "latest_experiment_path.txt").write_text(str(out_dir) + "\n", encoding="utf-8")
    data_config = TeammateDataConfig()
    run_config = IntegratedRunConfig(
        run_mode=args.mode,
        debug_pair_id=args.pair_id,
        max_pairs=args.max_pairs,
        save_pair_level_trades=args.save_trades,
        use_baseline_20stocks=args.use_baseline_20stocks,
        use_x_flow_lambda_proxy=args.use_x_flow_lambda_proxy,
        strategy_model=args.strategy_model,
        include_legacy_theoretical_ow=args.include_legacy_theoretical_ow,
        skip_stress=args.skip_stress,
        skip_sensitivity=args.skip_sensitivity,
        max_rows_per_pair=args.max_rows_per_pair,
        max_participation_rate_per_trade=args.max_participation_rate,
        max_abs_trade_adv_fraction=args.max_abs_trade_adv_fraction,
        max_abs_position_adv_fraction=args.max_abs_position_adv_fraction,
        target_impact_scale=args.target_impact_scale,
        alpha_scale=args.alpha_scale,
        liquidation_mode=args.liquidation_mode,
        liquidation_trigger_mode=args.liquidation_trigger_mode,
        liquidation_probability=args.liquidation_probability,
        liquidation_random_seed=args.liquidation_random_seed,
        wrong_model_mode=args.wrong_model_mode,
        max_liquidation_participation_rate=args.max_liquidation_participation_rate,
        carry_residual_overnight=args.carry_residual_overnight,
        liquidation_priority_over_alpha=args.liquidation_priority_over_alpha,
        resume_alpha_after_liquidation=args.resume_alpha_after_liquidation,
    )
    run_baseline_outputs = scenario in {"baseline", "all"}
    run_wrong = scenario in {"wrong_model", "all"} and not args.skip_stress
    run_signal = scenario in {"signal_delay", "all"} and not args.skip_stress
    run_forced = scenario in {"forced_liquidation", "all"} and not args.skip_stress
    run_sizing = scenario in {"sizing_sensitivity", "all"} and not args.skip_sensitivity
    if run_config.max_rows_per_pair is None:
        print("FULL OOS RUN: using all rows for selected pair(s).")
    ow_params = load_ow_transient_params(root / data_config.ow_params_path)
    rf_params = load_reduced_form_params(root / data_config.reduced_form_params_path)
    pairs = get_pair_records(data_config, run_config, root)
    selected_pair_ids = [int(row["pair_id"]) for row in pairs]
    cleaned_paths: list[str] = []
    if args.clean_output_figures:
        cleaned_paths.extend(str(p) for p in clean_figure_dir(out_dir / "figures"))
        for pair_id in selected_pair_ids:
            cleaned_paths.extend(str(p) for p in clean_figure_dir(out_dir / f"pair_{pair_id}" / "figures"))
    run_archive_dir = out_dir / "runs" / run_id
    (run_archive_dir / "figures").mkdir(parents=True, exist_ok=True)
    baseline_stocks = load_baseline_20stocks(data_config, root) if run_config.use_baseline_20stocks else None
    strategy_rows = []
    baseline_eval_rows = []
    wrong_rows = []
    stress_rows = []
    sensitivity_rows = []
    fitted_proxy_rows = []
    skipped_rows = []
    detailed_validation_rows = []
    pair_report_paths = []
    debug_paths = []
    processed_pair_ids = []
    rows_loaded_by_pair: dict[str, dict[str, int]] = {}

    metadata = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "scenario": scenario,
        "timestamp": run_timestamp.isoformat(),
        "command": " ".join(sys.argv),
        "pair_ids": selected_pair_ids,
        "mode": args.mode,
        "save_trades": bool(args.save_trades),
        "max_rows_per_pair": run_config.max_rows_per_pair,
        "full_out_of_sample_run": run_config.max_rows_per_pair is None,
        "git_commit": _git_commit(),
        "config": run_config.__dict__,
        "input_paths": {
            "rolling_pair_summary": data_config.rolling_pair_summary_path,
            "ow_params": data_config.ow_params_path,
            "reduced_form_params": data_config.reduced_form_params_path,
        },
        "figure_cleanup_enabled": bool(args.clean_output_figures),
        "experiment_folder_cleaned": bool(args.clean_experiment_folder),
        "cleaned_figure_files": cleaned_paths,
        "run_archive_dir": str(run_archive_dir),
    }
    (out_dir / "metadata" / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (out_dir / "metadata" / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (out_dir / "metadata" / "config.json").write_text(json.dumps(run_config.__dict__, indent=2, default=str), encoding="utf-8")
    (out_dir / "latest_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    for pair_row in pairs:
        pair_id = int(pair_row["pair_id"])
        processed_pair_ids.append(pair_id)
        pair_dir = out_dir / f"pair_{pair_id}"
        alpha_dir = pair_dir / "alpha"
        stress_dir = pair_dir / "stress"
        pair_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["reports", "figures", "tables", "trades", "stress", "debug"]:
            (pair_dir / subdir).mkdir(parents=True, exist_ok=True)
        stocks = baseline_stocks if baseline_stocks is not None else parse_pair_universe(pair_row)
        train_raw, test_raw = load_monthly_bin_data_for_pair(
            pair_row,
            stocks,
            data_config,
            sample_nrows=run_config.max_rows_per_pair,
        )
        rows_loaded_by_pair[str(pair_id)] = {"train_rows": int(len(train_raw)), "test_rows": int(len(test_raw))}
        checks = validate_pair_inputs(pair_row, train_raw, test_raw, ow_params)
        pd.DataFrame(checks).to_csv(pair_dir / "pair_input_validation.csv", index=False)

        alpha = build_alpha_input_for_pair(
            test_raw,
            pair_id,
            str(pair_row["test_month"]),
            stocks,
            alpha_horizon_minutes=run_config.alpha_horizon_minutes,
            rho=run_config.alpha_target_corr,
            alpha_decay_half_life_minutes=run_config.alpha_decay_half_life_minutes,
            output_dir=alpha_dir,
        )
        trades = run_reportable_strategy_on_pair(pair_row, train_raw, test_raw, alpha, ow_params, rf_params, run_config)
        if run_config.save_pair_level_trades:
            trades.to_csv(pair_dir / f"baseline_{run_config.strategy_model}_trades.csv", index=False)
            trades.to_csv(pair_dir / "trades" / f"baseline_{run_config.strategy_model}_trades.csv", index=False)
        trades.to_csv(pair_dir / f"{run_config.strategy_model}_trades_latest.csv", index=False)
        strategy_summary = _pair_summary(pair_id, trades)
        strategy_rows.append(strategy_summary)
        detailed_validation_rows.append(
            {
                "check": "reportable_strategy_is_proxy",
                "status": "PASS" if run_config.strategy_model == "OW_transient_proxy" else ("SKIP" if run_config.include_legacy_theoretical_ow else "WARN"),
                "message": f"strategy_model={run_config.strategy_model}",
                "pair_id": pair_id,
            }
        )
        detailed_validation_rows.append(
            {
                "check": "baseline_uses_OW_transient_proxy",
                "status": "PASS" if run_config.strategy_model == "OW_transient_proxy" else ("SKIP" if run_config.include_legacy_theoretical_ow else "WARN"),
                "message": f"reportable strategy_model={run_config.strategy_model}",
                "pair_id": pair_id,
            }
        )

        ow_eval = evaluate_marginal_impact_from_strategy_trades(
            train_raw, test_raw, trades, ow_params, pair_id, model_name="OW_transient"
        )
        rf_eval = evaluate_marginal_impact_from_strategy_trades(
            train_raw, test_raw, trades, rf_params, pair_id, model_name="reduced_form"
        )
        ow_eval.to_csv(pair_dir / "ow_transient_regression_evaluator.csv", index=False)
        rf_eval.to_csv(pair_dir / "reduced_form_regression_evaluator.csv", index=False)
        if run_config.save_pair_level_trades:
            ow_eval.to_csv(pair_dir / "trades" / "baseline_ow_transient_regression_evaluator.csv", index=False)
            rf_eval.to_csv(pair_dir / "trades" / "baseline_reduced_form_regression_evaluator.csv", index=False)
        baseline_eval_rows.append(
            {
                "pair_id": pair_id,
                "strategy_model": run_config.strategy_model,
                **summarize_regression_evaluator(ow_eval, "ow_regression"),
                **summarize_regression_evaluator(rf_eval, "reduced_form"),
            }
        )
        if run_baseline_outputs:
            if run_config.strategy_model != "theoretical_OW_legacy":
                proxy_summary = {
                    "pair_id": pair_id,
                    "strategy_model": run_config.strategy_model,
                    **compute_fitted_proxy_summary(trades),
                }
                fitted_proxy_rows.append(proxy_summary)
                trades.to_csv(pair_dir / "fitted_proxy_strategy_trades.csv", index=False)
                if run_config.save_pair_level_trades:
                    trades.to_csv(pair_dir / "trades" / f"fitted_proxy_strategy_{_proxy_model_from_strategy_name(run_config.strategy_model)}.csv", index=False)
                save_fitted_proxy_plots(trades, trades, pair_dir)
                proxy_config = FittedRegressionProxyConfig(model_name=_proxy_model_from_strategy_name(run_config.strategy_model))
                for check in validate_fitted_proxy_strategy(trades, proxy_config):
                    check["pair_id"] = pair_id
                    detailed_validation_rows.append(check)
        if run_config.include_legacy_theoretical_ow and run_config.strategy_model != "theoretical_OW_legacy":
            legacy = run_my_ow_strategy_on_pair(pair_row, train_raw, test_raw, stocks, alpha, ow_params, run_config)
            legacy["strategy_model"] = "theoretical_OW_legacy"
            legacy.to_csv(pair_dir / "legacy_theoretical_OW_trades.csv", index=False)
            if run_config.save_pair_level_trades:
                legacy.to_csv(pair_dir / "trades" / "legacy_theoretical_OW_trades.csv", index=False)
        for check in validate_strategy_pnl_timing(trades):
            check["pair_id"] = pair_id
            detailed_validation_rows.append(check)
        for check in validate_alpha_capture(trades):
            check["pair_id"] = pair_id
            detailed_validation_rows.append(check)
        for check in validate_fitted_regression_cost_sign(ow_eval, "ow_regression"):
            check["pair_id"] = pair_id
            detailed_validation_rows.append(check)
        for check in validate_fitted_regression_cost_sign(rf_eval, "reduced_form"):
            check["pair_id"] = pair_id
            detailed_validation_rows.append(check)
        for check in validate_economic_plausibility(trades, rf_eval, run_config):
            check["pair_id"] = pair_id
            detailed_validation_rows.append(check)

        if run_sizing:
            try:
                sizing_df = _run_sizing_sensitivity_for_pair(
                    pair_row,
                    train_raw,
                    test_raw,
                    stocks,
                    alpha,
                    ow_params,
                    rf_params,
                    run_config,
                    pair_dir,
                )
                sensitivity_rows.extend(sizing_df.to_dict("records"))
                detailed_validation_rows.append({"check": "sizing_uses_OW_transient_proxy", "status": "PASS" if run_config.strategy_model == "OW_transient_proxy" else "WARN", "message": f"strategy_model={run_config.strategy_model}", "pair_id": pair_id})
            except Exception as exc:  # noqa: BLE001
                skipped_rows.append({"pair_id": pair_id, "scenario_name": "sizing_sensitivity", "reason": str(exc)})
                print(f"WARNING: pair {pair_id} sizing sensitivity skipped: {exc}")

        if run_wrong:
            try:
                wrong_eval, wrong_summary = run_wrong_model_stress_pair(
                    pair_row, train_raw, test_raw, trades, ow_params, rf_params, run_config, stress_dir
                )
                wrong_rows.append(wrong_summary)
                stress_rows.append(wrong_summary)
                wrong_summary_path = stress_dir / "wrong_model_summary.csv"
                if wrong_summary_path.exists():
                    stress_rows.extend(pd.read_csv(wrong_summary_path).to_dict("records"))
                wrong_checks_path = stress_dir / "wrong_model_validation_checks.csv"
                if wrong_checks_path.exists():
                    for check in pd.read_csv(wrong_checks_path).to_dict("records"):
                        check["pair_id"] = pair_id
                        detailed_validation_rows.append(check)
            except Exception as exc:  # noqa: BLE001 - scenario failures should be reported and skipped.
                skipped_rows.append({"pair_id": pair_id, "scenario_name": "wrong_model", "reason": str(exc)})
                print(f"WARNING: pair {pair_id} wrong-model stress skipped: {exc}")

        if run_signal:
            try:
                signal_trades, signal_summary = run_signal_delay_stress_pair(
                    pair_row,
                    train_raw,
                    test_raw,
                    stocks,
                    alpha,
                    ow_params,
                    rf_params,
                    run_config,
                    stress_dir,
                )
                signal_rf_eval = evaluate_marginal_impact_from_strategy_trades(
                    train_raw, test_raw, signal_trades, rf_params, pair_id, model_name="reduced_form"
                )
                signal_ow_eval = evaluate_marginal_impact_from_strategy_trades(
                    train_raw, test_raw, signal_trades, ow_params, pair_id, model_name="OW_transient"
                )
                signal_eval = signal_ow_eval.merge(
                    signal_rf_eval[["stock", "trading_date", "datetime", "net_pnl_fitted_model", "fitted_impact_cost_signed"]],
                    on=["stock", "trading_date", "datetime"],
                    how="left",
                    suffixes=("_ow", "_rf"),
                )
                signal_eval.to_csv(stress_dir / f"signal_delay_{run_config.strategy_model}_evaluator.csv", index=False)
                signal_rf_eval.to_csv(stress_dir / "signal_delay_fitted_evaluator.csv", index=False)
                signal_summary.update(summarize_regression_evaluator(signal_rf_eval, "reduced_form"))
                signal_summary.update(summarize_regression_evaluator(signal_ow_eval, "ow_regression"))
                comparison = pd.DataFrame(
                    [
                        {
                            "pair_id": pair_id,
                            "strategy_model": run_config.strategy_model,
                            "baseline_OW_proxy_net_pnl_under_OW_eval": float(ow_eval["net_pnl_fitted_model"].sum()),
                            "delayed_OW_proxy_net_pnl_under_OW_eval": float(signal_ow_eval["net_pnl_fitted_model"].sum()),
                            "delta_OW_eval": float(signal_ow_eval["net_pnl_fitted_model"].sum() - ow_eval["net_pnl_fitted_model"].sum()),
                            "baseline_OW_proxy_net_pnl_under_RF_eval": float(rf_eval["net_pnl_fitted_model"].sum()),
                            "delayed_OW_proxy_net_pnl_under_RF_eval": float(signal_rf_eval["net_pnl_fitted_model"].sum()),
                            "delta_RF_eval": float(signal_rf_eval["net_pnl_fitted_model"].sum() - rf_eval["net_pnl_fitted_model"].sum()),
                            "turnover_baseline": float(trades["signed_volume"].abs().sum()),
                            "turnover_delayed": float(signal_trades["signed_volume"].abs().sum()),
                            "cost_baseline": float(ow_eval["fitted_impact_cost_signed"].sum()),
                            "cost_delayed": float(signal_ow_eval["fitted_impact_cost_signed"].sum()),
                        }
                    ]
                )
                comparison.to_csv(stress_dir / "signal_delay_comparison.csv", index=False)
                stress_rows.append(signal_summary)
                detailed_validation_rows.append({"check": "signal_delay_uses_OW_transient_proxy", "status": "PASS" if run_config.strategy_model == "OW_transient_proxy" else "WARN", "message": f"strategy_model={run_config.strategy_model}", "pair_id": pair_id})
            except Exception as exc:  # noqa: BLE001
                skipped_rows.append({"pair_id": pair_id, "scenario_name": "signal_delay", "reason": str(exc)})
                print(f"WARNING: pair {pair_id} signal-delay stress skipped: {exc}")

        if run_forced:
            try:
                _, forced_summary = run_forced_liquidation_stress_pair(
                    pair_row,
                    train_raw,
                    test_raw,
                    alpha,
                    ow_params,
                    rf_params,
                    run_config,
                    stress_dir,
                )
                forced_summary_path = stress_dir / "forced_liquidation_summary.csv"
                if forced_summary_path.exists():
                    stress_rows.extend(pd.read_csv(forced_summary_path).to_dict("records"))
                else:
                    stress_rows.append(forced_summary)
                forced_checks_path = stress_dir / "forced_liq_validation_checks.csv"
                if forced_checks_path.exists():
                    detailed_validation_rows.extend(pd.read_csv(forced_checks_path).to_dict("records"))
                detailed_validation_rows.append({"check": "forced_liq_uses_OW_transient_proxy", "status": "PASS" if run_config.strategy_model == "OW_transient_proxy" else "WARN", "message": f"strategy_model={run_config.strategy_model}", "pair_id": pair_id})
            except Exception as exc:  # noqa: BLE001
                skipped_rows.append({"pair_id": pair_id, "scenario_name": "forced_liquidation", "reason": str(exc)})
                print(f"WARNING: pair {pair_id} forced-liquidation stress skipped: {exc}")

        save_pair_plots(pair_dir, trades, ow_eval, rf_eval)
        debug_path = save_debug_strategy_path_sample(pair_dir, trades, ow_eval, rf_eval)
        if debug_path is not None:
            debug_paths.append(debug_path)
        archive_figure_dir(pair_dir / "figures", run_archive_dir / f"pair_{pair_id}" / "figures")
        legacy_exists = (pair_dir / "legacy_theoretical_OW_trades.csv").exists() or (pair_dir / "my_ow_trades.csv").exists()
        detailed_validation_rows.append({"check": "no_legacy_ow_in_final_outputs", "status": "PASS" if run_config.include_legacy_theoretical_ow or not legacy_exists else "FAIL", "message": f"legacy_outputs_present={legacy_exists}", "pair_id": pair_id})
        write_pair_report(
            pair_dir,
            pair_id,
            pair_row,
            strategy_summary,
            run_id=run_id,
            figures_cleaned=args.clean_output_figures,
        )
        pair_report_paths.append(pair_dir / f"pair_{pair_id}_integration_report.md")

    strategy_df = pd.DataFrame(strategy_rows)
    baseline_eval_df = pd.DataFrame(baseline_eval_rows)
    wrong_columns = [
        "pair_id",
        "scenario_name",
        "scenario_type",
        "rf_true_correct_model_pnl",
        "rf_true_wrong_model_pnl",
        "wrong_model_loss_rf_true",
        "ow_true_correct_model_pnl",
        "ow_true_wrong_model_pnl",
        "wrong_model_loss_ow_true",
        "total_fitted_cost_ow_regression",
        "total_fitted_cost_reduced_form",
        "total_cost_difference_rf_minus_ow",
        "net_pnl_difference_rf_minus_ow",
    ]
    stress_columns = [
        "pair_id",
        "scenario_name",
        "scenario_type",
        "total_net_pnl",
        "daily_sharpe",
        "total_signed_volume_turnover",
        "max_drawdown",
    ]
    wrong_df = pd.DataFrame(wrong_rows) if wrong_rows else pd.DataFrame(columns=wrong_columns)
    stress_df = pd.DataFrame(stress_rows) if stress_rows else pd.DataFrame(columns=stress_columns)
    sensitivity_df = pd.DataFrame(sensitivity_rows) if sensitivity_rows else pd.DataFrame(
        columns=[
            "pair_id",
            "target_impact_scale",
            "max_participation_rate",
            "total_net_pnl",
            "total_turnover",
            "max_participation_rate_realized",
            "mean_participation_rate",
            "max_abs_position",
            "total_fitted_cost_reduced_form",
            "net_pnl_under_reduced_form_eval",
        ]
    )
    fitted_proxy_df = pd.DataFrame(fitted_proxy_rows) if fitted_proxy_rows else pd.DataFrame(
        columns=["pair_id", "strategy_model", "total_net_pnl_fitted_proxy"]
    )
    skipped_df = pd.DataFrame(skipped_rows, columns=["pair_id", "scenario_name", "reason"])
    strategy_df.to_csv(out_dir / "all_pairs_strategy_summary.csv", index=False)
    baseline_eval_df.to_csv(out_dir / "all_pairs_baseline_fitted_evaluator_summary.csv", index=False)
    if run_wrong:
        wrong_df.to_csv(out_dir / "all_pairs_wrong_model_summary.csv", index=False)
    if run_wrong or run_signal or run_forced:
        stress_df.to_csv(out_dir / "all_pairs_stress_summary.csv", index=False)
    if run_sizing:
        sensitivity_df.to_csv(out_dir / "all_pairs_sensitivity_summary.csv", index=False)
    if run_baseline_outputs:
        fitted_proxy_df.to_csv(out_dir / "all_pairs_fitted_proxy_summary.csv", index=False)
    if len(skipped_df):
        skipped_df.to_csv(out_dir / "skipped_scenarios.csv", index=False)
    save_global_plots(out_dir)
    archive_figure_dir(out_dir / "figures", run_archive_dir / "figures")
    rolling_checks = validate_rolling_outputs(out_dir, processed_pair_ids, scenario=scenario)
    figure_checks = validate_figure_outputs(out_dir, processed_pair_ids, run_start_time, scenario=scenario)
    portfolio_plot_checks = []
    for pair_id in processed_pair_ids:
        checks = validate_portfolio_plot_data(out_dir / f"pair_{pair_id}")
        for check in checks:
            check["pair_id"] = pair_id
            portfolio_plot_checks.append(check)
    forced_liq_checks = []
    liq_cap = run_config.max_liquidation_participation_rate or run_config.max_participation_rate_per_trade
    if run_forced:
        for pair_id in processed_pair_ids:
            checks = validate_forced_liquidation_outputs(out_dir / f"pair_{pair_id}", liq_cap)
            for check in checks:
                check["pair_id"] = pair_id
                forced_liq_checks.append(check)
    all_checks = detailed_validation_rows + rolling_checks.to_dict("records") + figure_checks + portfolio_plot_checks + forced_liq_checks
    summarize_validation_checks(all_checks, out_dir, run_id=run_id)
    write_global_report(
        out_dir,
        processed_pair_ids,
        run_config,
        run_id=run_id,
        figures_cleaned=args.clean_output_figures,
        run_archive_dir=run_archive_dir,
    )
    metadata["rows_loaded_by_pair"] = rows_loaded_by_pair
    metadata["rows_processed_per_scenario"] = {
        "strategy_rows": int(len(strategy_df)),
        "stress_rows": int(len(stress_df)),
        "sensitivity_rows": int(len(sensitivity_df)),
    }
    if len(strategy_df):
        metadata["n_rows_processed"] = int(strategy_df["n_rows"].sum()) if "n_rows" in strategy_df.columns else int(strategy_df.get("rows", pd.Series(dtype=float)).sum())
        metadata["n_dates"] = int(strategy_df["number_of_unique_dates"].max()) if "number_of_unique_dates" in strategy_df.columns else None
        metadata["n_stock_days"] = int(strategy_df["number_of_stock_days"].sum()) if "number_of_stock_days" in strategy_df.columns else None
    (out_dir / "metadata" / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (out_dir / "latest_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    _copy_if_exists(out_dir / "integrated_rolling_report.md", out_dir / "reports" / "integrated_report.md")
    _copy_if_exists(out_dir / "integrated_rolling_report.txt", out_dir / "reports" / "integrated_report.txt")
    _copy_if_exists(out_dir / "integrated_validation_report.txt", out_dir / "reports" / "validation_report.txt")
    _copy_if_exists(out_dir / "integrated_validation_checks.csv", out_dir / "tables" / "validation_checks.csv")
    _copy_if_exists(out_dir / "all_pairs_strategy_summary.csv", out_dir / "tables" / "strategy_summary.csv")
    _copy_if_exists(out_dir / "all_pairs_strategy_summary.csv", out_dir / "tables" / "baseline_strategy_summary.csv")
    _copy_if_exists(out_dir / "all_pairs_baseline_fitted_evaluator_summary.csv", out_dir / "tables" / "baseline_fitted_evaluator_summary.csv")
    _copy_if_exists(out_dir / "all_pairs_stress_summary.csv", out_dir / "tables" / "stress_summary.csv")
    _copy_if_exists(out_dir / "all_pairs_fitted_proxy_summary.csv", out_dir / "tables" / "fitted_proxy_summary.csv")
    _copy_if_exists(out_dir / "all_pairs_sensitivity_summary.csv", out_dir / "tables" / "sensitivity_summary.csv")
    for pair_id in processed_pair_ids:
        pair_dir = out_dir / f"pair_{pair_id}"
        _copy_if_exists(pair_dir / f"pair_{pair_id}_integration_report.md", pair_dir / "reports" / "baseline_report.md")
        _copy_if_exists(pair_dir / "pair_input_validation.csv", pair_dir / "tables" / "pair_input_validation.csv")
        _copy_if_exists(pair_dir / "sizing_sensitivity_summary.csv", pair_dir / "tables" / "sizing_sensitivity_summary.csv")
        _copy_if_exists(pair_dir / "figures" / "cumulative_wealth_internal_vs_fitted.png", pair_dir / "figures" / "baseline_portfolio_cumulative_wealth_internal_vs_fitted.png")
        _copy_if_exists(pair_dir / "figures" / "drawdown_internal_vs_fitted.png", pair_dir / "figures" / "baseline_portfolio_drawdown_internal_vs_fitted.png")
        _copy_if_exists(pair_dir / "figures" / "gross_pnl_vs_fitted_cost_cumulative.png", pair_dir / "figures" / "baseline_gross_pnl_vs_fitted_costs.png")
        _copy_if_exists(pair_dir / "figures" / "participation_rate_histogram.png", pair_dir / "figures" / "baseline_participation_rate_histogram.png")
        _copy_if_exists(pair_dir / "figures" / "alpha_position_alignment_sample.png", pair_dir / "figures" / "baseline_alpha_position_sample.png")
        _copy_if_exists(pair_dir / "stress" / "wrong_model_regression_evaluator.csv", pair_dir / "stress" / "wrong_model_evaluator.csv")
        if (pair_dir / "stress" / "signal_delay_summary.csv").exists():
            signal_text = pd.read_csv(pair_dir / "stress" / "signal_delay_summary.csv").to_string(index=False)
            _write_simple_markdown_report(
                pair_dir / "reports" / "signal_delay_report.md",
                "Signal Delay Report",
                ["```text", signal_text, "```"],
            )
        _copy_if_exists(pair_dir / "stress" / "forced_liquidation_summary.csv", pair_dir / "tables" / "forced_liquidation_summary.csv")
        if (pair_dir / "stress" / "forced_liquidation_summary.csv").exists():
            forced_text = pd.read_csv(pair_dir / "stress" / "forced_liquidation_summary.csv").to_string(index=False)
            _write_simple_markdown_report(
                pair_dir / "reports" / "forced_liquidation_report.md",
                "Forced Liquidation Report",
                [
                    "Hard-block and capped-residual liquidation scenarios are evaluated separately.",
                    "",
                    "```text",
                    forced_text,
                    "```",
                ],
            )
        if (pair_dir / "stress" / "wrong_model_regression_evaluator.csv").exists():
            _write_simple_markdown_report(
                pair_dir / "reports" / "wrong_model_report.md",
                "Wrong Model Report",
                [
                    "Impact evaluator sensitivity uses the same theoretical OW trade path under multiple fitted evaluators.",
                    "Strategy misspecification uses fitted proxy trades generated under OW_transient and reduced_form assumptions, then evaluates each under both fitted models.",
                ],
            )
        if (pair_dir / "sizing_sensitivity_summary.csv").exists():
            _write_simple_markdown_report(
                pair_dir / "reports" / "sizing_sensitivity_report.md",
                "Sizing Sensitivity Report",
                ["Sizing sensitivity grid output is saved in pair tables."],
            )
    if scenario == "all":
        scenario_files = {
            "baseline": [
                out_dir / "all_pairs_strategy_summary.csv",
                out_dir / "all_pairs_baseline_fitted_evaluator_summary.csv",
                out_dir / "all_pairs_fitted_proxy_summary.csv",
            ],
            "signal_delay": [out_dir / "all_pairs_stress_summary.csv"],
            "wrong_model": [out_dir / "all_pairs_wrong_model_summary.csv"],
            "forced_liquidation": [out_dir / "all_pairs_stress_summary.csv"],
            "sizing_sensitivity": [out_dir / "all_pairs_sensitivity_summary.csv"],
        }
        for scenario_name, files in scenario_files.items():
            scenario_dir = out_dir / scenario_name
            (scenario_dir / "tables").mkdir(parents=True, exist_ok=True)
            (scenario_dir / "reports").mkdir(parents=True, exist_ok=True)
            (scenario_dir / "figures").mkdir(parents=True, exist_ok=True)
            for src in files:
                _copy_if_exists(src, scenario_dir / "tables" / src.name)
            _copy_if_exists(out_dir / "reports" / "integrated_report.md", scenario_dir / "reports" / "integrated_report.md")
            for pair_id in processed_pair_ids:
                pair_dir = out_dir / f"pair_{pair_id}"
                scenario_pair_dir = scenario_dir / f"pair_{pair_id}"
                if scenario_name == "forced_liquidation":
                    if (pair_dir / "stress").exists():
                        shutil.copytree(pair_dir / "stress", scenario_pair_dir / "stress", dirs_exist_ok=True)
                    for fig in (pair_dir / "figures").glob("forced_liq*.png"):
                        _copy_if_exists(fig, scenario_pair_dir / "figures" / fig.name)
                elif scenario_name == "sizing_sensitivity":
                    _copy_if_exists(pair_dir / "sizing_sensitivity_summary.csv", scenario_pair_dir / "tables" / "sizing_sensitivity_summary.csv")
                    for fig in (pair_dir / "figures").glob("sizing_sensitivity*.png"):
                        _copy_if_exists(fig, scenario_pair_dir / "figures" / fig.name)
                elif scenario_name == "baseline":
                    if (pair_dir / "trades").exists():
                        shutil.copytree(pair_dir / "trades", scenario_pair_dir / "trades", dirs_exist_ok=True)
                    for fig in (pair_dir / "figures").glob("baseline*.png"):
                        _copy_if_exists(fig, scenario_pair_dir / "figures" / fig.name)

    audit_paths = None
    if args.run_strategy_audit and scenario in {"baseline", "all"} and processed_pair_ids:
        try:
            from src.strategy_implementation_audit import run_strategy_implementation_audit

            audit_paths = run_strategy_implementation_audit(out_dir, processed_pair_ids[0])
            print(f"Strategy implementation audit report: {audit_paths['report']}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: strategy implementation audit failed: {exc}")

    print(f"Run ID: {run_id}")
    print(f"Experiment: {experiment_name}")
    print(f"Scenario: {scenario}")
    print(f"Processed pairs: {processed_pair_ids}")
    print(f"Saved experiment outputs to {out_dir}")
    print(f"Report: {out_dir / 'reports' / 'integrated_report.md'}")
    print(f"Validation report: {out_dir / 'reports' / 'validation_report.txt'}")
    if audit_paths is not None:
        print(f"Strategy implementation audit: {audit_paths['report']}")
        print(f"Strategy audit figures: {audit_paths['figures']}")
    if pair_report_paths:
        print(f"Pair report: {pair_report_paths[0]}")
    if debug_paths:
        print(f"Debug sample: {debug_paths[0]}")
    print(f"Figures: {out_dir / 'figures'} and pair-level figures under {out_dir / 'pair_<id>' / 'figures'}")
    print(f"Run-specific figures: {run_archive_dir / 'figures'} and {run_archive_dir / 'pair_<id>' / 'figures'}")
    print(f"Figure cleanup enabled: {args.clean_output_figures}")
    if run_config.max_rows_per_pair is not None:
        print("WARNING: This is a truncated debug run; do not interpret daily Sharpe or final performance.")
    else:
        print("FULL OOS RUN: max_rows_per_pair=None, all selected test rows were processed.")
    if len(strategy_df):
        row = strategy_df.iloc[0]
        print(
            "Key diagnostics: "
            f"gross_pnl={row.get('total_gross_pnl')}, "
            f"net_pnl_internal={row.get('total_net_pnl')}, "
            f"max_participation_rate={row.get('max_participation_rate')}, "
            f"mean_participation_rate={row.get('mean_participation_rate')}, "
            f"unique_dates={trades['date'].nunique() if 'trades' in locals() else 'n/a'}, "
            f"stock_days={trades[['stock', 'date']].drop_duplicates().shape[0] if 'trades' in locals() else 'n/a'}"
        )
    if len(wrong_df):
        row = wrong_df.iloc[0]
        print(
            "Fitted evaluator: "
            f"net_pnl_OW_regression={row.get('net_pnl_under_ow_regression_eval')}, "
            f"net_pnl_reduced_form={row.get('net_pnl_under_reduced_form_eval')}"
        )
    if len(fitted_proxy_df):
        row = fitted_proxy_df.iloc[0]
        print(
            "Fitted-regression proxy strategy: "
            f"net_pnl={row.get('total_net_pnl_fitted_proxy')}, "
            f"fitted_cost={row.get('total_fitted_proxy_cost')}, "
            f"turnover={row.get('total_turnover_fitted_proxy')}, "
            f"max_participation_rate={row.get('max_participation_rate_fitted_proxy')}"
        )
        print(f"Fitted proxy summary: {out_dir / 'all_pairs_fitted_proxy_summary.csv'}")
        if processed_pair_ids:
            print(f"Fitted proxy trades: {out_dir / f'pair_{processed_pair_ids[0]}' / 'fitted_proxy_strategy_trades.csv'}")
    forced_rows = stress_df.loc[
        stress_df.get("scenario_name", pd.Series(dtype=str)).astype(str).str.contains("forced_liq", na=False)
    ] if len(stress_df) and "scenario_name" in stress_df.columns else pd.DataFrame()
    if len(forced_rows):
        print("Forced liquidation stress summary:")
        cols = [
            "scenario_name",
            "liquidation_trigger_mode",
            "liquidation_mode",
            "net_pnl_under_ow_regression_eval",
            "net_pnl_under_reduced_form_eval",
            "number_of_liquidation_events",
            "number_no_liquidation_events",
            "liquidation_event_rate_realized",
            "number_with_residual_after_first_liquidation",
            "number_with_overnight_residual",
            "hard_block_cap_violation_rate",
        ]
        print(forced_rows[[c for c in cols if c in forced_rows.columns]].to_string(index=False))
        if processed_pair_ids:
            stress_dir = out_dir / f"pair_{processed_pair_ids[0]}" / "stress"
            event_files = sorted(stress_dir.glob("forced_liq_*_events.csv"))
            if event_files:
                print("Forced liquidation event files:")
                for path in event_files:
                    print(f"- {path}")
    print(strategy_df.to_string(index=False))
    print(wrong_df.to_string(index=False))
    if len(fitted_proxy_df):
        print(fitted_proxy_df.to_string(index=False))


if __name__ == "__main__":
    main()
