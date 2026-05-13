"""Validation checks for reduced-form strategy outputs."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.reduced_form_config import ReducedFormStrategyConfig


def _result(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}


def validate_reduced_form_strategy_output(
    trades_df: pd.DataFrame,
    config: ReducedFormStrategyConfig,
) -> dict[str, dict[str, str]]:
    """Validate reduced-form strategy output mechanics."""

    checks: dict[str, dict[str, str]] = {}
    required = {
        "signed_volume_trade",
        "position_before",
        "position_after",
        "target_impact",
        "lambda_t",
        "local_volume_state_v",
        "net_pnl",
        "dt_minutes",
        "beta",
        "gamma_prime_t",
        "impact_before_trade",
        "impact_after_trade",
        "gross_pnl",
        "signed_impact_cost",
        "quadratic_impact_cost",
        "participation_rate",
        "has_scaling",
        "target_formula_fallback",
    }
    missing = required.difference(trades_df.columns)
    checks["required_columns"] = _result("FAIL" if missing else "PASS", f"missing={sorted(missing)}")
    if missing:
        return checks

    critical = ["signed_volume_trade", "position_after", "target_impact", "lambda_t", "local_volume_state_v", "net_pnl"]
    nans = trades_df[critical].isna().sum().to_dict()
    checks["no_critical_nans"] = _result("PASS" if sum(nans.values()) == 0 else "FAIL", str(nans))
    checks["dt_non_negative"] = _result("PASS" if trades_df["dt_minutes"].min() >= 0 else "FAIL", f"min={trades_df['dt_minutes'].min():.3g}")
    checks["beta_positive"] = _result("PASS" if (trades_df["beta"] > 0).all() else "FAIL", f"min={trades_df['beta'].min():.3g}")
    checks["volume_state_floor"] = _result(
        "PASS" if (trades_df["local_volume_state_v"] >= config.min_volume_state - 1e-18).all() else "FAIL",
        f"min={trades_df['local_volume_state_v'].min():.3g}",
    )
    lambda_ok = trades_df["lambda_t"].replace([np.inf, -np.inf], np.nan).notna().all() and (trades_df["lambda_t"] > 0).all()
    checks["lambda_t_positive_finite"] = _result("PASS" if lambda_ok else "FAIL", "lambda_t finite and positive")
    gamma_ok = trades_df["gamma_prime_t"].replace([np.inf, -np.inf], np.nan).notna().all()
    checks["gamma_prime_finite"] = _result("PASS" if gamma_ok else "FAIL", "gamma_prime_t finite")
    target_ok = trades_df["target_impact"].replace([np.inf, -np.inf], np.nan).notna().all()
    checks["target_impact_finite"] = _result("PASS" if target_ok else "FAIL", "target impact finite")

    pos_error = (trades_df["position_after"] - (trades_df["position_before"] + trades_df["signed_volume_trade"])).abs().max()
    checks["position_dynamics"] = _result("PASS" if pos_error < 1e-8 else "FAIL", f"max_error={pos_error:.3g}")
    impact_error = (
        trades_df["impact_after_trade"]
        - (trades_df["impact_before_trade"] + trades_df["lambda_t"] * trades_df["signed_volume_trade"])
    ).abs().max()
    checks["impact_dynamics"] = _result("PASS" if impact_error < 1e-8 else "FAIL", f"max_error={impact_error:.3g}")

    max_decay_error = 0.0
    for _, group in trades_df.groupby(["stock", "date"], sort=False):
        prev = group["impact_after_trade"].shift(1).fillna(config.initial_impact)
        expected = np.exp(-group["beta"] * group["dt_minutes"]) * prev
        max_decay_error = max(max_decay_error, float((group["impact_before_trade"] - expected).abs().max()))
    checks["impact_decay"] = _result("PASS" if max_decay_error < 1e-8 else "FAIL", f"max_error={max_decay_error:.3g}")

    if config.liquidate_at_close:
        final_pos = trades_df.groupby(["stock", "date"], sort=False)["position_after"].tail(1).abs().max()
        checks["liquidation"] = _result("PASS" if final_pos < 1e-8 else "FAIL", f"max_final_abs_position={final_pos:.3g}")
    else:
        checks["liquidation"] = _result("WARN", "liquidate_at_close=False")

    wealth_error = (trades_df["net_pnl"] - (trades_df["gross_pnl"] - trades_df["signed_impact_cost"])).abs().max()
    checks["cost_to_pnl"] = _result("PASS" if wealth_error < 1e-8 else "FAIL", f"max_error={wealth_error:.3g}")
    min_quad = trades_df["quadratic_impact_cost"].min()
    checks["quadratic_cost_non_negative"] = _result("PASS" if min_quad >= -1e-12 else "FAIL", f"min={min_quad:.3g}")

    finite_part = trades_df["participation_rate"].replace([np.inf, -np.inf], np.nan).dropna()
    high_part = float((finite_part > 0.05).mean()) if len(finite_part) else 0.0
    checks["participation_rate"] = _result(
        "WARN" if high_part > 0.05 else "PASS",
        f"share rows >5%={high_part:.2%}; max={finite_part.max() if len(finite_part) else np.nan}",
    )
    scaling_pct = float(trades_df["has_scaling"].mean() * 100.0)
    checks["scaling_coverage"] = _result("WARN" if scaling_pct < 50 else "PASS", f"rows_with_scaling={scaling_pct:.2f}%")
    all_zero = bool((trades_df["signed_volume_trade"].abs() <= 1e-14).all())
    checks["strategy_not_all_zero"] = _result("WARN" if all_zero else "PASS", "all trades are zero" if all_zero else "non-zero trades present")
    fallback_share = float(trades_df["target_formula_fallback"].mean())
    checks["target_formula_fallback"] = _result(
        "WARN" if fallback_share > 0 else "PASS",
        f"fallback_share={fallback_share:.2%}",
    )
    if "target_impact_full_gamma" in trades_df.columns and "target_impact_heuristic" in trades_df.columns:
        diff = (trades_df["target_impact_full_gamma"] - trades_df["target_impact_heuristic"]).abs().replace([np.inf, -np.inf], np.nan)
        checks["full_vs_heuristic_diagnostic"] = _result("WARN", f"median_abs_diff={diff.median():.3g}; p95_abs_diff={diff.quantile(0.95):.3g}")
    return checks


def save_reduced_form_validation_report(
    checks: dict[str, dict[str, str]],
    trades_df: pd.DataFrame,
    summary_metrics: dict[str, Any],
    config: ReducedFormStrategyConfig,
    output_dir: Path,
) -> None:
    """Save reduced-form validation report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Reduced-Form Strategy Validation Report",
        "======================================",
        "",
        "Configuration:",
        f"- lambda_base: {config.lambda_base}",
        f"- beta: {config.effective_beta}",
        f"- impact_half_life_minutes: {config.impact_half_life_minutes}",
        f"- volume_window_minutes: {config.volume_window_minutes}",
        f"- translation_method: {config.translation_method}",
        f"- use_full_gamma_formula: {config.use_full_gamma_formula}",
        f"- use_slow_moving_liquidity_heuristic: {config.use_slow_moving_liquidity_heuristic}",
        "",
        "Checks:",
    ]
    for name, result in checks.items():
        lines.append(f"- {name}: {result['status']} - {result['message']}")
    lines.extend(["", "Summary metrics:"])
    for key, value in summary_metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Caveats:",
            "- lambda_base and beta are placeholders until teammate fitted parameters are provided.",
            "- This is a reduced-form AFS-style implementation, not the full nonlinear AFS model.",
            "- Baseline uses the slow-moving liquidity heuristic unless configured otherwise.",
            "- Outputs validate mechanics and are scenario-ready for section 2.7 stress tests.",
        ]
    )
    has_fail = any(result["status"] == "FAIL" for result in checks.values())
    lines.append("")
    lines.append("Conclusion: " + ("review failures before using outputs." if has_fail else "mechanics are valid subject to parameter caveats."))
    (output_dir / "reduced_form_validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
