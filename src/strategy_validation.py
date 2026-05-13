"""Validation checks for OW strategy outputs."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ow_strategy import invert_normalized_trade, normalize_signed_volume
from src.strategy_config import OWStrategyConfig


def _result(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}


def _max_abs(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.abs().max()) if len(values) else 0.0


def validate_ow_strategy_output(trades_df: pd.DataFrame, config: OWStrategyConfig) -> dict[str, dict[str, str]]:
    """Validate OW strategy mechanics under sigma/ADV-normalized trade convention."""

    checks: dict[str, dict[str, str]] = {}
    required = {
        "sigma",
        "ADV",
        "has_scaling",
        "impact_model_type",
        "required_normalized_trade",
        "normalized_trade",
        "signed_volume",
        "trade",
        "position_before",
        "position_after",
        "impact_before_trade",
        "impact_after_trade",
        "target_impact",
        "gross_pnl",
        "quadratic_impact_cost_normalized",
        "signed_impact_cost_normalized",
        "net_pnl",
        "dt_minutes",
        "decay_factor",
        "alpha_dot",
        "participation_rate",
        "skipped_due_to_missing_scaling",
    }
    missing = required.difference(trades_df.columns)
    checks["required_columns"] = _result("FAIL" if missing else "PASS", f"missing={sorted(missing)}")
    if missing:
        return checks

    critical = ["signed_volume", "normalized_trade", "position_after", "impact_after_trade", "net_pnl"]
    nan_counts = trades_df[critical].isna().sum().to_dict()
    checks["no_critical_nans"] = _result("PASS" if sum(nan_counts.values()) == 0 else "FAIL", str(nan_counts))

    has_scaling_pct = float(trades_df["has_scaling"].mean() * 100.0)
    checks["scaling_coverage"] = _result(
        "PASS" if has_scaling_pct > 0 else "WARN",
        f"rows_with_scaling={has_scaling_pct:.2f}%",
    )

    missing_scaling = ~trades_df["has_scaling"].astype(bool)
    non_liq_missing = missing_scaling & ~trades_df["is_liquidation"].astype(bool)
    traded_without_scaling = _max_abs(trades_df.loc[non_liq_missing, "signed_volume"])
    checks["no_trading_without_scaling"] = _result(
        "PASS" if traded_without_scaling < 1e-12 else "FAIL",
        f"max_abs_signed_volume_without_scaling={traded_without_scaling:.3g}",
    )

    valid = trades_df["has_scaling"].astype(bool) & trades_df["sigma"].gt(0) & trades_df["ADV"].gt(0)
    if valid.any():
        recomputed = normalize_signed_volume(
            trades_df.loc[valid, "signed_volume"].to_numpy(),
            trades_df.loc[valid, "sigma"].to_numpy(),
            trades_df.loc[valid, "ADV"].to_numpy(),
            config.impact_model_type,
        )
        formula_error = float(np.nanmax(np.abs(recomputed - trades_df.loc[valid, "normalized_trade"].to_numpy())))
    else:
        formula_error = 0.0
    checks["normalized_trade_formula"] = _result(
        "PASS" if formula_error < 1e-10 else "FAIL",
        f"max_error={formula_error:.3g}; model={config.impact_model_type}",
    )

    if valid.any():
        sample = trades_df.loc[valid, ["normalized_trade", "sigma", "ADV", "signed_volume"]].head(10000)
        reconstructed = [
            invert_normalized_trade(qt, sigma, adv, config.impact_model_type)
            for qt, sigma, adv in zip(sample["normalized_trade"], sample["sigma"], sample["ADV"], strict=False)
        ]
        inverse_error = float(np.nanmax(np.abs(np.asarray(reconstructed) - sample["signed_volume"].to_numpy())))
    else:
        inverse_error = 0.0
    checks["inverse_trade_formula"] = _result(
        "PASS" if inverse_error < 1e-6 else "FAIL",
        f"sample_max_error={inverse_error:.3g}",
    )

    impact_error = (
        trades_df["impact_after_trade"]
        - (trades_df["impact_before_trade"] + config.impact_lambda * trades_df["normalized_trade"])
    ).abs().max()
    checks["impact_dynamics"] = _result("PASS" if impact_error < 1e-10 else "FAIL", f"max_error={impact_error:.3g}")

    pos_error = (trades_df["position_after"] - (trades_df["position_before"] + trades_df["signed_volume"])).abs().max()
    checks["position_dynamics"] = _result("PASS" if pos_error < 1e-8 else "FAIL", f"max_error={pos_error:.3g}")

    if config.liquidate_at_close:
        final_pos = trades_df.groupby(["stock", "date"], sort=False)["position_after"].tail(1).abs().max()
        checks["liquidation"] = _result("PASS" if final_pos < 1e-8 else "FAIL", f"max_final_abs_position={final_pos:.3g}")
    else:
        checks["liquidation"] = _result("WARN", "liquidate_at_close is False")

    wealth_error = (
        trades_df["net_pnl"] - (trades_df["gross_pnl"] - trades_df["signed_impact_cost_normalized"])
    ).abs().max()
    checks["wealth_consistency"] = _result("PASS" if wealth_error < 1e-10 else "FAIL", f"max_error={wealth_error:.3g}")

    cost_error = (
        trades_df["signed_impact_cost_normalized"]
        - (trades_df["impact_before_trade"] * trades_df["normalized_trade"] + trades_df["quadratic_impact_cost_normalized"])
    ).abs().max()
    checks["cost_decomposition"] = _result("PASS" if cost_error < 1e-10 else "FAIL", f"max_error={cost_error:.3g}")

    min_quad = trades_df["quadratic_impact_cost_normalized"].min()
    checks["quadratic_cost_non_negative"] = _result("PASS" if min_quad >= -1e-12 else "FAIL", f"min={min_quad:.3g}")
    checks["dt_non_negative"] = _result("PASS" if trades_df["dt_minutes"].min() >= 0 else "FAIL", f"min={trades_df['dt_minutes'].min():.3g}")
    decay_ok = trades_df["decay_factor"].between(0, 1).all()
    checks["decay_factor_range"] = _result("PASS" if decay_ok else "FAIL", "decay in [0,1]")

    finite_participation = trades_df["participation_rate"].replace([np.inf, -np.inf], np.nan).dropna()
    absurd_share = float((finite_participation > 0.05).mean()) if len(finite_participation) else 0.0
    checks["participation_rate_diagnostic"] = _result(
        "WARN" if absurd_share > 0.05 else "PASS",
        f"share rows participation > 5%={absurd_share:.2%}; max={finite_participation.max() if len(finite_participation) else np.nan}",
    )

    first = trades_df.groupby(["stock", "date"], sort=False).head(1)
    first_pos_ok = np.allclose(first["position_before"], config.initial_position)
    first_impact_ok = np.allclose(first["impact_before_trade"], config.initial_impact)
    checks["group_reset"] = _result("PASS" if first_pos_ok and first_impact_ok else "FAIL", "first row position/impact reset")
    alpha_dot_ok = np.allclose(first["alpha_dot"], 0.0)
    checks["alpha_dot_first_row"] = _result("PASS" if alpha_dot_ok else "FAIL", "first alpha_dot per group is zero")

    small_dot = trades_df["alpha_dot"].abs() <= trades_df["alpha_dot"].abs().quantile(0.25)
    pos_alpha = trades_df["alpha"] > 0
    share_positive_target = float((trades_df.loc[small_dot & pos_alpha, "target_impact"] > 0).mean())
    checks["trade_sign_diagnostic"] = _result("WARN", f"share positive target when alpha>0 and alpha_dot small={share_positive_target:.2%}")
    return checks


def save_strategy_validation_report(
    checks: dict[str, dict[str, str]],
    trades_df: pd.DataFrame,
    summary_metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save OW strategy validation report text."""

    output_dir = Path(output_dir)
    lines = [
        "OW Strategy Validation Report",
        "=============================",
        "",
        "Check statuses:",
    ]
    for name, result in checks.items():
        lines.append(f"- {name}: {result['status']} - {result['message']}")
    lines.extend(["", "Key summary metrics:"])
    for key, value in summary_metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Caveats:",
            "- impact_lambda is currently a placeholder.",
            "- The OW dynamics use normalized_trade = sigma * signed_volume / ADV for the linear model.",
            "- Position and trade are in signed volume units; trade is kept as an alias for signed_volume.",
            "- Cost columns are normalized diagnostics until the final calibrated cost convention is agreed.",
            "- Results validate code mechanics but are not final economic results until calibrated OW parameters are provided.",
        ]
    )
    critical_fail = any(result["status"] == "FAIL" for result in checks.values())
    lines.append("")
    lines.append("Conclusion: " + ("mechanics need review before use." if critical_fail else "OW strategy mechanics are valid."))
    (output_dir / "ow_strategy_validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
