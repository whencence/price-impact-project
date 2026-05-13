"""Validation for section 2.7 stress-test outputs."""

from pathlib import Path

import numpy as np
import pandas as pd


def _result(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}


def validate_stress_outputs(output_dir: Path) -> dict[str, dict[str, str]]:
    """Validate saved stress-test outputs."""

    output_dir = Path(output_dir)
    checks: dict[str, dict[str, str]] = {}
    required_files = [
        "sensitivity_summary.csv",
        "stress_summary.csv",
        "all_scenarios_summary.csv",
    ]
    missing = [name for name in required_files if not (output_dir / name).exists()]
    checks["required_summary_files"] = _result("FAIL" if missing else "PASS", f"missing={missing}")
    if missing:
        return checks

    sensitivity = pd.read_csv(output_dir / "sensitivity_summary.csv")
    stress = pd.read_csv(output_dir / "stress_summary.csv")
    all_summary = pd.read_csv(output_dir / "all_scenarios_summary.csv")
    has_baseline = "baseline_OW" in set(all_summary["scenario_name"])
    checks["baseline_exists"] = _result("PASS" if has_baseline else "FAIL", "baseline_OW present")
    duplicates = all_summary["scenario_name"].duplicated().sum()
    checks["unique_scenario_names"] = _result("PASS" if duplicates == 0 else "FAIL", f"duplicates={duplicates}")

    metric_cols = ["total_net_pnl", "daily_sharpe", "total_turnover", "max_drawdown", "max_abs_impact"]
    finite_share = all_summary[metric_cols].replace([np.inf, -np.inf], np.nan).notna().mean().mean()
    checks["key_metrics_present"] = _result("WARN" if finite_share < 0.5 else "PASS", f"finite_metric_share={finite_share:.2%}")

    for scenario_type, check_name in [
        ("signal_delay_stress", "delayed_scenario_exists"),
        ("forced_liquidation_stress", "forced_liquidation_scenario_exists"),
        ("wrong_impact_parameter_stress", "wrong_impact_scenario_exists"),
    ]:
        exists = scenario_type in set(stress["scenario_type"])
        checks[check_name] = _result("PASS" if exists else "WARN", f"{scenario_type} present={exists}")

    comparison_cols = [c for c in stress.columns if c.endswith("_vs_baseline") or c.startswith("pct_")]
    checks["baseline_comparison_columns"] = _result("PASS" if comparison_cols else "FAIL", f"comparison_cols={comparison_cols}")
    baseline = all_summary.loc[all_summary["scenario_name"].eq("baseline_OW")]
    if not baseline.empty and abs(float(baseline.iloc[0]["total_net_pnl"])) < 1e-12:
        pct_cols = [c for c in all_summary.columns if c.startswith("pct_net_pnl")]
        pct_all_nan = all_summary[pct_cols].isna().all().all() if pct_cols else True
        checks["zero_baseline_pct_degradation"] = _result("WARN" if pct_all_nan else "FAIL", "baseline net pnl is near zero; pct degradation should be NaN")

    skipped = output_dir / "skipped_scenarios.csv"
    checks["skipped_scenarios"] = _result("WARN" if skipped.exists() else "PASS", "skipped_scenarios.csv exists" if skipped.exists() else "no skipped scenarios file")
    fig_dir = output_dir / "figures"
    figures = list(fig_dir.glob("*.png")) if fig_dir.exists() else []
    checks["plots_exist"] = _result("PASS" if figures else "WARN", f"n_figures={len(figures)}")

    if (output_dir / "forced_liquidation_summary.csv").exists():
        forced = pd.read_csv(output_dir / "forced_liquidation_summary.csv")
        events = float(forced.get("number_of_liquidation_events", pd.Series([0])).iloc[0])
        checks["forced_liquidation_events"] = _result("WARN" if events == 0 else "PASS", f"events={events:g}")
    if (output_dir / "signal_delay_trades.csv").exists():
        delayed = pd.read_csv(output_dir / "signal_delay_trades.csv", nrows=10000)
        alpha_cols = [c for c in delayed.columns if c.startswith("alpha_delayed_")]
        all_zero = bool(alpha_cols and (delayed[alpha_cols[0]].abs() <= 1e-14).all())
        checks["signal_delay_alpha_nonzero"] = _result("WARN" if all_zero else "PASS", f"delayed_alpha_all_zero_sample={all_zero}")
    return checks


def save_stress_validation_report(checks: dict[str, dict[str, str]], output_dir: Path) -> None:
    """Save text validation report for stress outputs."""

    output_dir = Path(output_dir)
    lines = [
        "Section 2.7 Stress Testing Validation Report",
        "===========================================",
        "",
        "Checks:",
    ]
    for name, result in checks.items():
        lines.append(f"- {name}: {result['status']} - {result['message']}")

    all_path = output_dir / "all_scenarios_summary.csv"
    if all_path.exists():
        all_summary = pd.read_csv(all_path)
        lines.extend(["", f"Number of scenarios: {len(all_summary)}"])
        baseline = all_summary.loc[all_summary["scenario_name"].eq("baseline_OW")]
        if not baseline.empty:
            lines.append("Baseline metrics:")
            for col in ["total_net_pnl", "daily_sharpe", "total_turnover", "max_drawdown", "max_abs_impact"]:
                lines.append(f"- {col}: {baseline.iloc[0].get(col)}")
        if "total_net_pnl" in all_summary:
            worst = all_summary.sort_values("total_net_pnl").iloc[0]
            lines.append(f"Worst scenario by net PnL: {worst['scenario_name']} ({worst['total_net_pnl']})")
        if "max_drawdown" in all_summary:
            dd = all_summary.sort_values("max_drawdown").iloc[0]
            lines.append(f"Largest drawdown scenario: {dd['scenario_name']} ({dd['max_drawdown']})")
        if "total_turnover" in all_summary:
            turnover = all_summary.sort_values("total_turnover", ascending=False).iloc[0]
            lines.append(f"Highest turnover scenario: {turnover['scenario_name']} ({turnover['total_turnover']})")

    has_fail = any(result["status"] == "FAIL" for result in checks.values())
    lines.extend(
        [
            "",
            "Conclusion: " + ("review failed checks before report use." if has_fail else "section 2.7 outputs are report-ready subject to parameter caveats."),
        ]
    )
    (output_dir / "stress_validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
