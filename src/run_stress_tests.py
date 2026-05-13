"""Run section 2.7 sensitivity analysis and stress tests."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.ow_strategy import attach_scaling_factors
from src.scaling_factors import compute_or_load_scaling_factors
from src.strategy_config import OWStrategyConfig
from src.stress_config import SensitivityConfig, StressRunConfig, StressTestConfig
from src.stress_tests import load_alpha_input, run_all_stress_tests
from src.stress_validation import save_stress_validation_report, validate_stress_outputs


def find_project_root(start: Path) -> Path:
    """Find repository root from script or notebook working directory."""

    start = start.resolve()
    if start.name in {"src", "notebooks"}:
        start = start.parent
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not find project root containing data/ and src/")


def parse_args() -> ArgumentParser:
    """Build CLI parser."""

    parser = ArgumentParser(description="Run section 2.7 stress tests")
    parser.add_argument("--alpha-input", default="outputs/alphas/strategy_alpha_input_h5m_rho010_H5m.csv")
    parser.add_argument("--output-dir", default="outputs/stress")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--save-trades", action="store_true")
    parser.add_argument("--allow-low-scaling-coverage", action="store_true")
    return parser


def main() -> None:
    """Run stress framework and save outputs."""

    args = parse_args().parse_args()
    project_root = find_project_root(Path.cwd())
    alpha_path = Path(args.alpha_input)
    if not alpha_path.is_absolute():
        alpha_path = project_root / alpha_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scaling_path = project_root / "outputs" / "scaling" / "scaling_factors_20d.csv"
    scaling = compute_or_load_scaling_factors(
        project_root / "data" / "binSamples",
        scaling_path,
        force_recompute=False,
        window_days=20,
    )
    alpha = load_alpha_input(alpha_path)
    base_ow_config = OWStrategyConfig(
        scaling_factors_path=str(scaling_path.relative_to(project_root)),
        use_scaling_factors=True,
        impact_lambda=1.0,
        impact_half_life_minutes=5.0,
        liquidate_at_close=True,
    )
    alpha = attach_scaling_factors(alpha, scaling, base_ow_config)
    scaling_coverage = float(alpha["has_scaling"].mean())
    if scaling_coverage < 0.5 and not args.allow_low_scaling_coverage:
        print(f"WARNING: low scaling coverage ({scaling_coverage:.2%}); scenarios may skip trading.")

    sensitivity_config = SensitivityConfig(save_scenario_trades=args.save_trades)
    stress_config = StressTestConfig(
        run_signal_delay=not args.skip_stress,
        run_forced_liquidation=not args.skip_stress,
        run_wrong_impact_params=not args.skip_stress,
    )
    run_config = StressRunConfig(
        alpha_input_path=str(alpha_path.relative_to(project_root)) if alpha_path.is_relative_to(project_root) else str(alpha_path),
        output_dir=str(output_dir),
        allow_low_scaling_coverage=args.allow_low_scaling_coverage,
    )
    if args.skip_sensitivity:
        sensitivity_config = SensitivityConfig(
            rho_grid=(),
            horizon_minutes_grid=(),
            alpha_decay_half_life_grid_minutes=(),
            impact_lambda_multiplier_grid=(),
            impact_half_life_grid_minutes=(),
        )

    sensitivity, stress = run_all_stress_tests(alpha, base_ow_config, sensitivity_config, stress_config, run_config)
    checks = validate_stress_outputs(output_dir)
    save_stress_validation_report(checks, output_dir)

    print(f"Saved sensitivity summary to {output_dir / 'sensitivity_summary.csv'}")
    print(f"Saved stress summary to {output_dir / 'stress_summary.csv'}")
    print(f"Saved all scenario summary to {output_dir / 'all_scenarios_summary.csv'}")
    print(f"Saved validation report to {output_dir / 'stress_validation_report.txt'}")
    print(pd.DataFrame.from_dict(checks, orient="index").to_string())
    print("WARNING: impact lambda remains a placeholder until calibrated parameters are provided.")


if __name__ == "__main__":
    main()
