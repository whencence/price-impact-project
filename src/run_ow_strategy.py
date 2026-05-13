"""Run the OW strategy from the validated synthetic alpha input."""

from pathlib import Path

import pandas as pd

from src.ow_strategy import attach_scaling_factors, run_ow_strategy
from src.scaling_factors import compute_or_load_scaling_factors
from src.strategy_config import OWStrategyConfig
from src.strategy_metrics import save_strategy_metrics, save_strategy_plots
from src.strategy_validation import save_strategy_validation_report, validate_ow_strategy_output


def find_project_root(start: Path) -> Path:
    """Find repository root from a script or notebook working directory."""

    start = start.resolve()
    if start.name == "src":
        start = start.parent
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not find project root containing data/ and src/")


def main() -> None:
    """Run OW strategy, metrics, validation, and plots."""

    project_root = find_project_root(Path.cwd())
    input_path = project_root / "outputs" / "alphas" / "strategy_alpha_input_h5m_rho010_H5m.csv"
    output_dir = project_root / "outputs" / "strategy" / "ow"
    scaling_path = project_root / "outputs" / "scaling" / "scaling_factors_20d.csv"
    bin_dir = project_root / "data" / "binSamples"
    output_dir.mkdir(parents=True, exist_ok=True)

    scaling = compute_or_load_scaling_factors(
        bin_dir=bin_dir,
        output_path=scaling_path,
        force_recompute=False,
        window_days=20,
    )

    config = OWStrategyConfig(
        scaling_factors_path=str(scaling_path.relative_to(project_root)),
        use_scaling_factors=True,
        fallback_sigma=None,
        fallback_ADV=None,
        impact_model_type="linear",
        impact_lambda=1.0,
        impact_half_life_minutes=5.0,
        max_abs_position=None,
        max_abs_trade=None,
        liquidate_at_close=True,
    )

    alpha_input = pd.read_csv(input_path)
    alpha_with_scaling = attach_scaling_factors(alpha_input, scaling, config)
    rows_with_scaling = int(alpha_with_scaling["has_scaling"].sum())
    trades = run_ow_strategy(alpha_with_scaling, config)
    trades.to_csv(output_dir / "ow_strategy_trades.csv", index=False)
    daily, summary = save_strategy_metrics(trades, output_dir)
    save_strategy_plots(trades, daily, output_dir)
    checks = validate_ow_strategy_output(trades, config)
    save_strategy_validation_report(checks, trades, summary, output_dir)

    print(f"Saved scaling factors to {scaling_path}")
    print(f"Rows with scaling: {rows_with_scaling} / {len(alpha_with_scaling)}")
    print(f"Rows skipped due to missing scaling: {int(trades['skipped_due_to_missing_scaling'].sum())}")
    print(f"Saved OW trades to {output_dir / 'ow_strategy_trades.csv'}")
    print("WARNING: impact_lambda remains a placeholder until calibrated OW parameters are provided.")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(pd.DataFrame.from_dict(checks, orient="index").to_string())


if __name__ == "__main__":
    main()
