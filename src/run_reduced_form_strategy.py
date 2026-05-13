"""Run the reduced-form AFS-style strategy from validated alpha input."""

from pathlib import Path

import pandas as pd

from src.reduced_form_config import ReducedFormStrategyConfig
from src.reduced_form_metrics import save_reduced_form_metrics, save_reduced_form_plots
from src.reduced_form_strategy import (
    attach_scaling_factors,
    merge_market_volume_from_bins,
    run_reduced_form_strategy,
)
from src.reduced_form_validation import (
    save_reduced_form_validation_report,
    validate_reduced_form_strategy_output,
)
from src.scaling_factors import compute_or_load_scaling_factors


def find_project_root(start: Path) -> Path:
    """Find repository root from a script or notebook working directory."""

    start = start.resolve()
    if start.name in {"src", "notebooks"}:
        start = start.parent
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not find project root containing data/ and src/")


def main() -> None:
    """Run reduced-form strategy, metrics, validation, and plots."""

    project_root = find_project_root(Path.cwd())
    input_path = project_root / "outputs" / "alphas" / "strategy_alpha_input_h5m_rho010_H5m.csv"
    scaling_path = project_root / "outputs" / "scaling" / "scaling_factors_20d.csv"
    bin_dir = project_root / "data" / "binSamples"
    output_dir = project_root / "outputs" / "strategy" / "reduced_form"
    output_dir.mkdir(parents=True, exist_ok=True)

    scaling = compute_or_load_scaling_factors(
        bin_dir=bin_dir,
        output_path=scaling_path,
        force_recompute=False,
        window_days=20,
    )

    config = ReducedFormStrategyConfig(
        lambda_base=1.0,
        impact_half_life_minutes=5.0,
        volume_window_minutes=5.0,
        use_slow_moving_liquidity_heuristic=True,
        use_full_gamma_formula=False,
        translation_method="position_formula",
        scaling_factors_path=str(scaling_path.relative_to(project_root)),
        use_scaling_factors=True,
        fallback_sigma=None,
        fallback_ADV=None,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"alpha input not found: {input_path}")
    alpha_input = pd.read_csv(input_path)
    if config.volume_col not in alpha_input.columns or "orderFlow" not in alpha_input.columns:
        alpha_input = merge_market_volume_from_bins(alpha_input, bin_dir, config)
    strategy_input = attach_scaling_factors(alpha_input, scaling, config)
    trades = run_reduced_form_strategy(strategy_input, config)
    trades.to_csv(output_dir / "reduced_form_strategy_trades.csv", index=False)
    daily, summary = save_reduced_form_metrics(trades, output_dir)
    save_reduced_form_plots(trades, daily, output_dir)
    checks = validate_reduced_form_strategy_output(trades, config)
    save_reduced_form_validation_report(checks, trades, summary, config, output_dir)

    print(f"Saved reduced-form trades to {output_dir / 'reduced_form_strategy_trades.csv'}")
    print(f"Saved metrics and validation report under {output_dir}")
    print("WARNING: lambda_base and beta are placeholders until fitted reduced-form parameters are provided.")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(pd.DataFrame.from_dict(checks, orient="index").to_string())


if __name__ == "__main__":
    main()
