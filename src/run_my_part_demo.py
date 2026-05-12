"""Runnable demo for sections 2.4, 2.5, and 2.7."""

from pathlib import Path

import pandas as pd

from src.alpha import build_all_alphas
from src.config import AlphaConfig, StrategyConfig, StressTestConfig
from src.metrics_minimal import compare_stress_results, summarize_performance
from src.mock_data import make_mock_impact_params, make_mock_market_data
from src.strategy import run_strategy
from src.stress_tests import run_all_stress_tests


def _print_summary(name: str, trades: pd.DataFrame) -> None:
    metrics = summarize_performance(trades)
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def main() -> None:
    """Run a local deterministic demo and save CSV outputs."""

    output_dir = Path("outputs") / "my_part_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    market_data = make_mock_market_data()
    tickers = sorted(market_data["ticker"].unique().tolist())
    impact_params = make_mock_impact_params(tickers)

    alpha_config = AlphaConfig(
        intraday_horizon_steps=5,
        intraday_strength=2.0,
        intraday_decay=0.02,
        overnight_strength=1.0,
        alpha_clip=0.02,
    )
    strategy_config = StrategyConfig(
        risk_aversion=1.0,
        max_position=50_000.0,
        max_trade_size=5_000.0,
        liquidation_at_close=True,
        position_penalty=0.05,
        trade_penalty=0.0,
        target_position_scale=1_000_000.0,
    )
    stress_config = StressTestConfig(
        signal_delay_steps=5,
        forced_liquidation_time="10:00",
        wrong_model_assumed="OW",
        wrong_model_true="advanced",
    )

    alphas = build_all_alphas(market_data, alpha_config)
    alphas.to_csv(output_dir / "alphas.csv", index=False)

    trades_intraday = run_strategy(alphas, impact_params, "OW", "alpha_intraday", strategy_config)
    trades_overnight = run_strategy(alphas, impact_params, "OW", "alpha_overnight", strategy_config)
    trades_combined = run_strategy(alphas, impact_params, "OW", "alpha_combined", strategy_config)

    trades_intraday.to_csv(output_dir / "trades_intraday.csv", index=False)
    trades_overnight.to_csv(output_dir / "trades_overnight.csv", index=False)
    trades_combined.to_csv(output_dir / "trades_combined.csv", index=False)

    stress_results = run_all_stress_tests(
        alphas, impact_params, "alpha_combined", strategy_config, stress_config
    )
    stress_comparison = compare_stress_results(stress_results)
    stress_comparison.to_csv(output_dir / "stress_comparison.csv")

    _print_summary("Intraday alpha strategy", trades_intraday)
    _print_summary("Overnight alpha strategy", trades_overnight)
    _print_summary("Combined alpha strategy", trades_combined)
    print("\nStress comparison")
    print(stress_comparison.round(6))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()

