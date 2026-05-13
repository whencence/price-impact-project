"""Configuration for section 2.7 sensitivity and stress testing."""

from dataclasses import dataclass
from datetime import time

import pandas as pd


@dataclass(frozen=True)
class SensitivityConfig:
    """Sensitivity-analysis parameter grids."""

    rho_grid: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50)
    horizon_minutes_grid: tuple[float, ...] = (1.0, 5.0, 10.0)
    alpha_decay_half_life_grid_minutes: tuple[float, ...] = (1.0, 5.0, 30.0, 60.0)
    impact_lambda_multiplier_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    impact_half_life_grid_minutes: tuple[float, ...] = (1.0, 5.0, 30.0, 60.0)
    baseline_rho: float = 0.10
    baseline_horizon_minutes: float = 5.0
    baseline_alpha_decay_half_life_minutes: float = 5.0
    baseline_impact_lambda_multiplier: float = 1.0
    baseline_impact_half_life_minutes: float = 5.0
    max_scenarios: int | None = None
    save_scenario_trades: bool = False

    def __post_init__(self) -> None:
        """Validate sensitivity grids."""

        if any(rho <= 0 or rho >= 1 for rho in self.rho_grid):
            raise ValueError("all rho values must be in (0, 1)")
        for name in [
            "horizon_minutes_grid",
            "alpha_decay_half_life_grid_minutes",
            "impact_half_life_grid_minutes",
        ]:
            if any(value <= 0 for value in getattr(self, name)):
                raise ValueError(f"all {name} values must be positive")
        if any(value <= 0 for value in self.impact_lambda_multiplier_grid):
            raise ValueError("all impact lambda multipliers must be positive")
        if self.max_scenarios is not None and self.max_scenarios <= 0:
            raise ValueError("max_scenarios must be None or positive")


@dataclass(frozen=True)
class StressTestConfig:
    """Stress-test scenario configuration."""

    signal_delay_minutes: float = 1.0
    forced_liquidation_time: str = "12:00:00"
    resume_after_liquidation: bool = False
    wrong_impact_lambda_multiplier_assumed: float = 1.0
    wrong_impact_lambda_multiplier_true: float = 2.0
    wrong_impact_half_life_assumed_minutes: float = 5.0
    wrong_impact_half_life_true_minutes: float = 30.0
    run_signal_delay: bool = True
    run_forced_liquidation: bool = True
    run_wrong_impact_params: bool = True
    run_wrong_impact_model: bool = False

    def __post_init__(self) -> None:
        """Validate stress-test settings."""

        if self.signal_delay_minutes < 0:
            raise ValueError("signal_delay_minutes must be non-negative")
        if pd.isna(pd.to_datetime(self.forced_liquidation_time, errors="coerce")):
            raise ValueError("forced_liquidation_time must be parseable")
        for name in ["wrong_impact_lambda_multiplier_assumed", "wrong_impact_lambda_multiplier_true"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ["wrong_impact_half_life_assumed_minutes", "wrong_impact_half_life_true_minutes"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def liquidation_clock_time(self) -> time:
        """Return forced liquidation time as a Python time."""

        return pd.to_datetime(self.forced_liquidation_time).time()


@dataclass(frozen=True)
class StressRunConfig:
    """Top-level stress-run configuration."""

    alpha_input_path: str = "outputs/alphas/strategy_alpha_input_h5m_rho010_H5m.csv"
    output_dir: str = "outputs/stress"
    strategy_model: str = "OW"
    baseline_alpha_col: str = "alpha_for_strategy"
    random_seed: int = 42
    allow_low_scaling_coverage: bool = False
    run_reduced_form_if_available: bool = False
    fitted_model_params_path: str | None = None

    def __post_init__(self) -> None:
        """Validate top-level run settings."""

        if self.strategy_model not in {"OW", "reduced_form"}:
            raise ValueError("strategy_model must be 'OW' or 'reduced_form'")
        if not self.baseline_alpha_col:
            raise ValueError("baseline_alpha_col must be non-empty")
